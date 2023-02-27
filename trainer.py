from typing import Any
from tqdm import tqdm
import datetime
import sqlite3 as db
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch import optim

from pipelines import (
    EMBEDDING_COLUMNS,
    TARGET_CATEGORIES,
    TARGET_COLUMN,
    embedding_pipe,
    user_features_pipe,
    sku_features_pipe,
)
from query import QUERY

from config import (
    get_logger,
    DB_FILE_PATH,
    US_EURO_SIZE_THRESHOLD,
    RegressorConfig,
)
from torch_models import RatingRegressor, get_history_df

from datasets import SizeSquirrelRecommenderDataset as ProjectDataset

logger = get_logger(__name__)


class Trainer:
    _query = QUERY
    _db_file_path = DB_FILE_PATH
    embedding_pipe = embedding_pipe
    user_features_pipe = user_features_pipe
    sku_features_pipe = sku_features_pipe
    _embedding_columns = EMBEDDING_COLUMNS["user"] + EMBEDDING_COLUMNS["sku"]
    _target_column = TARGET_COLUMN

    def __init__(self, model_config: RegressorConfig):
        self.model_config = model_config
        logger.info(f"Creating trainer with ModelConfig: {self.model_config}")
        self._conn: db.Connection = db.connect(self._db_file_path)
        self._cursor = self._conn.cursor()
        self.user_sku_df: pd.DataFrame = None
        self.model_callbacks = []

    @property
    def embedding_columns(self):
        return self._embedding_columns

    @property
    def target_column(self):
        return self._target_column

    def load_data(self):
        logger.info(f"using file at '{self._db_file_path}'")
        self.user_sku_df = pd.read_sql_query(self._query, self._conn)
        # logger.info(f"Loading data with query: {QUERY}'")
        logger.info(
            f"Loaded data: {self.user_sku_df.shape[1]} columns and {self.user_sku_df.shape[0]:,} rows"
        )

    @staticmethod
    def get_sizing_system(size):
        return "US" if size < US_EURO_SIZE_THRESHOLD else "EURO"

    @staticmethod
    def convert_shoe_size_to_inches(size):
        if size is None:
            return None
        size = float(size)

        # takes an agnostic shoe size, stores it as inches
        size_in_inches = None

        if size > US_EURO_SIZE_THRESHOLD:
            # we're inputing a EUR size
            size_in_inches = ((size - 31.333) / 1.333 + 1) / 3 + 7.333
        else:
            # we're inputing a US size
            size_in_inches = (size * 0.333) + 7.333
        return size_in_inches

    @staticmethod
    def compute_sku_id(df, validate_brand=False):
        # check whether this definition makes sense
        if validate_brand:
            max_brand_id_per_brand_name = (
                df.groupby(["brand_name"])
                .agg(brand_id_count=("brand_id", "nunique"))["brand_id_count"]
                .max()
            )
            assert max_brand_id_per_brand_name == 1

        return (
            df["brand_name"].astype(str)
            + "__"
            + df["model"]
            + "__"
            + df["shoe_gender"]
            + "__"
            + df["size"].astype(str)
        )

    def transform_data(self):
        logger.info(f"transforming data: computing `sizing system` and `size_in`")
        self.user_sku_df["sizing_system"] = self.user_sku_df["size"].apply(
            self.get_sizing_system
        )
        self.user_sku_df["size_in"] = self.user_sku_df["size"].apply(
            self.convert_shoe_size_to_inches
        )
        self.user_sku_df["sku_id"] = self.compute_sku_id(self.user_sku_df)
        self.user_sku_df["rating"] = self.user_sku_df["rating"].astype(float)

    def get_split_training_set(
        self, test_size=None, stratify_split=True, chronological_split=False
    ):
        if chronological_split:
            self.user_sku_df.sort_values("id", inplace=True)
        df_train, df_test = train_test_split(
            self.user_sku_df,
            test_size=test_size if test_size else self.model_config.test_size,
            stratify=self.user_sku_df[self._target_column] if stratify_split else None,
            shuffle=not (chronological_split),
            random_state=1234,
        )
        return df_train, df_test

    def fit_pipelines(self, df_train):
        # features
        logger.info("fitting `embedding_pipe`")
        self.embedding_pipe.fit(df_train)
        logger.info("fitting `user_features_pipe`")
        self.user_features_pipe.fit(df_train)
        logger.info("fitting `sku_features_pipe`")
        self.sku_features_pipe.fit(df_train)
        logger.info("fitting `target_pipe`")
        self.model_config.target_pipe.fit(df_train[self.target_column])

    def get_embedding_inputs(self, df):
        embedding_df = self.embedding_pipe.transform(df)
        embedding_inputs = {
            col: torch.tensor(embedding_df[col].values).unsqueeze(1)
            for col in self.embedding_columns
        }
        embedding_vocabs = {
            col: embedding_df[col].unique() for col in self.embedding_columns
        }
        logger.info(f"Creating {len(embedding_inputs.keys())} model inputs")
        return embedding_inputs, embedding_vocabs

    def get_user_features_inputs(self, df):
        user_features_inputs = torch.tensor(
            self.user_features_pipe.transform(df).values
        )
        return user_features_inputs

    def get_sku_features_inputs(self, df):
        sku_features_inputs = torch.tensor(self.sku_features_pipe.transform(df).values)
        return sku_features_inputs

    def get_targets(self, df):
        y = self.model_config.target_pipe.transform(df[self.target_column])
        return torch.tensor(y)

    def get_inputs_dict(self, df):
        inputs_dict, embedding_vocabs = self.get_embedding_inputs(df)
        inputs_dict["user_features"] = self.get_user_features_inputs(df)
        inputs_dict["sku_features"] = self.get_sku_features_inputs(df)
        self.user_features_dim = inputs_dict["user_features"].shape[1]
        self.sku_features_dim = inputs_dict["sku_features"].shape[1]
        return inputs_dict, embedding_vocabs

    def create_dummy_classifier(self, targets_train):
        # broken
        pass

    def create_dummy_regressor(self, targets_train):
        mean_proba = targets_train.mean().T.reshape(1, -1)

        def constant_output(x):
            batch_size = tf.shape(x)[0]
            return tf.tile(
                tf.constant(mean_proba, dtype=torch.float32), [batch_size, 1]
            )

        user_input = tf.keras.layers.Input(shape=(1,), name="user_id")
        out = tf.keras.layers.Lambda(constant_output)(user_input)
        logger.info("Dummy regressor created: always predicting the mean value")
        self.model = tf.keras.Model(inputs=[user_input], outputs=out)

    @staticmethod
    def get_class_weight(df_train):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=TARGET_CATEGORIES, y=df_train["rating"]
        )
        return {idx: class_weight for idx, class_weight in enumerate(class_weights)}

    def create_torch_model(self, embedding_vocabs, user_features_dim, sku_features_dim):

        self.model = RatingRegressor(
            self.model_config.embedding_dim,
            embedding_vocabs,
            user_features_dim,
            sku_features_dim,
        ).to(self.model_config.device)
        logger.info("RatingPredictor` model created")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.model_config.learning_rate
        )
        logger.info(f"optimizer created")

    def train_one_epoch(self, training_loader):
        sum_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(training_loader):
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.model_config.loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            sum_loss += loss.item()
        epoch_mean_loss = sum_loss / (batch_idx + 1)
        return epoch_mean_loss

    def validate_one_epoch(self, validation_loader):
        sum_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_loader):
                # Make predictions for this batch
                outputs = self.model(inputs)
                # Compute the loss
                loss = self.model_config.loss_fn(outputs, targets)
                # Gather data and report
                sum_loss += loss.item()
            epoch_mean_loss = sum_loss / (batch_idx + 1)
            return epoch_mean_loss

    def fit_with_cross_validation(
        self,
        inputs_train,
        targets_train,
        embedding_vocabs,
        n_folds,
    ):
        torch.manual_seed(42)
        user_features_dim = inputs_train["user_features"].shape[1]
        sku_features_dim = inputs_train["sku_features"].shape[1]

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=self.model_config.shuffle_stratified_kfold,
        )
        logger.info(
            f"StratifiedKFold cross-validator with {n_folds} folds created. shuffle: {self.model_config.shuffle_stratified_kfold}"
        )
        results = {
            key: get_history_df(n_folds, self.model_config.max_epochs)
            for key in ["train_loss", "val_loss"]
        }
        for n_fold, (train_idxs, val_idxs) in enumerate(
            skf.split(targets_train, targets_train)
        ):
            fold_str = f"Fold #{n_fold + 1}"
            logger.info(f"{fold_str:.^50}")
            train_dataset = ProjectDataset(
                inputs_train,
                targets_train,
                idxs=train_idxs,
                device=self.model_config.device,
            )
            logger.info(f"Training dataset with {len(train_idxs)} rows created")
            val_dataset = ProjectDataset(
                inputs_train,
                targets_train,
                idxs=val_idxs,
                device=self.model_config.device,
            )
            logger.info(f"Validation dataset with {len(val_idxs)} rows created")
            train_loader = DataLoader(
                train_dataset, batch_size=self.model_config.batch_size
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.model_config.batch_size
            )

            self.create_torch_model(
                embedding_vocabs, user_features_dim, sku_features_dim
            )
            with tqdm(range(self.model_config.max_epochs), mininterval=2) as epoch_bar:
                for n_epoch in epoch_bar:
                    epoch_mean_train_loss = self.train_one_epoch(train_loader)
                    results["train_loss"].iloc[n_epoch, n_fold] = epoch_mean_train_loss
                    epoch_mean_val_loss = self.validate_one_epoch(val_loader)
                    results["val_loss"].iloc[n_epoch, n_fold] = epoch_mean_val_loss
                    epoch_bar.set_description(
                        f"training loss: {epoch_mean_train_loss:.2f} - validation loss: {epoch_mean_val_loss:.2f}"
                    )
        return results

    def evaluate_model(self, df_test):
        inputs_test, _ = self.get_inputs_dict(df_test)
        targets_test = self.get_targets(df_test)
        logger.info("evaluating model")
        self.model.evaluate(inputs_test, targets_test)

    def append_predictions(self, df_test):
        inputs_test, _ = self.get_inputs_dict(df_test)
        pred_test = self.model.predict(inputs_test)
        if self.model_config.model_type == "classifier":
            y_pred = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1, pred_test)
            df_test["predicted_rating"] = y_pred
            rating_proba_columns = [f"proba_rating_{n}" for n in range(1, 6)]
            df_test[rating_proba_columns] = pred_test
        else:
            df_test["predicted_rating"] = pred_test
            df_test["rounded_predicted_rating"] = df_test["predicted_rating"].apply(
                np.round
            )
        return df_test

    def plot_results(self, results, plot_key="loss"):
        if not (isinstance(results, list)):
            results = [results]
        _, ax = plt.subplots(1, figsize=(7, 7))
        for idx, these_results in enumerate(results):

            line_objs = ax.plot(
                these_results.history[plot_key],
                label=f"{plot_key} / {idx + 1}",
                linestyle="--",
            )

            color = line_objs[-1].get_color()
            ax.plot(
                these_results.history[f"val_{plot_key}"],
                label=f"val_{plot_key} / {idx + 1}",
                linestyle="-",
                color=color,
            )
        ax.set_xlabel("Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()
        return ax

    @staticmethod
    def plot_confusion_matrix(df_dict, fig_height=5, query_str=None):
        n_df = len(df_dict)
        fig, axs = plt.subplots(1, n_df, figsize=(fig_height * n_df, fig_height))
        if not (isinstance(axs, np.ndarray)):
            axs = [axs]
        for ax, (name, df) in zip(axs, df_dict.items()):
            if query_str is not None:
                _df = df.query(query_str)
            else:
                _df = df
            cm = confusion_matrix(
                _df["rating"].astype(int), _df["predicted_rating"].astype(int)
            )
            ax = sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Predicted Rating")
            ax.set_xticklabels(range(1, 6))
            ax.set_ylabel("Actual Rating")
            ax.set_yticklabels(range(1, 6))
            ax.invert_yaxis()
            ax.set_title(name)
        plt.show()
        return fig, axs


if __name__ == "__main__":
    model_config = RegressorConfig(
        learning_rate=0.05,
        max_epochs=100,
        embedding_dim=7,
        batch_size=512,
        combine_func="dot",
    )
    trainer = Trainer(model_config)
    # load data
    trainer.load_data()
    trainer.transform_data()
    df_train, df_test = trainer.get_split_training_set()
    trainer.fit_pipelines(df_train)
    inputs_train, embedding_vocabs = trainer.get_inputs_dict(df_train)
    targets_train = trainer.get_targets(df_train)
    results = trainer.fit_with_cross_validation(
        inputs_train,
        targets_train,
        embedding_vocabs,
        n_folds=5,
    )
