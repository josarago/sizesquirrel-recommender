import os
import logging
import datetime
import sqlite3 as db
import numpy as np
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from pipelines import EMBEDDING_COLUMNS, TARGET_COLUMNS, embedding_pipe, target_pipe
from query import QUERY

DATA_DIR_PATH = "data"
DB_FILENAME = "production-database.20230207.sanitized.db"
DB_FILE_PATH = os.path.join(DATA_DIR_PATH, DB_FILENAME)
US_EURO_SIZE_THRESHOLD = 25

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    test_size: float = 0.3
    embedding_dim: int = 4
    learning_rate: float = 0.005
    batch_size: int = 256
    checkpoint_path: str = os.path.join(os.getcwd(), "model_checkpoints")
    validation_split: float = 0.2
    epochs: int = 2_000
    fit_verbose: int = 1
    asym_loss_gamma: float = 0.5
    classification_loss: str = "categorical_crossentropy"
    embedding_func: str = "subtract"


class AsymmetricsMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, gamma=0.5):
        super().__init__()
        self._gamma = gamma

    def call(self, y_true, y_pred):
        """
        if alpha = 0.5 this is equivalent to the MSE Loss
        """
        asym_factor = tf.abs(
            tf.constant(self._gamma)
            - tf.cast(tf.math.greater(y_pred, y_true), tf.float32)
        )
        return tf.reduce_mean(asym_factor * tf.math.square(y_pred - y_true), axis=-1)


class Trainer:
    _query = QUERY
    _db_file_path = DB_FILE_PATH
    embedding_pipe = embedding_pipe
    target_pipe = target_pipe
    _embedding_columns = EMBEDDING_COLUMNS
    _target_columns = TARGET_COLUMNS

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        logger.info(f"Creating trainer with ModelConfig: {self.model_config}")
        self._conn: db.Connection = db.connect(self._db_file_path)
        self._cursor = self._conn.cursor()
        self.df: pd.DataFrame = None
        self.model_callbacks = []

    @property
    def embedding_columns(self):
        return self._embedding_columns

    @property
    def target_columns(self):
        return self._target_columns

    def load_data(self):
        logger.info(f"using file at '{self._db_file_path}'")
        self.df = pd.read_sql_query(self._query, self._conn)
        # logger.info(f"Loading data with query: {QUERY}'")
        logger.info(
            f"Loaded data: {self.df.shape[1]} columns and {self.df.shape[0]:,} rows"
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
        self.df["sizing_system"] = self.df["size"].apply(self.get_sizing_system)
        self.df["size_in"] = self.df["size"].apply(self.convert_shoe_size_to_inches)
        self.df["sku_id"] = self.compute_sku_id(self.df)

    def get_split_training_set(self, test_size=None, chronological_split=False):
        if chronological_split:
            self.df.sort_values("id", inplace=True)
        df_train, df_test = train_test_split(
            self.df,
            test_size=test_size if test_size else self.model_config.test_size,
            shuffle=not (chronological_split),
            random_state=1234,
        )
        return df_train, df_test

    def fit_pipelines(self, df_train):
        self.embedding_pipe.fit(df_train)
        self.target_pipe.fit(df_train)

    def get_embedding_inputs(self, df):
        embedding_df = self.embedding_pipe.transform(df)
        inputs = {
            col: embedding_df[col].values.reshape(-1, 1)
            for col in self.embedding_columns
        }
        vocabularies = {
            col: embedding_df[col].unique() for col in self.embedding_columns
        }
        logger.info(f"Creating {len(inputs.keys())} model inputs")
        return inputs, vocabularies

    def get_targets(self, df):
        y = self.target_pipe.transform(df)
        return y

    def compute_user_item_mat_df(self):
        self.user_sku_mat_df = self.df.pivot_table(
            index=["user_id"], columns=["sku_id"], values=self._target_columns
        )
        logger.info(f"{self.user_sku_mat_parsity:.2%}")

    @property
    def all_users(self):
        return self.df["user_id"].unique()

    @property
    def all_skus(self):
        return self.df["sku_id"].unique()

    @property
    def user_sku_mat_parsity(self):
        return self.user_sku_mat_df.notnull().sum().sum() / (
            self.user_sku_mat_df.shape[0] * self.user_sku_mat_df.shape[1]
        )

    def create_dummy_classifier(self, targets_train):
        mean_proba = targets_train.mean().values.T.reshape(1, -1)

        def constant_output(x):
            batch_size = tf.shape(x)[0]
            return tf.tile(tf.constant(mean_proba, dtype=tf.float32), [batch_size, 1])

        user_input = layers.Input(shape=(1,), name="user_id")
        out = layers.Lambda(constant_output)(user_input)
        self.model = Model(inputs=[user_input], outputs=out)

    def create_classifier(self, inputs_train, vocabularies):
        # user pipeline
        user_input = layers.Input(shape=(1,), name="user_id")

        user_as_integer = layers.IntegerLookup(vocabulary=vocabularies["user_id"])(
            user_input
        )

        user_embedding = layers.Embedding(
            input_dim=inputs_train["user_id"].shape[0] + 1,
            output_dim=self.model_config.embedding_dim,
            embeddings_regularizer="l2",
        )(user_as_integer)

        user_bias = tf.keras.layers.Embedding(
            input_dim=inputs_train["user_id"].shape[0] + 1,
            output_dim=1,
            embeddings_regularizer="l2",
        )(user_as_integer)

        # sku pipeline
        sku_input = layers.Input(shape=(1,), name="sku_id")
        sku_as_integer = layers.IntegerLookup(vocabulary=vocabularies["sku_id"])(
            sku_input
        )

        sku_embedding = layers.Embedding(
            input_dim=inputs_train["sku_id"].shape[0] + 1,
            output_dim=self.model_config.embedding_dim,
            embeddings_regularizer="l2",
        )(sku_as_integer)

        sku_bias = tf.keras.layers.Embedding(
            input_dim=inputs_train["sku_id"].shape[0] + 1,
            output_dim=1,
            embeddings_regularizer="l2",
        )(sku_as_integer)

        if self.model_config.embedding_func == "subtract":
            # dot product
            logger.info("Model will subtract embeddings")
            subtracted = layers.Subtract()([user_embedding, sku_embedding])
            added = layers.Add()([subtracted, user_bias, sku_bias])

        elif self.model_config.embedding_func == "add":
            # - original model ~LightFM
            logger.info("Model will add embeddings")
            dot = layers.Dot(axes=2)([user_embedding, sku_embedding])
            added = tf.keras.layers.Add()([dot, user_bias, sku_bias])
        else:
            raise ValueError("model_type can be either `subtract` or `add`")
        flatten = layers.Flatten()(added)
        # hidden0 = layers.Dense(11, activation="relu")(flatten)
        # hidden0 = layers.Dense(7, activation="relu")(flatten)
        out = layers.Dense(5, activation="softmax")(flatten)
        # model input/output definition
        self.model = Model(inputs=[user_input, sku_input], outputs=out)

    def compile_model(self):
        self.model.compile(
            loss=self.model_config.classification_loss,
            metrics=[
                "categorical_crossentropy",
                "kullback_leibler_divergence",
                "categorical_accuracy",
            ],
            optimizer=tf.optimizers.Adam(learning_rate=self.model_config.learning_rate),
        )

    def load_model(self, inputs_train, vocabularies):
        self.create_classifier(inputs_train, vocabularies)

    def create_call_backs(
        self,
        early_stopping=True,
        reduce_lr=False,
        tensorboard_on=False,
        model_checkpoint=False,
    ):
        if early_stopping:
            early_stopping_kwargs = dict(
                monitor="val_loss",
                patience=10,
                verbose=2,
                restore_best_weights=True,
            )
            logger.info(
                f"Adding EarlyStopping callback with parameters: {early_stopping_kwargs}"
            )
            self.model_callbacks.append(
                tf.keras.callbacks.EarlyStopping(**early_stopping_kwargs)
            )
        if reduce_lr:
            reduce_lr_kwargs = dict(
                monitor="val_loss",
                factor=0.5,
                patience=20,
                verbose=2,
                min_lr=self.model_config.learning_rate / 100,
            )
            logger.info(
                f"Adding ReduceLROnPlateau callback with parameters: {reduce_lr_kwargs}"
            )
            self.model_callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(**reduce_lr_kwargs)
            )

        if tensorboard_on:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1
            )
            logger.info("model callbacks created")
            self.model_callbacks.append(self._tensorboard_callback)

        if model_checkpoint:
            self.model_callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_config.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                )
            )

    def fit(self, Xs_train, y_train):

        tf.keras.backend.clear_session()
        tf.random.set_seed(123)

        results = self.model.fit(
            Xs_train,
            y_train,
            epochs=self.model_config.epochs,
            batch_size=self.model_config.batch_size,
            validation_split=self.model_config.validation_split,
            verbose=self.model_config.fit_verbose,
            callbacks=self.model_callbacks,
        )
        return results

    def fit_with_cross_validation(self, inputs_train, targets_train, n_splits=5):
        results = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=321)
        for idx, (train_idx, val_idx) in enumerate(
            skf.split(inputs_train["user_id"], inputs_train["sku_id"], targets_train)
        ):
            logger.info(f"Split #{idx + 1}")
            _Xs_train = [
                inputs_train["user_id"][train_idx, :],
                inputs_train["sku_id"][train_idx, :],
            ]
            _y_train = targets_train.iloc[train_idx, :]

            _Xs_val = [
                inputs_train["user_id"][val_idx, :],
                inputs_train["sku_id"][val_idx, :],
            ]
            _y_val = targets_train.iloc[val_idx, :]
            tf.keras.backend.clear_session()
            tf.random.set_seed(345)

            _results = self.model.fit(
                _Xs_train,
                _y_train,
                epochs=self.model_config.epochs,
                batch_size=self.model_config.batch_size,
                validation_data=(_Xs_val, _y_val),
                verbose=self.model_config.fit_verbose,
                callbacks=self.model_callbacks,
            )
            results.append(_results)
        return results

    def evaluate_model(self, df_test):
        inputs_test, _ = self.get_embedding_inputs(df_test)
        targets_test = self.get_targets(df_test)
        logger.info("evaluating model")
        self.model.evaluate(inputs_test, targets_test)

    def append_predictions(self, df_test):
        inputs_test, _ = self.get_embedding_inputs(df_test)
        pred_test = self.model.predict(inputs_test)
        y_pred = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1, pred_test)
        df_test["predicted_rating"] = y_pred
        rating_proba_columns = [f"proba_rating_{n}" for n in range(1, 6)]
        df_test[rating_proba_columns] = pred_test
        return df_test

    def plot_results(self, results, plot_key="loss"):
        if not (isinstance(results, list)):
            results = [results]
        _, ax = plt.subplots(1, figsize=(7, 7))
        for idx, these_results in enumerate(results):

            ax.plot(
                these_results.history[plot_key],
                label=f"{plot_key} / {idx + 1}",
                color="k",
            )
            ax.plot(
                these_results.history[f"val_{plot_key}"],
                "-",
                label=f"val_{plot_key} / {idx + 1}",
                color="r",
            )
        ax.set_xlabel("Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()
        return ax

    @staticmethod
    def plot_confusion_matrix(df_dict, fig_height=5):
        n_df = len(df_dict)
        fig, axs = plt.subplots(1, n_df, figsize=(fig_height * n_df, fig_height))
        if not (isinstance(axs, np.ndarray)):
            axs = [axs]
        for ax, (name, df) in zip(axs, df_dict.items()):

            cm = confusion_matrix(
                df["predicted_rating"].astype(int), df["rating"].astype(int)
            )
            ax = sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Actual Rating")
            ax.set_xticklabels(range(1, 6))
            ax.set_ylabel("Predicted Rating")
            ax.set_yticklabels(range(1, 6))
            ax.invert_yaxis()
            ax.set_title(name)
        plt.show()
        return fig, axs


if __name__ == "__main__":
    model_config = ModelConfig(
        fit_verbose=0,
        classification_loss="categorical_crossentropy",
        embedding_func="subtract",
    )
    trainer = Trainer(model_config)
    # load data
    trainer.load_data()
    trainer.transform_data()
    # df_features, df_targets = trainer.get_training_set()
    df_train, df_test = trainer.get_split_training_set()
    trainer.fit_pipelines(df_train)
    #
    inputs_train, vocabularies = trainer.get_embedding_inputs(df_train)
    targets_train = trainer.get_targets(df_train)
    # initialize and train model
    trainer.create_classifier(inputs_train, vocabularies)

    # trainer.create_dummy_classifier(targets_train)
    trainer.compile_model()
    trainer.create_call_backs()
    results = trainer.fit_with_cross_validation(inputs_train, targets_train, n_splits=3)
    # trainer.evaluate_model(df_test)
