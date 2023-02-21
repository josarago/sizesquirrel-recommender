import os

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
from sklearn.utils.class_weight import compute_class_weight


import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
import keras_tuner as kt

from pipelines import (
    EMBEDDING_COLUMNS,
    TARGET_CATEGORIES,
    TARGET_COLUMN,
    embedding_pipe,
    target_pipe,
    user_features_pipe,
    sku_features_pipe,
)
from query import QUERY

from config import get_logger, DB_FILE_PATH, US_EURO_SIZE_THRESHOLD

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    test_size: float = 0.3
    embedding_dim: int = 4
    learning_rate: float = 0.005
    batch_size: int = 256
    checkpoint_path: str = os.path.join(os.getcwd(), "model_checkpoints")
    validation_split: float = 0.2
    epochs: int = 2_000
    fit_verbose: int = 0
    asym_loss_gamma: float = 0.5
    classification_loss: str = "categorical_crossentropy"
    embedding_func: str = "subtract"
    early_stopping__patience: int = 50
    early_stopping__restore_best_weights: bool = False


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
    user_features_pipe = user_features_pipe
    sku_features_pipe = sku_features_pipe
    _embedding_columns = EMBEDDING_COLUMNS
    _target_column = TARGET_COLUMN

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        logger.info(f"Creating trainer with ModelConfig: {self.model_config}")
        self._conn: db.Connection = db.connect(self._db_file_path)
        self._cursor = self._conn.cursor()
        self.user_sku_df: pd.DataFrame = None
        self.model_callbacks = []
        self.embedding_vocabs = None
        self.tuner: kt.RandomSearch = None

    @property
    def embedding_columns(self):
        return self._embedding_columns

    @property
    def target_columns(self):
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
        self.embedding_pipe.fit(df_train)
        self.user_features_pipe.fit(df_train)
        self.sku_features_pipe.fit(df_train)
        self.target_pipe.fit(df_train[self._target_column])

    def get_embedding_inputs(self, df):
        embedding_df = self.embedding_pipe.transform(df)
        embedding_inputs = {col: embedding_df[[col]] for col in self.embedding_columns}
        embedding_vocabs = {
            col: embedding_df[col].unique() for col in self.embedding_columns
        }
        logger.info(f"Creating {len(embedding_inputs.keys())} model inputs")
        return embedding_inputs, embedding_vocabs

    def get_user_features_inputs(self, df):
        user_features_inputs = self.user_features_pipe.transform(df)
        return user_features_inputs

    def get_sku_features_inputs(self, df):
        sku_features_inputs = self.sku_features_pipe.transform(df)
        return sku_features_inputs

    def get_targets(self, df):
        y = self.target_pipe.transform(df[self._target_column])
        return y

    def get_inputs_dict(self, df):
        inputs_dict, embedding_vocabs = self.get_embedding_inputs(df)
        inputs_dict["user_features"] = self.get_user_features_inputs(df)
        inputs_dict["sku_features"] = self.get_sku_features_inputs(df)
        self.user_features_dim = inputs_dict["user_features"].shape[1]
        self.sku_features_dim = inputs_dict["sku_features"].shape[1]
        self.embedding_vocabs = embedding_vocabs
        return inputs_dict, embedding_vocabs

    def create_dummy_classifier(self, targets_train):
        mean_proba = targets_train.mean().T.reshape(1, -1)

        def constant_output(x):
            batch_size = tf.shape(x)[0]
            return tf.tile(tf.constant(mean_proba, dtype=tf.float32), [batch_size, 1])

        user_input = layers.Input(shape=(1,), name="user_id")
        out = layers.Lambda(constant_output)(user_input)
        self.model = Model(inputs=[user_input], outputs=out)

    def create_classifier(self, hp):
        # sku pipeline
        hp_embedding_dim = hp.Int("embedding_dim", min_value=3, max_value=8, step=1)
        if hasattr(self, "model"):
            del model
        sku_id_input = layers.Input(shape=(1,), name="sku_id")
        sku_as_integer = layers.IntegerLookup(
            vocabulary=self.embedding_vocabs["sku_id"]
        )(sku_id_input)

        sku_embedding = layers.Embedding(
            input_dim=len(self.embedding_vocabs["sku_id"]) + 1,
            output_dim=hp_embedding_dim,
            # embeddings_regularizer=regularizers.L2(l2=0.02),
        )(sku_as_integer)

        flattened_sku_embedding = layers.Flatten()(sku_embedding)

        sku_features_input = layers.Input(
            shape=(self.sku_features_dim,), name="sku_features"
        )

        concat_sku = layers.Concatenate(axis=1)(
            [flattened_sku_embedding, sku_features_input]
        )

        # user pipeline
        user_id_input = layers.Input(shape=(1,), name="user_id")

        user_as_integer = layers.IntegerLookup(
            vocabulary=self.embedding_vocabs["user_id"]
        )(user_id_input)

        user_embedding = layers.Embedding(
            input_dim=len(self.embedding_vocabs["user_id"]) + 1,
            output_dim=hp_embedding_dim,
            # embeddings_regularizer=regularizers.L2(l2=0.02),
        )(user_as_integer)

        flattened_user_embedding = layers.Flatten()(user_embedding)

        user_features_input = layers.Input(
            shape=(self.user_features_dim,), name="user_features"
        )

        concat_user = layers.Concatenate(axis=1)(
            [flattened_user_embedding, user_features_input]
        )

        denser_user = layers.Dense(concat_sku.shape[1])(concat_user)

        if self.model_config.embedding_func == "subtract":
            # dot product
            logger.info("Model uses `layers.Subtract`")
            tmp_out = layers.Subtract()([denser_user, concat_sku])
            # added = layers.Add()([subtracted, user_bias, sku_bias])

        elif self.model_config.embedding_func == "dot":
            # - original model ~LightFM
            logger.info("Model will use `layers.Dot`")
            # dot = layers.Dot(axes=2)([user_embedding, sku_embedding])
            # added = tf.keras.layers.Add()([dot, user_bias, sku_bias])
            # tmp_out = layers.Flatten()(added)
        else:
            raise ValueError("`embedding_func` can be either `subtract` or `dot`")

        hp_hidden_layer_dim = hp.Int(
            "hidden_layer_dim", min_value=4, max_value=10, step=2
        )
        hidden = layers.Dense(hp_hidden_layer_dim, activation="relu")(tmp_out)
        hp_dropout_rate = hp.Choice("dropout_rate", values=[0.4, 0.5, 0.6, 0.7])
        dropout = layers.Dropout(hp_dropout_rate)(hidden)
        # hidden0 = layers.Dense(7, activation="relu")(flatten)
        out = layers.Dense(5, kernel_regularizer="l2", activation="softmax")(dropout)
        # model input/output definition
        model = Model(
            inputs=[
                user_id_input,
                user_features_input,
                sku_id_input,
                sku_features_input,
            ],
            outputs=out,
        )

        hp_learning_rate = hp.Float(
            "learning_rate", 1e-5, 1e-1, sampling="log", default=1e-3
        )
        model.compile(
            loss=self.model_config.classification_loss,
            metrics=[
                "categorical_crossentropy",
                "kullback_leibler_divergence",
                "categorical_accuracy",
            ],
            optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
        )
        return model

    # def compile_model(self):
    #     self.model.compile(
    #         loss=self.model_config.classification_loss,
    #         metrics=[
    #             "categorical_crossentropy",
    #             "kullback_leibler_divergence",
    #             "categorical_accuracy",
    #         ],
    #         optimizer=tf.optimizers.Adam(learning_rate=self.model_config.learning_rate),
    #     )

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
                patience=self.model_config.early_stopping__patience,
                verbose=2,
                restore_best_weights=self.model_config.early_stopping__restore_best_weights,
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

    @staticmethod
    def get_class_weight(df_train):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=TARGET_CATEGORIES, y=df_train["rating"]
        )
        return {idx: class_weight for idx, class_weight in enumerate(class_weights)}

    def fit(
        self,
        inputs_dict,
        targets_train,
        embedding_vocabs,
        validation_data=None,
        class_weight=None,
    ):

        tf.keras.backend.clear_session()
        tf.random.set_seed(123)
        self.create_classifier()

        self.compile_model()
        self.create_call_backs()

        if validation_data is not None:
            validation_split = None
        else:
            validation_split = self.model_config.validation_split
        results = self.model.fit(
            inputs_dict,
            targets_train,
            epochs=self.model_config.epochs,
            batch_size=self.model_config.batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=self.model_config.fit_verbose,
            callbacks=self.model_callbacks,
            class_weight=class_weight,
        )
        return results

    def initialize_tuner(self):
        self.tuner = kt.RandomSearch(
            self.create_classifier,
            objective="categorical_crossentropy",
            max_trials=100,
            seed=10101010,
            directory="tuner",
            project_name="sizesquirrel-recommender",
        )
        logger.info("Hyperband tuner created")

    def search(self, inputs_train, targets_train):
        self.tuner.search(
            inputs_train,
            targets_train,
            validation_split=0.2,
            callbacks=[self.model_callbacks],
            verbose=1,
        )
        self.model = self.tuner.get_best_models(num_models=1)

    def fit_with_cross_validation(
        self,
        inputs_dict,
        targets_train,
        embedding_vocabs,
        n_splits,
        class_weight=None,
    ):
        results = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=321)
        labels_train = self.target_pipe.inverse_transform(targets_train)
        for idx, (train_idx, val_idx) in enumerate(
            skf.split(labels_train, labels_train)
        ):
            inputs_train_fold = {
                key: df.iloc[train_idx, :] for key, df in inputs_dict.items()
            }
            targets_train_fold = targets_train[train_idx, :]

            inputs_val_fold = {
                key: df.iloc[val_idx, :] for key, df in inputs_dict.items()
            }

            targets_val_fold = targets_train[val_idx, :]
            tf.keras.backend.clear_session()
            tf.random.set_seed(345)
            _results = self.fit(
                inputs_train_fold,
                targets_train_fold,
                embedding_vocabs,
                validation_data=(inputs_val_fold, targets_val_fold),
                class_weight=class_weight,
            )
            logger.info(f"Model trained on split #{idx + 1}")
            results.append(_results)
        return results

    def evaluate_model(self, df_test):
        inputs_test, _ = self.get_inputs_dict(df_test)
        targets_test = self.get_targets(df_test)
        logger.info("evaluating model")
        self.model.evaluate(inputs_test, targets_test)

    def append_predictions(self, df_test):
        inputs_test, _ = self.get_inputs_dict(df_test)
        pred_test = self.model.predict(inputs_test)
        y_pred = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1, pred_test)
        df_test["predicted_rating"] = y_pred
        rating_proba_columns = [f"proba_rating_{n}" for n in range(1, 6)]
        df_test[rating_proba_columns] = pred_test
        return df_test

    def plot_results(self, results, plot_key="loss"):
        training_color = np.array([0, 0, 0])
        loss_color = np.array([1, 0, 0])
        if not (isinstance(results, list)):
            results = [results]
        _, ax = plt.subplots(1, figsize=(7, 7))
        for idx, these_results in enumerate(results):

            ax.plot(
                these_results.history[plot_key],
                label=f"{plot_key} / {idx + 1}",
                color=training_color,
            )
            ax.plot(
                these_results.history[f"val_{plot_key}"],
                "-",
                label=f"val_{plot_key} / {idx + 1}",
                color=loss_color,
            )

            training_color = 1 - ((1 - training_color) * 0.8)
            loss_color = 1 - ((1 - loss_color) * 0.8)
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
    model_config = ModelConfig(
        fit_verbose=0,
        learning_rate=0.00005,
        epochs=1_000,
        classification_loss="categorical_crossentropy",
        embedding_func="subtract",
        embedding_dim=3,
        batch_size=1024,
    )
    trainer = Trainer(model_config)
    # load data
    trainer.load_data()
    trainer.transform_data()
    # df_features, df_targets = trainer.get_training_set()
    df_train, df_test = trainer.get_split_training_set()
    trainer.fit_pipelines(df_train)
    #
    # embedding_inputs_train, vocabularies = trainer.get_embedding_inputs(df_train)
    # user_features_inputs_train = trainer.get_user_features_inputs(df_train)
    inputs_train, embedding_vocabs = trainer.get_inputs_dict(df_train)
    targets_train = trainer.get_targets(df_train)

    trainer.initialize_tuner()
    trainer.create_call_backs()
    trainer.search(inputs_train, targets_train)
    # results = trainer.fit(
    #     inputs_train,
    #     targets_train,
    #     embedding_vocabs,
    #     class_weight=None,
    # )

    # trainer.evaluate_model(df_test)
