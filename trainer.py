import os
import logging
import datetime
import sqlite3 as db
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


from pipelines import embedding_pipe
from query import QUERY

DATA_DIR_PATH = "data"
DB_FILENAME = "production-database.20230207.sanitized.db"
DB_FILE_PATH = os.path.join(DATA_DIR_PATH, DB_FILENAME)
US_EURO_SIZE_THRESHOLD = 25

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    test_size: float = 0.3
    embedding_dim: int = 2
    learning_rate: float = 0.0005
    batch_size: int = 1024
    checkpoint_path: str = (os.path.join(os.getcwd(), "model_checkpoints"),)
    validation_split: float = 0.4
    epochs: int = 2_000
    fit_verbose: int = 0


class Trainer:
    _query = QUERY
    _db_file_path = DB_FILE_PATH
    embedding_pipe = embedding_pipe
    _embedding_columns = ["user_id", "sku_id"]
    _target_columns = ["rating"]

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._conn: db.Connection = db.connect(self._db_file_path)
        self._cursor = self._conn.cursor()
        self.df: pd.DataFrame = None

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

    def transform_data(self):
        logger.info(f"transforming data: computing `sizing system` and `size_in`")
        self.df["sizing_system"] = self.df["size"].apply(self.get_sizing_system)
        self.df["size_in"] = self.df["size"].apply(self.convert_shoe_size_to_inches)

    def get_training_set(self):
        # Since we don't have date it is important to keep the dataframe sorted
        # ids are incremented in chronological order
        logger.info(
            f"sorting data by increasing (user-item) `id` number. (chronological order)"
        )
        self.df.sort_values("id", inplace=True)
        df_features = self.df[["user_id", "sku_id"]]
        y = self.df[["rating"]]

        return df_features, y

    def split_training_set(self, df_features, y, test_size=None):

        df_train, df_test, y_train, y_test = train_test_split(
            df_features,
            y,
            test_size=test_size if test_size else self.model_config.test_size,
            shuffle=True,
        )
        return df_train, df_test, y_train, y_test

    def compute_user_item_mat_df(self, values=["rating"]):
        self.user_sku_mat_df = self.df.pivot_table(
            index=["user_id"], columns=["sku_id"], values=values
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

    def initialize_model(self, embedding_df_train):
        tf.keras.backend.clear_session()
        tf.random.set_seed(123)

        # user pipeline
        user_input = layers.Input(shape=(1,), name="user")

        user_as_integer = layers.IntegerLookup(
            vocabulary=embedding_df_train["user_id"].unique()
        )(user_input)

        user_embedding = layers.Embedding(
            input_dim=embedding_df_train.shape[0] + 1,
            output_dim=self.model_config.embedding_dim,
            embeddings_regularizer="l2",
        )(user_as_integer)

        user_bias = tf.keras.layers.Embedding(
            input_dim=embedding_df_train.shape[0] + 1,
            output_dim=1,
            # embeddings_regularizer="l2",
        )(user_as_integer)

        # sku pipeline
        sku_input = layers.Input(shape=(1,), name="sku")
        sku_as_integer = layers.IntegerLookup(
            vocabulary=embedding_df_train["sku_id"].unique()
        )(sku_input)

        sku_embedding = layers.Embedding(
            input_dim=embedding_df_train.shape[0] + 1,
            output_dim=self.model_config.embedding_dim,
            embeddings_regularizer="l2",
        )(sku_as_integer)

        sku_bias = tf.keras.layers.Embedding(
            input_dim=embedding_df_train.shape[0] + 1,
            output_dim=1,
            # embeddings_regularizer="l2"
        )(sku_as_integer)

        # dot product
        dot = layers.Dot(axes=2)([user_embedding, sku_embedding])
        add = tf.keras.layers.Add()([dot, user_bias, sku_bias])
        flatten = layers.Flatten()(add)
        scale = tf.keras.layers.Lambda(lambda x: 4 * tf.nn.sigmoid(x) + 1)(flatten)

        # model input/output definition
        self.model = Model(inputs=[user_input, sku_input], outputs=scale)
        self.model.compile(
            loss="mae",
            metrics=["mse", "mae"],
            optimizer=tf.optimizers.Adam(learning_rate=self.model_config.learning_rate),
        )

    def get_model_inputs(self, embedding_df):
        Xs = [embedding_df[col].values.reshape(-1, 1) for col in self.embedding_columns]
        logger.info(f"Creating {len(Xs)} model inputs")
        return Xs

    def fit(self, Xs_train, y_train, tensorboard_on=False):
        self._model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_config.checkpoint_path,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
        )

        self._early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=50,
            verbose=2,
            restore_best_weights=True,
        )

        self._reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,
            patience=20,
            verbose=2,
            min_lr=self.model_config.learning_rate / 100,
        )

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        logger.info("model callbacks created")

        callbacks = [
            self._early_stopping,
            self._reduce_lr,
            # self._model_checkpoint_callback
        ]
        if tensorboard_on:
            callbacks.append(self._tensorboard_callback)

        results = self.model.fit(
            Xs_train,
            y_train,
            epochs=self.model_config.epochs,
            batch_size=self.model_config.batch_size,
            validation_split=self.model_config.validation_split,
            verbose=self.model_config.fit_verbose,
            callbacks=callbacks,
        )
        return results

    def evaluate_model(self, df_test, y_test):
        embedding_df_test = self.embedding_pipe.transform(
            df_test[self.embedding_columns]
        )
        Xs_test = self.get_model_inputs(embedding_df_test)
        logger.info("evaluating model")
        self.model.evaluate(Xs_test, y_test)

    def plot_results(self, results, plot_key="loss"):
        fig, ax = plt.subplots(1, figsize=(7, 7))
        ax.plot(results.history[plot_key], label=plot_key)
        ax.plot(results.history[f"val_{plot_key}"], "-", label=f"val_{plot_key}")
        ax.set_xlabel("Epoch")
        plt.legend()
        plt.grid(True)
        return ax


if __name__ == "__main__":
    model_config = ModelConfig()
    trainer = Trainer(model_config)
    trainer.load_data()
    trainer.transform_data()
    df_features, y = trainer.get_training_set()
    df_train, df_test, y_train, y_test = trainer.split_training_set(df_features, y)
    embedding_df_train = trainer.embedding_pipe.fit_transform(
        df_train[trainer.embedding_columns]
    )
    #
    Xs_train = trainer.get_model_inputs(embedding_df_train)

    # model evaluation
    trainer.initialize_model(embedding_df_train)
    results = trainer.fit(Xs_train, y_train)

    # model evaluation
    trainer.evaluate_model(df_test, y_test)
