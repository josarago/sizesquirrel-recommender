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

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

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

from config import (
    get_logger,
    DB_FILE_PATH,
    US_EURO_SIZE_THRESHOLD,
    ClassifierConfig,
    RegressorConfig,
)

logger = get_logger(__name__)


class Trainer:
    _query = QUERY
    _db_file_path = DB_FILE_PATH
    embedding_pipe = embedding_pipe
    user_features_pipe = user_features_pipe
    sku_features_pipe = sku_features_pipe
    target_pipe = target_pipe
    _embedding_columns = EMBEDDING_COLUMNS["user"] + EMBEDDING_COLUMNS["sku"]
    _target_column = TARGET_COLUMN

    def __init__(self, model_config):
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
        self.target_pipe.fit(df_train[self.target_columns])

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
        y = self.target_pipe.transform(df[self.target_columns])
        if self.model_config.model_type == "regressor":
            return y.astype(float)
        return y

    def get_inputs_dict(self, df):
        inputs_dict, embedding_vocabs = self.get_embedding_inputs(df)
        # inputs_dict["user_features"] = self.get_user_features_inputs(df)
        # inputs_dict["sku_features"] = self.get_sku_features_inputs(df)
        # self.user_features_dim = inputs_dict["user_features"].shape[1]
        # self.sku_features_dim = inputs_dict["sku_features"].shape[1]
        return inputs_dict, embedding_vocabs

    def create_dummy_classifier(self, targets_train):
        mean_proba = targets_train.mean().T.reshape(1, -1)

        def constant_output(x):
            batch_size = tf.shape(x)[0]
            return tf.tile(tf.constant(mean_proba, dtype=tf.float32), [batch_size, 1])

        user_input = layers.Input(shape=(1,), name="user_id")
        out = layers.Lambda(constant_output)(user_input)
        self.model = Model(inputs=[user_input], outputs=out)

    def create_model(self, vocabularies):
        # sku pipeline
        if hasattr(self, "model"):
            del self.model
        model_inputs = dict()
        as_integer = dict()
        embeddings = dict()
        biases = dict()

        for name, vocabulary in vocabularies.items():
            model_inputs[name] = layers.Input(shape=(1,), name=name)
            as_integer[name] = layers.IntegerLookup(vocabulary=vocabulary)(
                model_inputs[name]
            )
            embeddings[name] = layers.Embedding(
                input_dim=len(vocabulary) + 1,
                output_dim=self.model_config.embedding_dim,
                # embeddings_regularizer=regularizers.L2(l2=0.02),
            )(as_integer[name])

            biases[name] = layers.Embedding(
                input_dim=len(vocabulary) + 1,
                output_dim=1,
                # embeddings_regularizer=regularizers.L2(l2=0.02),
            )(as_integer[name])

        # we sum all user embeddings
        user_pooled_embedding = tf.keras.layers.Add()(
            [
                layer
                for name, layer in embeddings.items()
                if name in EMBEDDING_COLUMNS["user"]
            ]
        )

        user_pooled_bias = tf.keras.layers.Add()(
            [
                layer
                for name, layer in biases.items()
                if name in EMBEDDING_COLUMNS["user"]
            ]
        )

        sku_pooled_embedding = tf.keras.layers.Add()(
            [
                layer
                for name, layer in embeddings.items()
                if name in EMBEDDING_COLUMNS["sku"]
            ]
        )

        sku_pooled_bias = tf.keras.layers.Add()(
            [
                layer
                for name, layer in biases.items()
                if name in EMBEDDING_COLUMNS["sku"]
            ]
        )

        dot = tf.keras.layers.Dot(axes=2, name="dot")(
            [user_pooled_embedding, sku_pooled_embedding]
        )
        add = tf.keras.layers.Add(name="add_pooled_embeddings_and_biases")(
            [dot, user_pooled_bias, sku_pooled_bias]
        )
        flatten = tf.keras.layers.Flatten(name="flatten")(add)

        # sku_features_input = layers.Input(
        #     shape=(self.sku_features_dim,), name="sku_features"
        # )

        # user_features_input = layers.Input(
        #     shape=(self.user_features_dim,), name="user_features"
        # )

        # concat_sku = layers.Concatenate(axis=1)(
        #     [flattened_sku_embedding, sku_features_input]
        # )

        # sku_bias = tf.keras.layers.Embedding(
        #     input_dim=len(vocabularies["sku_id"]) + 1,
        #     output_dim=1,
        #     embeddings_regularizer="l2",
        # )(sku_as_integer)

        # user pipeline
        # if self.model_config.embedding_func == "subtract":
        #     # dot product
        #     logger.info("Model uses `layers.Subtract`")
        #     tmp_out = layers.Subtract()([denser_user, concat_sku])
        #     # added = layers.Add()([subtracted, user_bias, sku_bias])

        # elif self.model_config.embedding_func == "dot":
        #     # - original model ~LightFM
        #     logger.info("Model will use `layers.Dot`")
        #     # dot = layers.Dot(axes=2)([user_embedding, sku_embedding])
        #     # added = tf.keras.layers.Add()([dot, user_bias, sku_bias])
        #     # tmp_out = layers.Flatten()(added)
        # else:
        #     raise ValueError("`embedding_func` can be either `subtract` or `dot`")

        # hidden = layers.Dense(11, activation="relu")(tmp_out)
        # dropout = layers.Dropout(0.5)(tmp_out)
        # hidden0 = layers.Dense(7, activation="relu")(flatten)
        out = layers.Dense(
            5, kernel_regularizer="l2", activation=self.model_config.output_activation
        )(flatten)
        # model input/output definition
        self.model = Model(
            inputs=model_inputs,
            outputs=out,
        )

    def compile_model(self):

        self.model.compile(
            loss=self.model_config.loss,
            metrics=self.model_config.tracked_metrics,
            optimizer=tf.optimizers.Adam(learning_rate=self.model_config.learning_rate),
        )

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
        self.create_model(embedding_vocabs)
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
        for idx, (train_idx, val_idx) in enumerate(
            skf.split(targets_train, targets_train)
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
                "-",
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
        fit_verbose=0,
        learning_rate=0.00005,
        epochs=1_000,
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
    # trainer.create_model(embedding_vocabs)
    results = trainer.fit(
        inputs_train,
        targets_train,
        embedding_vocabs,
        class_weight=None,
    )
    # print("something")
    trainer.evaluate_model(df_test)
