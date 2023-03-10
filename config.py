import os
from dataclasses import dataclass
import logging
from dataclasses import dataclass
import tensorflow as tf
from pipelines import TARGET_CATEGORIES, classifier_target_pipe, regressor_target_pipe

# logging
LOGGING_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
LOGGING_DATE_FORMAT = "%Y-%m-%d:%H:%M:%S"


def get_logger(name):
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging.INFO,
    )
    logger = logging.getLogger(name)
    return logger


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
    early_stopping__patience: int = 50
    early_stopping__restore_best_weights: bool = True


# trainer data config
DATA_DIR_PATH = "data"
DB_FILENAME = "production-database.20230207.sanitized.db"
DB_FILE_PATH = os.path.join(DATA_DIR_PATH, DB_FILENAME)
US_EURO_SIZE_THRESHOLD = 25

# SizeManager config
SIZING_DIR_NAME = "sizing_systems"
SIZING_SYSTEM_DIR_PATH = os.path.join(DATA_DIR_PATH, SIZING_DIR_NAME)


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


@dataclass
class ClassifierConfig:
    target_pipe = classifier_target_pipe
    test_size: float = 0.3
    embedding_dim: int = 4
    learning_rate: float = 0.005
    batch_size: int = 256
    checkpoint_path: str = os.path.join(os.getcwd(), "model_checkpoints")
    validation_split: float = 0.2
    epochs: int = 2_000
    fit_verbose: int = 1
    embedding_func: str = "subtract"
    early_stopping__patience: int = 50
    early_stopping__restore_best_weights: bool = True
    # model_type specific
    model_type: str = "classifier"
    loss = "sparse_categorical_crossentropy"
    output_activation = "softmax"
    tracked_metrics = [
        "sparse_categorical_crossentropy",
        "kullback_leibler_divergence",
        "sparse_categorical_accuracy",
    ]


@dataclass
class RegressorConfig:
    target_pipe = regressor_target_pipe
    test_size: float = 0.3
    embedding_dim: int = 5
    learning_rate: float = 0.005
    batch_size: int = 256
    checkpoint_path: str = os.path.join(os.getcwd(), "model_checkpoints")
    validation_split: float = 0.2
    epochs: int = 2_000
    fit_verbose: int = 1
    embedding_func: str = "subtract"
    early_stopping__patience: int = 50
    early_stopping__restore_best_weights: bool = True
    # model_type specific
    model_type: str = "regressor"
    loss = AsymmetricsMeanSquaredError(0.5)
    output_activation = tf.keras.layers.Lambda(
        lambda x: (len(TARGET_CATEGORIES) - 1) * tf.nn.sigmoid(x) + 1
    )
    tracked_metrics = [
        "mean_absolute_error",
        tf.keras.metrics.RootMeanSquaredError(
            name="root_mean_squared_error", dtype=None
        ),
    ]
