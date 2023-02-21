import os
import logging
from dataclasses import dataclass


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


# trainer data config
DATA_DIR_PATH = "data"
DB_FILENAME = "production-database.20230207.sanitized.db"
DB_FILE_PATH = os.path.join(DATA_DIR_PATH, DB_FILENAME)
US_EURO_SIZE_THRESHOLD = 25

# SizeManager config
SIZING_DIR_NAME = "sizing_systems"
SIZING_SYSTEM_DIR_PATH = os.path.join(DATA_DIR_PATH, SIZING_DIR_NAME)


@dataclass
class ModelConfig:
    model_type: str = "classifier"
    test_size: float = 0.3
    embedding_dim: int = 4
    learning_rate: float = 0.005
    batch_size: int = 256
    checkpoint_path: str = os.path.join(os.getcwd(), "model_checkpoints")
    validation_split: float = 0.2
    epochs: int = 2_000
    fit_verbose: int = 1
    asym_loss_gamma: float = 0.5
    classification_loss: str = "sparse_categorical_crossentropy"
    embedding_func: str = "subtract"
    early_stopping__patience: int = 50
    early_stopping__restore_best_weights: bool = True
