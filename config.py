import os
from dataclasses import dataclass
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


DATA_DIR_PATH = "data"
DB_FILENAME = "production-database.20230207.sanitized.db"
DB_FILE_PATH = os.path.join(DATA_DIR_PATH, DB_FILENAME)

US_EURO_SIZE_THRESHOLD = 32


def get_logger(use_file:bool = False, name=None):  
	logger = getLogger(__name__)
	logger.setLevel(INFO)
	handler1 = StreamHandler()
	handler1.setFormatter(Formatter('[%(asctime)s](%(name)s) | %(levelname)s: %(message)s'))
	logger.addHandler(handler1)
	return logger

@dataclass
class ModelConfig:
	test_size: float = 0.3
	embedding_dim: int = 5
	learning_rate: float = 0.0005
	batch_size: int = 512
	checkpoint_path: str = "/tmp/model_checkpoint"
	validation_split: float = 0.3
	epochs: int = 2_000
	fit_verbose: int = 1

