import os
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
    if use_file:
        filename = OUTPUT_DIR +'train'
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger
