import os
import logging

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
