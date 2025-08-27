# rich/loguru setup
from loguru import logger
import sys

def setup_logging(level="INFO"):
    logger.remove()
    logger.add(sys.stdout, level=level, enqueue=True, backtrace=False, diagnose=False)
    return logger
