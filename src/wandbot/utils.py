import datetime
import logging
import os
from typing import Any


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    )
    logger = logging.getLogger(name)
    return logger


class Timer:
    def __init__(self) -> None:
        self.start = datetime.datetime.utcnow()
        self.stop = self.start

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop = datetime.datetime.utcnow()

    @property
    def elapsed(self) -> float:
        return (self.stop - self.start).total_seconds()
