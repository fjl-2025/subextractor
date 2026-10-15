"""日志配置。"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """初始化根日志：默认 INFO，verbose 时 DEBUG。"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
