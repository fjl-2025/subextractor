"""日志配置。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def _file_log_path() -> Path | None:
    """打包版（尤其无控制台的 GUI exe）把日志写到应用目录 logs/ 下。"""
    if not getattr(sys, "frozen", False):
        return None
    from ..models import registry

    log_dir = registry.app_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return log_dir / "subextractor.log"


def setup_logging(verbose: bool = False) -> None:
    """初始化根日志：默认 INFO，verbose 时 DEBUG。

    打包版无控制台（GUI exe）时自动追加滚动文件日志，便于排查问题；
    CLI exe 有控制台则同时输出到屏幕。
    """
    handlers: list[logging.Handler] = []
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler())
    log_path = _file_log_path()
    if log_path is not None:
        handlers.append(
            RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        )
    if not handlers:  # 极端情况下无 stdout 且文件不可写
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
