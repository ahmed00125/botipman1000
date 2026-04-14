"""Structured logging via loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from quant.config import settings


def setup_logging(level: str | None = None) -> None:
    logger.remove()
    lvl = (level or settings.log_level).upper()
    logger.add(
        sys.stderr,
        level=lvl,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.log_dir / "quant_{time:YYYY-MM-DD}.log",
        level=lvl,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )
