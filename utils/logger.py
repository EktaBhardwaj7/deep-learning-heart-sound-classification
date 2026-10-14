"""Logging utilities: file + console + optional TensorBoard."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "heart_sound",
               log_file: str | Path | None = None,
               level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class TensorBoardWriter:
    """Silent wrapper - falls back to no-op if tensorboard is unavailable."""
    def __init__(self, log_dir: str | Path):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
        except Exception:
            self.writer = None

    def add_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()