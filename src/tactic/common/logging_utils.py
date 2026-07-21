"""File-based logger setup shared by all pipeline stages."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str,
    file_path: Path,
    level: int = logging.WARNING,
    mode: str = "w",
    propagate: bool = False,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Create (or reconfigure) a named logger that writes to a file.

    Ensures the parent directory exists, (re)places any existing handlers on the
    logger, and attaches a single `FileHandler` with the given format.

    Args:
        name: Logger name (namespace). Re-using the same name reconfigures that logger.
        file_path: Path to the log file. Parent directories are created if needed.
        level: Logging level (e.g., logging.INFO, logging.WARNING). Defaults to WARNING.
        mode: File open mode for the handler: "w" to overwrite (default) or "a" to append.
        propagate: Whether log records should propagate to ancestor loggers. Defaults to False
            to avoid duplicate lines if a root handler exists.
        fmt: Log record format string.

    Returns:
        The configured `logging.Logger` instance.

    Notes:
        - This function removes any existing handlers on the named logger to prevent
          duplicate outputs when called multiple times in the same process.
        - The handler uses UTF-8 encoding.
    """
    # Ensure destination folder exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove any existing handlers to avoid duplicates
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.FileHandler(file_path, mode=mode, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    return logger
