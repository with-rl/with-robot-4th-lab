"""Centralized logging helpers."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOGGERS = set()


def get_logger(
    name: Optional[str] = None,
    filename: Optional[str] = None,
    is_save: bool = False,
) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name (defaults to root logger)
        filename: Log file path (defaults to logs/<logger_name>.log)
        is_save: Whether to save logs to file

    Returns:
        Configured logger instance
    """
    global _CONFIGURED

    # Configure root logger once
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        _CONFIGURED = True

    logger_name = name or "robot_agent"
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers by checking if handlers already exist
    if logger.handlers:
        return logger

    _LOGGERS.add(logger_name)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    if is_save:
        # Determine log file path
        if filename:
            log_path = Path(filename).expanduser()
        else:
            safe_name = logger_name.replace(".", "_")
            log_path = Path("logs") / f"{safe_name}.log"

        # Create log directory if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add file handler
        handler = RotatingFileHandler(
            str(log_path),
            encoding="utf-8",
            maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=3,
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger
