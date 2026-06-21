"""
Sterling Logger
Sets up consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Create and configure a logger for a Sterling module.

    Args:
        name:     Logger name (module name, e.g. "sterling.stt")
        level:    Log level string: DEBUG | INFO | WARNING | ERROR
        log_file: Optional path to a log file. Console output always enabled.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # If already configured (e.g. the bootstrap logger created before config was
    # loaded), don't add duplicate console handlers — but DO honour a newly
    # requested level and file handler. Without this, logging.level and
    # logging.file from config.yaml were silently ignored on the second call.
    if logger.handlers:
        logger.setLevel(numeric_level)
        for h in logger.handlers:
            h.setLevel(numeric_level)
        if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            try:
                fmt = logging.Formatter(
                    fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
                    datefmt="%H:%M:%S",
                )
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)
            except OSError as e:
                logger.warning(f"Could not open log file '{log_file}': {e}")
        return logger

    logger.setLevel(numeric_level)

    # Stop messages propagating to the root (or parent) logger.
    # Without this, 'sterling.stt' propagates up to 'sterling', and if both
    # have StreamHandlers the same line prints twice.
    logger.propagate = False

    # Formatter — clean, readable
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler (always on)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning(f"Could not open log file '{log_file}': {e}")

    return logger
