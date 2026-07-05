"""
Logging Module
==============

Configures structured Python logging for the backend.

Call :func:`setup_logging` once during application startup
(handled automatically by the lifespan handler in ``main.py``).
Use :func:`get_logger` to obtain module-scoped loggers.

Usage::

    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started for camera %d", camera_id)
"""

import logging
import sys
from typing import Optional

from app.core.config import settings

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_is_configured = False


def setup_logging() -> None:
    """Configure the root logger.

    Sets log level from :attr:`Settings.LOG_LEVEL`, applies a
    uniform formatter, and directs output to ``stdout``.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _is_configured
    if _is_configured:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    _is_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger scoped to *name*.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name or __name__)
