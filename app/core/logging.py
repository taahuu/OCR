"""Logging configuration.

The application never logs document contents, extracted text or filenames
(filenames can contain personal data). Only metadata such as document type,
page counts, durations and error codes is logged.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "ocr_api"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the application."""
    logging.basicConfig(level=level, format=_LOG_FORMAT, force=True)
    logging.getLogger(LOGGER_NAME).setLevel(level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under the application logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
