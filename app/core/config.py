"""Centralised, environment-driven application settings.

Every tunable limit and OCR option lives here. Values are read from
environment variables (or a local ``.env`` file) so nothing is hardcoded in
the service layer.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Tesseract language codes such as "eng" or "eng+deu". Validating the format
# prevents option injection into the Tesseract command line.
_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")


class Settings(BaseSettings):
    """Application configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -----------------------------------------------------
    app_name: str = "OCR Document Extraction API"
    environment: Literal["development", "testing", "production"] = "development"
    log_level: str = "INFO"

    # --- Upload and document limits -------------------------------------
    max_upload_size_mb: int = Field(default=20, gt=0, le=500)
    max_pdf_pages: int = Field(default=50, gt=0, le=2000)
    max_image_pixels: int = Field(default=50_000_000, gt=0)
    max_image_dimension: int = Field(default=12_000, gt=0)
    max_render_pixels: int = Field(default=50_000_000, gt=0)

    # --- OCR -------------------------------------------------------------
    ocr_language: str = "eng"
    tesseract_cmd: str | None = None
    tesseract_config: str = "--psm 3"
    ocr_timeout_seconds: int = Field(default=60, gt=0)
    document_timeout_seconds: int = Field(default=300, gt=0)
    render_dpi: int = Field(default=300, ge=72, le=600)
    ocr_mode: Literal["auto", "force", "never"] = "auto"
    native_text_min_chars: int = Field(default=25, ge=0)

    # --- Concurrency -----------------------------------------------------
    max_concurrent_ocr: int = Field(default=2, ge=1, le=64)
    queue_timeout_seconds: int = Field(default=30, gt=0)
    tesseract_omp_thread_limit: int = Field(default=1, ge=1)

    # --- Preprocessing ---------------------------------------------------
    preprocessing_enabled: bool = True
    low_confidence_threshold: float = Field(default=70.0, ge=0, le=100)
    preprocess_min_long_side: int = Field(default=1500, gt=0)

    @field_validator("ocr_language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        if not _LANGUAGE_PATTERN.match(value):
            raise ValueError(
                "OCR_LANGUAGE must look like 'eng' or 'eng+deu' "
                "(letters, digits, underscore, '+')."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown LOG_LEVEL: {value!r}")
        return level

    @field_validator("tesseract_cmd", mode="before")
    @classmethod
    def _empty_cmd_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()
