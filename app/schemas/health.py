"""Health-check schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    environment: str
    ocr_engine_available: bool
    ocr_engine_version: str | None = None
    ocr_language: str
    ocr_language_available: bool
