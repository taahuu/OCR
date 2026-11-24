"""FastAPI dependency providers.

Services are created once at startup (see ``app.main``) and stored on
``app.state``; routes receive them through these providers, which also makes
them trivial to override in tests.
"""

from __future__ import annotations

from fastapi import Request

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.ocr_service import OcrEngine


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_ocr_engine(request: Request) -> OcrEngine:
    return request.app.state.ocr_engine
