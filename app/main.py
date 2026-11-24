"""Application factory and ASGI entry point (``uvicorn app.main:app``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import MULTIPART_OVERHEAD_BYTES, RequestSizeLimitMiddleware
from app.services.document_service import DocumentService
from app.services.ocr_service import OcrEngine, TesseractOcrService

logger = get_logger("main")

_DESCRIPTION = """
Upload a **PDF, image (PNG/JPEG/TIFF/BMP/WEBP), or DOCX** and receive extracted text, layout, tables (where detectable), and heuristic structured fields.

* PDF pages with selectable text are read directly (no OCR needed).
* Scanned PDF pages and images are OCR'd locally with Tesseract.
* DOCX paragraphs and tables are extracted locally without OCR.
* Document type and common fields are classified/extracted with local rules; no external AI API is used.
* Mixed PDFs are handled per page; page order is preserved.

No external API or key is required.
"""


def create_app(
    settings: Settings | None = None, ocr_engine: OcrEngine | None = None
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Override settings (used by tests); defaults to the environment.
        ocr_engine: Override the OCR engine (used by tests to inject a fake).
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = ocr_engine or TesseractOcrService(settings)
        app.state.settings = settings
        app.state.ocr_engine = engine
        app.state.document_service = DocumentService(settings, engine)

        status = engine.status()
        if not status.available:
            logger.error("OCR engine NOT available - extraction requests will fail")
        elif not status.language_available:
            logger.error("OCR language %r is not installed", settings.ocr_language)
        logger.info(
            "Starting %s v%s env=%s ocr_engine=%s language=%s",
            settings.app_name, __version__, settings.environment,
            status.version or "unavailable", settings.ocr_language,
        )
        yield
        logger.info("Shutting down %s", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
