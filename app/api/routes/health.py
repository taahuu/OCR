"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app import __version__
from app.core.config import Settings
from app.dependencies.providers import get_app_settings, get_ocr_engine
from app.schemas.health import HealthResponse
from app.services.ocr_service import OcrEngine

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and OCR engine availability",
    responses={503: {"model": HealthResponse, "description": "OCR engine unavailable"}},
)
async def health(
    settings: Settings = Depends(get_app_settings),
    engine: OcrEngine = Depends(get_ocr_engine),
) -> JSONResponse:
    """Return ``200`` when Tesseract and the configured language are usable,
    otherwise ``503`` so load balancers and orchestrators can react."""
    engine_status = await run_in_threadpool(engine.status)
    healthy = engine_status.available and engine_status.language_available
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.environment,
        ocr_engine_available=engine_status.available,
        ocr_engine_version=engine_status.version,
        ocr_language=settings.ocr_language,
        ocr_language_available=engine_status.language_available,
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())
