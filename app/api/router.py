"""Aggregates all v1 routes under ``/api/v1``."""

from fastapi import APIRouter

from app.api.routes import health, ocr

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(ocr.router)
