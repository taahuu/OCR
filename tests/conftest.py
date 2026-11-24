"""Shared fixtures: settings, fake OCR engine, app/client factories."""

from __future__ import annotations

import shutil
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.services.ocr_service import EngineStatus, OcrResult

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

requires_tesseract = pytest.mark.skipif(
    not TESSERACT_AVAILABLE, reason="tesseract binary not installed"
)


def make_settings(**overrides) -> Settings:
    """Settings that ignore any local .env file."""
    values = {"environment": "testing", "log_level": "WARNING"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeOcrEngine:
    """Deterministic stand-in for Tesseract that records how it was used."""

    def __init__(
        self,
        text: str = "fake text",
        confidence: float | None = 90.0,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.error = error
        self.calls: list[tuple[int, int]] = []  # image sizes passed to recognize()
        self.rotation = 0
        self.responses: list[OcrResult] = []  # consumed first, one per call

    def recognize(self, image: Image.Image) -> OcrResult:
        self.calls.append(image.size)
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        words = len(self.text.split())
        return OcrResult(
            text=self.text,
            confidence=self.confidence,
            word_count=words,
            reliable_word_count=words if (self.confidence or 0) >= 60 else 0,
        )

    def detect_rotation(self, image: Image.Image) -> int:
        return self.rotation

    def status(self) -> EngineStatus:
        return EngineStatus(available=True, version="fake-1.0", language_available=True)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def fake_ocr() -> FakeOcrEngine:
    return FakeOcrEngine()


@pytest.fixture
def client_factory() -> Callable[..., TestClient]:
    """Build a TestClient with custom settings / OCR engine.

    ``raise_server_exceptions=False`` lets tests observe the real 500 response
    produced by the catch-all handler.
    """
    created: list[TestClient] = []

    def factory(settings: Settings | None = None, ocr_engine=None) -> TestClient:
        app: FastAPI = create_app(settings or make_settings(), ocr_engine=ocr_engine)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()  # run lifespan so app.state is populated
        created.append(client)
        return client

    yield factory
    for client in created:
        client.__exit__(None, None, None)


@pytest.fixture
def client(client_factory, fake_ocr) -> TestClient:
    """Default client wired to the fake OCR engine."""
    return client_factory(ocr_engine=fake_ocr)
