"""Tesseract OCR via Pytesseract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import pytesseract
from PIL import Image
from pytesseract import Output

from app.core.config import Settings
from app.core.exceptions import (
    OcrEngineError,
    OcrEngineUnavailableError,
    OcrTimeoutError,
)
from app.core.logging import get_logger

logger = get_logger("ocr")

# Words at or above this Tesseract confidence count as "reliable" when two OCR
# attempts of the same page are compared.
RELIABLE_WORD_CONFIDENCE = 60.0
_WORD_LEVEL = 5  # Tesseract TSV level for individual words


@dataclass(frozen=True)
class OcrResult:
    """Outcome of one OCR pass over one image."""

    text: str
    confidence: float | None  # mean word confidence 0-100; None if no words found
    word_count: int
    reliable_word_count: int
    words: tuple[dict[str, int | float | str], ...] = ()

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True)
class EngineStatus:
    """Availability information reported by the health endpoint."""

    available: bool
    version: str | None
    language_available: bool


class OcrEngine(Protocol):
    """Interface the rest of the application depends on (easy to fake in tests)."""

    def recognize(self, image: Image.Image) -> OcrResult: ...

    def detect_rotation(self, image: Image.Image) -> int: ...

    def status(self) -> EngineStatus: ...


class TesseractOcrService:
    """Runs Tesseract through Pytesseract and normalises the output."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        # Tesseract is multi-threaded via OpenMP; with several concurrent jobs,
        # one thread per process is faster overall and avoids CPU contention.
        os.environ.setdefault("OMP_THREAD_LIMIT", str(settings.tesseract_omp_thread_limit))

    def recognize(self, image: Image.Image) -> OcrResult:
        """OCR one image and return text plus confidence.

        Raises:
            OcrTimeoutError: Tesseract exceeded ``OCR_TIMEOUT_SECONDS``.
            OcrEngineUnavailableError: the Tesseract binary was not found.
            OcrEngineError: any other Tesseract failure.
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._settings.ocr_language,
                config=self._settings.tesseract_config,
                output_type=Output.DICT,
                timeout=self._settings.ocr_timeout_seconds,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrEngineUnavailableError() from exc
        except RuntimeError as exc:
            # pytesseract signals a killed (timed-out) subprocess with RuntimeError.
            if "timeout" in str(exc).lower():
                raise OcrTimeoutError() from exc
            logger.warning("Tesseract runtime error: %s", type(exc).__name__)
            raise OcrEngineError() from exc
        except pytesseract.TesseractError as exc:
            logger.warning("Tesseract failed with status %s", getattr(exc, "status", "?"))
            raise OcrEngineError() from exc

        return self._parse_data(data)

    def detect_rotation(self, image: Image.Image) -> int:
        """Return the clockwise rotation (0/90/180/270) needed to make text upright.

        Returns 0 when orientation cannot be determined (little text, OSD data
        missing, low OSD confidence). Never raises for detection problems.
        """
        try:
            osd = pytesseract.image_to_osd(
                image,
                output_type=Output.DICT,
                timeout=self._settings.ocr_timeout_seconds,
            )
            rotate = int(osd.get("rotate", 0)) % 360
            confidence = float(osd.get("orientation_conf", 0))
        except (pytesseract.TesseractError, RuntimeError, ValueError, TypeError):
            return 0
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrEngineUnavailableError() from exc
        return rotate if rotate in (90, 180, 270) and confidence >= 2.0 else 0

    def status(self) -> EngineStatus:
        """Report whether Tesseract and the configured language(s) are usable."""
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception:  # noqa: BLE001 - any failure means "not available"
            return EngineStatus(available=False, version=None, language_available=False)
        try:
            installed = set(pytesseract.get_languages(config=""))
            wanted = set(self._settings.ocr_language.split("+"))
            language_ok = wanted <= installed
        except Exception:  # noqa: BLE001
            language_ok = False
        return EngineStatus(available=True, version=version, language_available=language_ok)

    @staticmethod
    def _parse_data(data: dict[str, list]) -> OcrResult:
        """Rebuild text (lines / paragraphs) and mean confidence from TSV data."""
        lines: dict[tuple[int, int, int, int], list[str]] = {}
        confidences: list[float] = []
        word_boxes: list[dict[str, int | float | str]] = []

        for i, raw_word in enumerate(data["text"]):
            word = str(raw_word).strip()
            if int(data["level"][i]) != _WORD_LEVEL or not word:
                continue
            key = (
                int(data["page_num"][i]),
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            lines.setdefault(key, []).append(word)
            confidence = float(data["conf"][i])
            if confidence >= 0:
                confidences.append(confidence)
                # Coordinates are relative to the source image in pixels.
                word_boxes.append({
                    "text": word,
                    "confidence": round(confidence, 2),
                    "left": int(data.get("left", [0] * len(data["text"]))[i]),
                    "top": int(data.get("top", [0] * len(data["text"]))[i]),
                    "width": int(data.get("width", [0] * len(data["text"]))[i]),
                    "height": int(data.get("height", [0] * len(data["text"]))[i]),
                    "block_number": int(data["block_num"][i]),
                    "paragraph_number": int(data["par_num"][i]),
                    "line_number": int(data["line_num"][i]),
                })

        output: list[str] = []
        previous_paragraph: tuple[int, int, int] | None = None
        for key, words in lines.items():
            paragraph = key[:3]
            if previous_paragraph is not None and paragraph != previous_paragraph:
                output.append("")  # blank line between paragraphs / blocks
            output.append(" ".join(words))
            previous_paragraph = paragraph

        mean_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
        return OcrResult(
            text="\n".join(output).strip(),
            confidence=mean_confidence,
            word_count=len(confidences),
            reliable_word_count=sum(1 for c in confidences if c >= RELIABLE_WORD_CONFIDENCE),
            words=tuple(word_boxes),
        )
