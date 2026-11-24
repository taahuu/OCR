"""Document orchestration: routes a validated upload through PDF/image handling,
native-text detection, preprocessing and OCR, and assembles the response.

All blocking work (PyMuPDF, OpenCV, Tesseract) runs in a worker thread so the
event loop stays free. A semaphore bounds how many documents are OCR'd at once.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    OcrEngineUnavailableError,
    OcrTimeoutError,
    ProcessingError,
    ServerBusyError,
)
from app.core.logging import get_logger
from app.schemas.ocr import ExtractionMetadata, ExtractionResponse, PageResult, WordBox
from app.services.image_service import ImageService
from app.services.docx_service import DocxService
from app.services.layout_service import LayoutService
from app.services.information_extraction_service import InformationExtractionService
from app.services.table_extraction_service import TableExtractionService
from app.services.ocr_service import OcrEngine, OcrResult
from app.services.pdf_service import PdfService
from app.services.preprocessing_service import PreprocessingService
from app.utils.file_validation import DocumentFormat, ValidatedUpload

logger = get_logger("document")

_GENERIC_PAGE_ERROR = "This page could not be processed."


@dataclass
class _Outcome:
    """Pages plus the first error seen (re-raised if every page failed)."""

    pages: list[PageResult]
    first_error: AppError | None = None
    render_dpi: int | None = None
    tables: list[dict] | None = None


class DocumentService:
    """Entry point of the processing pipeline used by the API layer."""

    def __init__(
        self,
        settings: Settings,
        ocr_engine: OcrEngine,
        image_service: ImageService | None = None,
        pdf_service: PdfService | None = None,
        preprocessing_service: PreprocessingService | None = None,
        docx_service: DocxService | None = None,
        information_service: InformationExtractionService | None = None,
    ) -> None:
        self._settings = settings
        self._ocr = ocr_engine
        self._images = image_service or ImageService(settings)
        self._pdfs = pdf_service or PdfService(settings)
        self._preprocessing = preprocessing_service or PreprocessingService(settings)
        self._docx = docx_service or DocxService()
        self._information = information_service or InformationExtractionService()
        self._table_extractor = TableExtractionService()
        self._slots = asyncio.Semaphore(settings.max_concurrent_ocr)

    # ------------------------------------------------------------------ async

    async def extract(self, data: bytes, upload: ValidatedUpload) -> ExtractionResponse:
        """Extract text from a validated upload.

        Waits for a free processing slot (up to ``QUEUE_TIMEOUT_SECONDS``),
        then runs the blocking pipeline in a worker thread. The interface is
        deliberately a single awaitable so it can later be moved behind a job
        queue without changing the API layer.
        """
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self._settings.queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ServerBusyError() from exc

        try:
            return await run_in_threadpool(self.extract_sync, data, upload)
        finally:
            self._slots.release()

    # ------------------------------------------------------------------- sync

    def extract_sync(self, data: bytes, upload: ValidatedUpload) -> ExtractionResponse:
        """Blocking implementation of the pipeline (also used directly by tests)."""
        started = time.monotonic()
        deadline = started + self._settings.document_timeout_seconds

        if upload.format is DocumentFormat.PDF:
            outcome = self._process_pdf(data, deadline)
        elif upload.format is DocumentFormat.DOCX:
            outcome = self._process_docx(data)
        else:
            outcome = self._process_image(data)

        pages = outcome.pages
        failed = [p for p in pages if p.status == "failed"]
        if len(failed) == len(pages):
            logger.warning("All pages failed: type=%s pages=%d", upload.format.document_type, len(pages))
            raise outcome.first_error or ProcessingError()

        status = "partial" if failed else "success"
        duration_ms = int((time.monotonic() - started) * 1000)
        ocr_pages = sum(1 for p in pages if p.method == "ocr" and p.status == "success")
        native_pages = sum(1 for p in pages if p.method == "native_text" and p.status == "success")

        logger.info(
            "Processed document: type=%s status=%s pages=%d native=%d ocr=%d failed=%d duration_ms=%d",
            upload.format.document_type, status, len(pages), native_pages, ocr_pages,
            len(failed), duration_ms,
        )

        return ExtractionResponse(
            status=status,
            filename=upload.filename,
            document_type=upload.format.document_type,
            detected_format=upload.format.value,
            total_pages=len(pages),
            pages=pages,
            full_text="\n\n".join(p.text for p in pages if p.status == "success" and p.text),
            tables=outcome.tables or [],
            structured_data=self._information.extract(
                "\n\n".join(p.text for p in pages if p.status == "success" and p.text),
                outcome.tables or [],
            ),
            metadata=ExtractionMetadata(
                processing_time_ms=duration_ms,
                ocr_language=self._settings.ocr_language,
                native_pages=native_pages,
                ocr_pages=ocr_pages,
                failed_pages=len(failed),
                render_dpi=outcome.render_dpi,
            ),
        )

    # ------------------------------------------------------------------ image

    def _process_image(self, data: bytes) -> _Outcome:
        image = self._images.load(data)
        gray = self._preprocessing.to_grayscale(image)
        result, steps = self._ocr_with_retry(gray)
        page = self._ocr_page(1, result, steps)
        tables = self._table_extractor.from_ocr_words(result.words, 1)
        return _Outcome(pages=[page], tables=tables)

    # ------------------------------------------------------------------- docx

    def _process_docx(self, data: bytes) -> _Outcome:
        text, tables = self._docx.extract(data)
        page = self._native_page(1, text)
        return _Outcome(pages=[page], tables=tables)

    # -------------------------------------------------------------------- pdf

    def _process_pdf(self, data: bytes, deadline: float) -> _Outcome:
        pages: list[PageResult] = []
        tables: list[dict] = []
        first_error: AppError | None = None
        render_dpi: int | None = None

        with self._pdfs.open_document(data) as document:
            for index in range(document.page_count):
                page_number = index + 1

                if time.monotonic() > deadline:
                    error = OcrTimeoutError("The document processing time limit was exceeded.")
                    first_error = first_error or error
                    pages.append(self._failed_page(page_number, "ocr", error.message))
                    continue

                try:
                    pdf_page = document.load_page(index)
                    tables.extend(self._pdfs.extract_tables(pdf_page))
                    page_result, dpi = self._process_pdf_page(page_number, pdf_page)
                    pages.append(page_result)
                    if page_result.method == "ocr":
                        tables.extend(self._table_extractor.from_ocr_words(page_result.words, page_number))
                    render_dpi = dpi or render_dpi
                except OcrEngineUnavailableError:
                    raise  # engine missing: fail the whole request, not per page
                except AppError as exc:
                    first_error = first_error or exc
                    pages.append(self._failed_page(page_number, "ocr", exc.message))
                except Exception as exc:  # noqa: BLE001 - isolate one bad page
                    logger.warning(
                        "Page %d failed unexpectedly: %s", page_number, type(exc).__name__,
                        exc_info=True,
                    )
                    first_error = first_error or ProcessingError()
                    pages.append(self._failed_page(page_number, "ocr", _GENERIC_PAGE_ERROR))

        return _Outcome(pages=pages, first_error=first_error, render_dpi=render_dpi, tables=tables)

    def _process_pdf_page(self, page_number: int, pdf_page) -> tuple[PageResult, int | None]:
        """Decide native vs OCR for one page and process it."""
        mode = self._settings.ocr_mode
        native_text = self._pdfs.extract_native_text(pdf_page)
        # A page counts as "native" when it has enough embedded text, or some
        # text and no images at all (e.g. a short typed page). A page with a
        # little text *and* images may be a scan with a small text overlay.
        has_enough_native = len(native_text) >= self._settings.native_text_min_chars or (
            bool(native_text) and not self._pdfs.has_images(pdf_page)
        )

        if mode == "never" or (mode == "auto" and has_enough_native):
            native_words = []
            try:
                for item in pdf_page.get_text("words", sort=True):
                    x0, y0, x1, y1, word, block, line, _word_no = item
                    native_words.append(WordBox(
                        text=str(word), confidence=None, left=int(x0), top=int(y0),
                        width=int(x1 - x0), height=int(y1 - y0),
                        block_number=int(block), paragraph_number=0, line_number=int(line),
                    ))
            except Exception:  # noqa: BLE001 - native text remains available without boxes
                native_words = []
            return self._native_page(page_number, native_text, native_words), None

        image, dpi = self._pdfs.render_page(pdf_page)
        result, steps = self._ocr_with_retry(image)

        if mode == "auto" and native_text and not result.has_text:
            # Short embedded text and OCR found nothing: trust the embedded text.
            return self._native_page(page_number, native_text), dpi
        return self._ocr_page(page_number, result, steps), dpi

    # -------------------------------------------------------------------- OCR

    def _ocr_with_retry(self, gray: Image.Image) -> tuple[OcrResult, list[str]]:
        """Plain OCR first; preprocess and retry only when confidence is low.

        Keeps the better of the two results, so preprocessing can never make
        the output worse than the baseline.
        """
        baseline = self._ocr.recognize(gray)

        if not self._settings.preprocessing_enabled or self._preprocessing.is_blank(gray):
            return baseline, []

        confident = (
            baseline.confidence is not None
            and baseline.confidence >= self._settings.low_confidence_threshold
        )
        if confident:
            return baseline, []

        rotation = self._ocr.detect_rotation(gray)
        enhanced = self._preprocessing.enhance(gray, rotation_degrees=rotation)
        if not enhanced.steps:
            return baseline, []

        retry = self._ocr.recognize(enhanced.image)
        if self._is_better(retry, baseline):
            return retry, enhanced.steps
        return baseline, []

    @staticmethod
    def _is_better(candidate: OcrResult, current: OcrResult) -> bool:
        """Prefer more reliably-read words; break ties on mean confidence."""
        if candidate.reliable_word_count != current.reliable_word_count:
            return candidate.reliable_word_count > current.reliable_word_count
        return (candidate.confidence or 0.0) > (current.confidence or 0.0)

    # ---------------------------------------------------------------- results

    @staticmethod
    def _native_page(page_number: int, text: str, words: list[WordBox] | None = None) -> PageResult:
        return PageResult(
            page_number=page_number, method="native_text", status="success",
            text=text, confidence=None, layout=LayoutService.from_text(text), words=words or [],
        )

    @staticmethod
    def _ocr_page(page_number: int, result: OcrResult, steps: list[str]) -> PageResult:
        return PageResult(
            page_number=page_number, method="ocr", status="success",
            text=result.text, confidence=result.confidence,
            preprocessing_applied=steps,
            layout=LayoutService.from_text(result.text),
            words=[WordBox(**word) for word in result.words],
        )

    @staticmethod
    def _failed_page(page_number: int, method: str, message: str) -> PageResult:
        return PageResult(
            page_number=page_number, method=method,  # type: ignore[arg-type]
            status="failed", text="", confidence=None, error=message,
        )
