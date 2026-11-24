"""PDF handling with PyMuPDF: safe open, native text extraction, page rendering."""

from __future__ import annotations

import math

import pymupdf
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import (
    CorruptDocumentError,
    EncryptedPdfError,
    PageLimitExceededError,
)
from app.core.logging import get_logger

logger = get_logger("pdf")

# MuPDF prints repairable-error chatter to stderr; we surface failures through
# exceptions instead.
pymupdf.TOOLS.mupdf_display_errors(False)

_POINTS_PER_INCH = 72.0


class PdfService:
    """Opens untrusted PDFs and provides per-page text and rendering."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def open_document(self, data: bytes) -> pymupdf.Document:
        """Open a PDF from memory and enforce encryption and page limits.

        The caller must close the returned document (``with`` is supported).

        Raises:
            CorruptDocumentError: not a parseable PDF, or it has no pages.
            EncryptedPdfError: the PDF requires a password.
            PageLimitExceededError: more pages than ``MAX_PDF_PAGES``.
        """
        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - MuPDF raises several types
            logger.info("PDF open failed: %s", type(exc).__name__)
            raise CorruptDocumentError("The PDF file is corrupted or unreadable.") from exc

        try:
            if document.needs_pass or document.is_encrypted:
                raise EncryptedPdfError()
            page_count = document.page_count
            if page_count < 1:
                raise CorruptDocumentError("The PDF contains no pages.")
            if page_count > self._settings.max_pdf_pages:
                raise PageLimitExceededError(
                    f"The PDF has {page_count} pages; the maximum allowed is "
                    f"{self._settings.max_pdf_pages}."
                )
        except Exception:
            document.close()
            raise
        return document

    @staticmethod
    def extract_native_text(page: pymupdf.Page) -> str:
        """Return the embedded (selectable) text of a page, reading-order sorted."""
        return page.get_text("text", sort=True).strip()

    @staticmethod
    def has_images(page: pymupdf.Page) -> bool:
        """True if the page embeds at least one raster image."""
        return bool(page.get_images(full=False))


    @staticmethod
    def extract_tables(page: pymupdf.Page) -> list[dict]:
        """Extract tables when MuPDF's table detector can identify a grid."""
        try:
            finder = page.find_tables()
            extracted = []
            for index, table in enumerate(finder.tables, start=1):
                rows = table.extract()
                if rows:
                    extracted.append({
                        "table_number": index,
                        "headers": [str(x or "").strip() for x in rows[0]],
                        "rows": [[str(x or "").strip() for x in row] for row in rows[1:]],
                    })
            return extracted
        except Exception:  # noqa: BLE001 - table detection must never block text extraction
            # Older PyMuPDF builds or unsupported page layouts: preserve text output.
            return []

    def render_page(self, page: pymupdf.Page) -> tuple[Image.Image, int]:
        """Render a page to an 8-bit grayscale image.

        The requested DPI is lowered automatically when the page would exceed
        ``MAX_RENDER_PIXELS`` (guards against huge page sizes).

        Returns:
            ``(image, effective_dpi)``
        """
        rect = page.rect
        dpi = float(self._settings.render_dpi)
        width_px = rect.width * dpi / _POINTS_PER_INCH
        height_px = rect.height * dpi / _POINTS_PER_INCH

        if width_px < 1 or height_px < 1:
            raise CorruptDocumentError("The PDF page has invalid dimensions.")

        pixels = width_px * height_px
        if pixels > self._settings.max_render_pixels:
            dpi *= math.sqrt(self._settings.max_render_pixels / pixels)
            logger.info("Page too large; render DPI reduced to %.0f", dpi)

        scale = dpi / _POINTS_PER_INCH
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )
        try:
            image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            pixmap = None  # release MuPDF memory promptly
        return image, int(round(dpi))
