import pymupdf
import pytest

from app.core.exceptions import (
    CorruptDocumentError,
    EncryptedPdfError,
    PageLimitExceededError,
)
from app.services.pdf_service import PdfService
from tests import document_factory as docs
from tests.conftest import make_settings


@pytest.fixture
def service() -> PdfService:
    return PdfService(make_settings())


class TestOpenDocument:
    def test_opens_valid_pdf(self, service):
        with service.open_document(docs.make_native_pdf([["a"], ["b"], ["c"]])) as document:
            assert document.page_count == 3

    def test_corrupted_pdf(self, service):
        with pytest.raises(CorruptDocumentError):
            service.open_document(docs.CORRUPT_PDF)

    def test_truncated_pdf(self, service):
        data = docs.make_native_pdf([["hello world " * 20]])
        with pytest.raises(CorruptDocumentError):
            service.open_document(data[:60])

    def test_pdf_without_pages(self, service):
        with pytest.raises(CorruptDocumentError):
            service.open_document(docs.ZERO_PAGE_PDF)

    def test_page_limit_enforced(self):
        service = PdfService(make_settings(max_pdf_pages=2))
        with pytest.raises(PageLimitExceededError, match="3 pages"):
            service.open_document(docs.make_blank_pdf(3))

    def test_exactly_at_page_limit_is_allowed(self):
        service = PdfService(make_settings(max_pdf_pages=3))
        with service.open_document(docs.make_blank_pdf(3)) as document:
            assert document.page_count == 3

    def test_password_protected_pdf_rejected(self, service):
        source = pymupdf.open()
        source.new_page()
        encrypted = source.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner"
        )
        with pytest.raises(EncryptedPdfError):
            service.open_document(encrypted)


class TestNativeText:
    def test_extracts_embedded_text_in_order(self, service):
        data = docs.make_native_pdf([["First line", "Second line"], ["Page two"]])
        with service.open_document(data) as document:
            assert service.extract_native_text(document[0]) == "First line\nSecond line"
            assert service.extract_native_text(document[1]) == "Page two"

    def test_scanned_page_has_no_native_text_but_has_images(self, service):
        with service.open_document(docs.make_scanned_pdf([["Scanned"]])) as document:
            assert service.extract_native_text(document[0]) == ""
            assert service.has_images(document[0]) is True

    def test_native_page_has_no_images(self, service):
        with service.open_document(docs.make_native_pdf([["text"]])) as document:
            assert service.has_images(document[0]) is False

    def test_blank_page_returns_empty_string(self, service):
        with service.open_document(docs.make_blank_pdf()) as document:
            assert service.extract_native_text(document[0]) == ""


class TestRenderPage:
    def test_renders_grayscale_at_configured_dpi(self):
        service = PdfService(make_settings(render_dpi=150))
        with service.open_document(docs.make_blank_pdf()) as document:
            image, dpi = service.render_page(document[0])
        assert image.mode == "L"
        assert dpi == 150
        # US Letter 612x792 pt -> 8.5x11 in at 150 DPI
        assert image.size == (1275, 1650)

    def test_dpi_is_lowered_when_page_would_be_too_large(self):
        service = PdfService(make_settings(render_dpi=300, max_render_pixels=1_000_000))
        with service.open_document(docs.make_blank_pdf()) as document:
            image, dpi = service.render_page(document[0])
        assert dpi < 300
        assert image.width * image.height <= 1_000_000 * 1.02

    def test_rendered_scan_contains_dark_pixels(self, service):
        with service.open_document(docs.make_scanned_pdf([["Some text"]])) as document:
            image, _ = service.render_page(document[0])
        assert image.getextrema()[0] < 100  # black text present
