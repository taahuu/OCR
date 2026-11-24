"""HTTP-level tests. Fast tests use a fake OCR engine; integration tests use real Tesseract."""

import pymupdf
import pytest

from app.core.exceptions import (
    OcrEngineError,
    OcrEngineUnavailableError,
    OcrTimeoutError,
)
from app.services.ocr_service import EngineStatus
from tests import document_factory as docs
from tests.conftest import FakeOcrEngine, make_settings, requires_tesseract

EXTRACT = "/api/v1/ocr/extract"
LONG_TEXT = "This line contains clearly more than twenty five characters."

PNG = docs.image_bytes(docs.render_text_image(["hello"], size=(400, 300)))
JPG = docs.image_bytes(docs.render_text_image(["hello"], size=(400, 300)), "JPEG")


def upload(client, data: bytes, name: str = "doc.png", content_type: str | None = "image/png"):
    files = {"file": (name, data, content_type)} if content_type else {"file": (name, data)}
    return client.post(EXTRACT, files=files)


def assert_error(response, status: int, code: str):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    text = response.text
    for leak in ("Traceback", "/home/", "site-packages", ".py"):
        assert leak not in text
    return body["error"]


class TestHealthAndInfo:
    def test_health_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ocr_engine_available"] is True
        assert body["ocr_language"] == "eng"
        assert body["version"]

    def test_health_returns_503_when_engine_missing(self, client_factory):
        class Missing(FakeOcrEngine):
            def status(self):
                return EngineStatus(available=False, version=None, language_available=False)

        response = client_factory(ocr_engine=Missing()).get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_supported_formats(self, client):
        body = client.get("/api/v1/ocr/supported-formats").json()
        extensions = " ".join(f["extension"] for f in body["formats"])
        assert ".pdf" in extensions and ".png" in extensions and ".jpg" in extensions
        assert body["max_upload_size_mb"] == 20
        assert body["max_pdf_pages"] == 50

    def test_openapi_documents_all_endpoints(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/health" in paths
        assert "/api/v1/ocr/supported-formats" in paths
        assert "post" in paths["/api/v1/ocr/extract"]
        assert client.get("/docs").status_code == 200


class TestSuccessfulExtraction:
    def test_png(self, client):
        response = upload(client, PNG)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["filename"] == "doc.png"
        assert body["document_type"] == "image"
        assert body["detected_format"] == "png"
        assert body["total_pages"] == 1
        assert body["full_text"] == "fake text"
        page = body["pages"][0]
        assert page == {
            "page_number": 1, "method": "ocr", "status": "success", "text": "fake text",
            "confidence": 90.0, "preprocessing_applied": [], "error": None,
            "layout": {"paragraphs": ["fake text"], "headings": [], "line_count": 1},
            "words": [],
        }
        assert set(body["metadata"]) == {
            "processing_time_ms", "ocr_language", "native_pages", "ocr_pages",
            "failed_pages", "render_dpi",
        }

    def test_jpeg(self, client):
        response = upload(client, JPG, "photo.JPG", "image/jpeg")
        assert response.status_code == 200
        assert response.json()["detected_format"] == "jpeg"

    def test_native_pdf_does_not_use_ocr(self, client, fake_ocr):
        response = upload(client, docs.make_native_pdf([[LONG_TEXT]]), "n.pdf", "application/pdf")
        body = response.json()
        assert response.status_code == 200
        assert body["document_type"] == "pdf"
        assert body["pages"][0]["method"] == "native_text"
        assert body["full_text"] == LONG_TEXT
        assert fake_ocr.calls == []

    def test_multi_page_mixed_pdf(self, client):
        data = docs.make_mixed_pdf([("native", [LONG_TEXT]), ("scanned", ["x"]), ("scanned", ["y"])])
        body = upload(client, data, "m.pdf", "application/pdf").json()
        assert body["total_pages"] == 3
        assert [p["page_number"] for p in body["pages"]] == [1, 2, 3]
        assert [p["method"] for p in body["pages"]] == ["native_text", "ocr", "ocr"]
        assert body["metadata"]["native_pages"] == 1 and body["metadata"]["ocr_pages"] == 2

    def test_schema_is_identical_for_image_and_pdf(self, client):
        image_body = upload(client, PNG).json()
        pdf_body = upload(client, docs.make_scanned_pdf([["a"]]), "a.pdf", "application/pdf").json()
        assert image_body.keys() == pdf_body.keys()
        assert image_body["pages"][0].keys() == pdf_body["pages"][0].keys()

    def test_hostile_filename_is_sanitised_in_response(self, client):
        body = upload(client, PNG, "../../etc/passwd.png").json()
        assert body["filename"] == "passwd.png"


class TestRejectedUploads:
    def test_empty_file(self, client):
        assert_error(upload(client, b""), 400, "empty_file")

    def test_unsupported_extension(self, client):
        assert_error(upload(client, b"GIF89a....", "a.gif", "image/gif"), 415, "unsupported_format")

    def test_unsupported_content_with_allowed_extension(self, client):
        assert_error(upload(client, b"MZ\x90\x00" + b"\0" * 64, "evil.png"), 415, "unsupported_format")

    def test_extension_and_content_mismatch(self, client):
        error = assert_error(upload(client, PNG, "x.pdf", "application/pdf"), 415, "unsupported_format")
        assert "extension" in error["message"]

    def test_mime_and_content_mismatch(self, client):
        assert_error(upload(client, PNG, "x.png", "application/pdf"), 415, "unsupported_format")

    def test_corrupted_png(self, client):
        assert_error(upload(client, PNG[: len(PNG) // 2]), 400, "corrupt_document")

    def test_corrupted_pdf(self, client):
        assert_error(upload(client, docs.CORRUPT_PDF, "c.pdf", "application/pdf"), 400, "corrupt_document")

    def test_pdf_without_pages(self, client):
        assert_error(upload(client, docs.ZERO_PAGE_PDF, "z.pdf", "application/pdf"), 400, "corrupt_document")

    def test_pdf_over_page_limit(self, client_factory, fake_ocr):
        client = client_factory(make_settings(max_pdf_pages=2), fake_ocr)
        error = assert_error(upload(client, docs.make_blank_pdf(3), "p.pdf", "application/pdf"), 422, "page_limit_exceeded")
        assert "2" in error["message"]
        assert fake_ocr.calls == []

    def test_password_protected_pdf(self, client):
        source = pymupdf.open()
        source.new_page()
        data = source.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="a", owner_pw="b")
        assert_error(upload(client, data, "e.pdf", "application/pdf"), 422, "encrypted_pdf")

    def test_image_over_pixel_limit(self, client_factory, fake_ocr):
        client = client_factory(make_settings(max_image_pixels=10_000), fake_ocr)
        assert_error(upload(client, PNG), 422, "image_too_large")

    def test_missing_file_field(self, client):
        error = assert_error(client.post(EXTRACT, data={"other": "x"}), 422, "validation_error")
        assert "file" in error["message"]

    def test_wrong_method_and_unknown_route_use_error_envelope(self, client):
        assert_error(client.get(EXTRACT), 405, "method_not_allowed")
        assert_error(client.get("/api/v1/nope"), 404, "not_found")


class TestSizeLimit:
    @pytest.fixture
    def small_limit_client(self, client_factory, fake_ocr):
        return client_factory(make_settings(max_upload_size_mb=1), fake_ocr)

    def test_file_far_over_limit_is_rejected_from_content_length(self, small_limit_client, fake_ocr):
        data = PNG + b"\0" * (3 * 1024 * 1024)
        assert_error(upload(small_limit_client, data), 413, "file_too_large")
        assert fake_ocr.calls == []

    def test_file_slightly_over_limit_is_rejected_by_the_reader(self, small_limit_client):
        data = PNG + b"\0" * (1024 * 1024 + 10)  # passes the body-size allowance, exceeds the file limit
        assert_error(upload(small_limit_client, data), 413, "file_too_large")

    def test_streamed_body_without_content_length_is_aborted(self, small_limit_client):
        boundary = "xBOUNDARYx"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        tail = f"\r\n--{boundary}--\r\n".encode()

        def body():
            yield head
            for _ in range(6):
                yield PNG[:8] + b"\0" * (512 * 1024)
            yield tail

        response = small_limit_client.post(
            EXTRACT, content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        assert_error(response, 413, "file_too_large")

    def test_file_at_the_limit_is_accepted(self, small_limit_client):
        # Valid PNG padded with trailing bytes to exactly 1 MB (PNG readers ignore trailing data).
        data = PNG + b"\0" * (1024 * 1024 - len(PNG))
        assert len(data) == 1024 * 1024
        assert upload(small_limit_client, data).status_code == 200


class TestInternalFailures:
    def test_unexpected_exception_returns_generic_500(self, client_factory):
        client = client_factory(ocr_engine=FakeOcrEngine(error=RuntimeError("boom secret /srv/app")))
        # A page failure on an image propagates as an unexpected error.
        response = upload(client, PNG)
        error = assert_error(response, 500, "internal_error")
        assert "boom" not in error["message"] and "secret" not in error["message"]

    @pytest.mark.parametrize(
        ("exc", "status", "code"),
        [
            (OcrTimeoutError(), 504, "ocr_timeout"),
            (OcrEngineError(), 500, "ocr_failed"),
            (OcrEngineUnavailableError(), 503, "ocr_engine_unavailable"),
        ],
    )
    def test_engine_errors_map_to_status_codes(self, client_factory, exc, status, code):
        client = client_factory(ocr_engine=FakeOcrEngine(error=exc))
        assert_error(upload(client, PNG), status, code)

    def test_partial_failure_returns_200_with_partial_status(self, client_factory):
        class Flaky(FakeOcrEngine):
            attempts = 0

            def recognize(self, image):
                self.attempts += 1
                if self.attempts == 2:
                    raise OcrTimeoutError()
                return super().recognize(image)

        client = client_factory(ocr_engine=Flaky())
        body = upload(client, docs.make_scanned_pdf([["a"], ["b"]]), "s.pdf", "application/pdf").json()
        assert body["status"] == "partial"
        assert body["pages"][1]["status"] == "failed"
        assert body["pages"][1]["error"]


# ---------------------------------------------------------------- integration

@pytest.mark.integration
@requires_tesseract
class TestEndToEndWithRealTesseract:
    @pytest.fixture(scope="class")
    def real_client(self):
        from fastapi.testclient import TestClient

        from app.main import create_app

        with TestClient(create_app(make_settings())) as client:
            yield client

    @staticmethod
    def assert_contains(text: str, *words: str):
        for word in words:
            assert word in text, f"{word!r} missing from OCR output: {text!r}"

    def test_health_reports_real_engine(self, real_client):
        body = real_client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["ocr_engine_version"].startswith("5.")

    def test_png(self, real_client):
        data = docs.image_bytes(docs.render_text_image(docs.SAMPLE_LINES))
        body = upload(real_client, data).json()
        self.assert_contains(body["full_text"], "INVOICE", "Acme Corporation", "1250", "business")
        assert body["pages"][0]["confidence"] > 85

    def test_jpeg(self, real_client):
        data = docs.image_bytes(docs.render_text_image(docs.SAMPLE_LINES), "JPEG")
        body = upload(real_client, data, "scan.jpg", "image/jpeg").json()
        self.assert_contains(body["full_text"], "INVOICE", "Acme", "1250")

    def test_normal_pdf_text_is_exact_and_needs_no_ocr(self, real_client):
        data = docs.make_native_pdf([docs.SAMPLE_LINES])
        body = upload(real_client, data, "n.pdf", "application/pdf").json()
        assert body["pages"][0]["method"] == "native_text"
        assert body["full_text"] == "\n".join(docs.SAMPLE_LINES)

    def test_scanned_multi_page_pdf(self, real_client):
        data = docs.make_scanned_pdf([docs.SAMPLE_LINES, ["Second page about Tesseract engines", "Reference code ZX-9981"]])
        body = upload(real_client, data, "s.pdf", "application/pdf").json()
        assert body["total_pages"] == 2
        assert [p["method"] for p in body["pages"]] == ["ocr", "ocr"]
        self.assert_contains(body["pages"][0]["text"], "INVOICE", "Acme")
        self.assert_contains(body["pages"][1]["text"], "Second page", "Tesseract", "ZX-9981")
        assert body["metadata"]["render_dpi"] == 300

    def test_mixed_pdf(self, real_client):
        data = docs.make_mixed_pdf([("native", [LONG_TEXT]), ("scanned", docs.SAMPLE_LINES)])
        body = upload(real_client, data, "m.pdf", "application/pdf").json()
        assert [p["method"] for p in body["pages"]] == ["native_text", "ocr"]
        assert body["pages"][0]["text"] == LONG_TEXT
        self.assert_contains(body["pages"][1]["text"], "INVOICE", "Acme")
        assert body["full_text"].index("clearly") < body["full_text"].index("INVOICE")

    def test_upside_down_scan_is_corrected_by_preprocessing(self, real_client):
        page = docs.render_text_image(docs.SAMPLE_LINES).rotate(180)
        body = upload(real_client, docs.image_bytes(page)).json()
        self.assert_contains(body["full_text"], "INVOICE", "Acme", "1250")
        assert "rotate_180" in body["pages"][0]["preprocessing_applied"]

    def test_uneven_lighting_is_recovered_by_preprocessing(self, real_client):
        import numpy as np
        from PIL import Image

        gray = np.asarray(docs.render_text_image(docs.SAMPLE_LINES).convert("L")).astype(float)
        shaded = Image.fromarray((gray * np.linspace(0.2, 1, gray.shape[1])[None, :]).astype("uint8"))
        body = upload(real_client, docs.image_bytes(shaded)).json()
        self.assert_contains(body["full_text"], "INVOICE", "Acme", "1250")
        assert "binarize" in body["pages"][0]["preprocessing_applied"]


def test_docx_upload_extracts_text_and_tables(client):
    from io import BytesIO
    from docx import Document
    document = Document()
    document.add_paragraph("INVOICE")
    document.add_paragraph("Invoice No: INV-100")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "Book"
    table.cell(1, 1).text = "500"
    stream = BytesIO()
    document.save(stream)
    response = client.post(EXTRACT, files={"file": ("invoice.docx", stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_type"] == "word"
    assert body["detected_format"] == "docx"
    assert "Invoice No: INV-100" in body["full_text"]
    assert body["tables"][0]["headers"] == ["Item", "Amount"]
    assert body["structured_data"]["document_type"] == "invoice"
