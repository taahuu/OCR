"""Orchestration tests: native-vs-OCR decisions, retry logic, failures, concurrency."""

import asyncio
import threading
import time

import pytest

from app.core.exceptions import (
    OcrEngineUnavailableError,
    OcrTimeoutError,
    ServerBusyError,
)
from app.services.document_service import DocumentService
from app.services.ocr_service import OcrResult
from app.utils.file_validation import DocumentFormat, ValidatedUpload
from tests import document_factory as docs
from tests.conftest import FakeOcrEngine, make_settings

PDF = ValidatedUpload("doc.pdf", DocumentFormat.PDF, 1)
PNG = ValidatedUpload("pic.png", DocumentFormat.PNG, 1)
LONG_TEXT = "This line contains clearly more than twenty five characters."


def run(service: DocumentService, data: bytes, upload: ValidatedUpload = PDF):
    return service.extract_sync(data, upload)


def result(text="word", confidence=90.0, words=1, reliable=None) -> OcrResult:
    return OcrResult(text, confidence, words, words if reliable is None else reliable)


class TestNativeVersusOcr:
    def test_native_pdf_never_calls_ocr(self, settings, fake_ocr):
        response = run(DocumentService(settings, fake_ocr), docs.make_native_pdf([[LONG_TEXT], [LONG_TEXT]]))
        assert fake_ocr.calls == []
        assert [p.method for p in response.pages] == ["native_text", "native_text"]
        assert all(p.confidence is None for p in response.pages)
        assert response.metadata.native_pages == 2 and response.metadata.ocr_pages == 0
        assert response.metadata.render_dpi is None  # nothing was rendered

    def test_scanned_pdf_is_ocrd_page_by_page(self, settings, fake_ocr):
        response = run(DocumentService(settings, fake_ocr), docs.make_scanned_pdf([["a"], ["b"], ["c"]]))
        assert len(fake_ocr.calls) == 3
        assert [p.method for p in response.pages] == ["ocr"] * 3
        assert response.metadata.ocr_pages == 3
        assert response.metadata.render_dpi == settings.render_dpi

    def test_mixed_pdf_preserves_order_and_method_per_page(self, settings, fake_ocr):
        fake_ocr.text = "scanned content"
        data = docs.make_mixed_pdf(
            [("native", [LONG_TEXT]), ("scanned", ["x"]), ("native", [LONG_TEXT + " two"]), ("scanned", ["y"])]
        )
        response = run(DocumentService(settings, fake_ocr), data)
        assert [(p.page_number, p.method) for p in response.pages] == [
            (1, "native_text"), (2, "ocr"), (3, "native_text"), (4, "ocr"),
        ]
        assert len(fake_ocr.calls) == 2
        assert response.full_text == "\n\n".join(
            [LONG_TEXT, "scanned content", LONG_TEXT + " two", "scanned content"]
        )

    def test_short_text_only_page_stays_native(self, settings, fake_ocr):
        response = run(DocumentService(settings, fake_ocr), docs.make_native_pdf([["Page 3"]]))
        assert fake_ocr.calls == []
        assert response.pages[0].text == "Page 3"

    def test_blank_page_is_ocrd_and_returns_empty_text(self, settings):
        engine = FakeOcrEngine(text="", confidence=None)
        response = run(DocumentService(settings, engine), docs.make_blank_pdf())
        assert response.status == "success"
        assert response.pages[0].text == ""

    def test_ocr_mode_force_ocrs_native_pages(self, fake_ocr):
        service = DocumentService(make_settings(ocr_mode="force"), fake_ocr)
        response = run(service, docs.make_native_pdf([[LONG_TEXT]]))
        assert len(fake_ocr.calls) == 1
        assert response.pages[0].method == "ocr"

    def test_ocr_mode_never_skips_ocr_for_scans(self, fake_ocr):
        service = DocumentService(make_settings(ocr_mode="never"), fake_ocr)
        response = run(service, docs.make_scanned_pdf([["scan"]]))
        assert fake_ocr.calls == []
        assert response.pages[0].method == "native_text"
        assert response.pages[0].text == ""

    def test_images_are_always_ocrd_and_use_the_same_schema(self, settings, fake_ocr):
        png = docs.image_bytes(docs.render_text_image(["x"], size=(300, 300)))
        response = run(DocumentService(settings, fake_ocr), png, PNG)
        assert response.document_type == "image"
        assert response.detected_format == "png"
        assert response.total_pages == 1
        assert response.pages[0].method == "ocr"
        assert response.pages[0].confidence == 90.0
        pdf_response = run(DocumentService(settings, fake_ocr), docs.make_scanned_pdf([["a"]]))
        assert response.model_dump().keys() == pdf_response.model_dump().keys()


class TestPreprocessingRetry:
    @pytest.fixture
    def scan(self):
        return docs.make_scanned_pdf([["Some page text"]])

    def test_confident_result_is_not_retried(self, settings, scan):
        engine = FakeOcrEngine(confidence=95)
        response = run(DocumentService(settings, engine), scan)
        assert len(engine.calls) == 1
        assert response.pages[0].preprocessing_applied == []

    def test_low_confidence_triggers_preprocessed_retry_and_keeps_better_result(self, settings, scan):
        engine = FakeOcrEngine()
        engine.rotation = 90
        engine.responses = [result("garbled", 30, words=5, reliable=0), result("clean text", 92, words=5)]
        response = run(DocumentService(settings, engine), scan)
        assert len(engine.calls) == 2
        page = response.pages[0]
        assert page.text == "clean text"
        assert page.confidence == 92
        assert "rotate_90" in page.preprocessing_applied

    def test_worse_retry_is_discarded(self, settings, scan):
        engine = FakeOcrEngine()
        engine.rotation = 90
        engine.responses = [result("baseline", 60, words=10, reliable=6), result("worse", 80, words=2)]
        page = run(DocumentService(settings, engine), scan).pages[0]
        assert page.text == "baseline"
        assert page.preprocessing_applied == []

    def test_retry_disabled_by_setting(self, scan):
        engine = FakeOcrEngine(confidence=20)
        engine.rotation = 90
        run(DocumentService(make_settings(preprocessing_enabled=False), engine), scan)
        assert len(engine.calls) == 1

    def test_blank_page_is_not_retried(self, settings):
        engine = FakeOcrEngine(text="", confidence=None)
        run(DocumentService(settings, engine), docs.make_blank_pdf())
        assert len(engine.calls) == 1


class TestFailures:
    def test_one_failed_page_gives_partial_status(self, settings):
        class FailsOnSecond(FakeOcrEngine):
            def recognize(self, image):
                if len(self.calls) == 1:
                    self.calls.append(image.size)
                    raise OcrTimeoutError()
                return super().recognize(image)

        response = run(DocumentService(settings, FailsOnSecond()), docs.make_scanned_pdf([["a"], ["b"], ["c"]]))
        assert response.status == "partial"
        assert [p.status for p in response.pages] == ["success", "failed", "success"]
        failed = response.pages[1]
        assert failed.text == "" and failed.error == OcrTimeoutError.default_message
        assert response.metadata.failed_pages == 1
        assert response.full_text == "fake text\n\nfake text"

    def test_all_pages_failing_raises_the_underlying_error(self, settings):
        engine = FakeOcrEngine(error=OcrTimeoutError())
        with pytest.raises(OcrTimeoutError):
            run(DocumentService(settings, engine), docs.make_scanned_pdf([["a"], ["b"]]))

    def test_unexpected_page_error_does_not_leak_details(self, settings):
        engine = FakeOcrEngine(error=ValueError("secret internal detail"))
        engine_ok_page = docs.make_mixed_pdf([("native", [LONG_TEXT]), ("scanned", ["x"])])
        response = run(DocumentService(settings, engine), engine_ok_page)
        assert response.status == "partial"
        assert "secret" not in (response.pages[1].error or "")

    def test_engine_unavailable_fails_the_whole_request(self, settings):
        engine = FakeOcrEngine(error=OcrEngineUnavailableError())
        data = docs.make_mixed_pdf([("native", [LONG_TEXT]), ("scanned", ["x"])])
        with pytest.raises(OcrEngineUnavailableError):
            run(DocumentService(settings, engine), data)

    def test_image_ocr_failure_is_raised(self, settings):
        png = docs.image_bytes(docs.render_text_image(["x"], size=(300, 300)))
        with pytest.raises(OcrTimeoutError):
            run(DocumentService(settings, FakeOcrEngine(error=OcrTimeoutError())), png, PNG)

    def test_document_deadline_marks_remaining_pages_failed(self):
        class SlowFirstPage(FakeOcrEngine):
            def recognize(self, image):
                time.sleep(1.3)
                return super().recognize(image)

        settings = make_settings(document_timeout_seconds=1)
        response = run(DocumentService(settings, SlowFirstPage()), docs.make_scanned_pdf([["a"], ["b"], ["c"]]))
        assert response.status == "partial"
        assert [p.status for p in response.pages] == ["success", "failed", "failed"]
        assert "time limit" in response.pages[1].error


class TestConcurrency:
    def test_concurrent_documents_are_capped_by_semaphore(self):
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        class Tracking(FakeOcrEngine):
            def recognize(self, image):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.15)
                with lock:
                    state["active"] -= 1
                return super().recognize(image)

        service = DocumentService(make_settings(max_concurrent_ocr=2), Tracking())
        data = docs.make_scanned_pdf([["a"]])

        async def burst():
            return await asyncio.gather(*(service.extract(data, PDF) for _ in range(6)))

        responses = asyncio.run(burst())
        assert len(responses) == 6
        assert state["peak"] == 2

    def test_server_busy_when_no_slot_frees_up_in_time(self):
        service = DocumentService(
            make_settings(max_concurrent_ocr=1, queue_timeout_seconds=1), FakeOcrEngine()
        )

        async def scenario():
            await service._slots.acquire()  # occupy the only slot
            with pytest.raises(ServerBusyError):
                await service.extract(docs.make_blank_pdf(), PDF)

        asyncio.run(scenario())

    def test_slot_is_released_after_failure(self):
        service = DocumentService(
            make_settings(max_concurrent_ocr=1), FakeOcrEngine(error=OcrTimeoutError())
        )

        async def scenario():
            for _ in range(3):
                with pytest.raises(OcrTimeoutError):
                    await service.extract(docs.make_scanned_pdf([["a"]]), PDF)

        asyncio.run(scenario())
