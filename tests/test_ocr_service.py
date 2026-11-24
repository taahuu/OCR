import pytest
import pytesseract
from PIL import Image

from app.core.exceptions import OcrEngineError, OcrEngineUnavailableError, OcrTimeoutError
from app.services.ocr_service import TesseractOcrService
from tests import document_factory as docs
from tests.conftest import make_settings, requires_tesseract


def _tsv(*words):
    """Build the dict pytesseract returns. Each word: (text, conf, block, par, line)."""
    keys = ("level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text")
    data = {key: [] for key in keys}
    # a structural (non-word) row that must be ignored
    for key, value in zip(keys, (4, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, "")):
        data[key].append(value)
    for index, (text, conf, block, par, line) in enumerate(words, start=1):
        for key, value in zip(keys, (5, 1, block, par, line, index, index * 10, 20, 8, 12, conf, text)):
            data[key].append(value)
    return data


class TestParsing:
    def test_rebuilds_lines_and_paragraphs(self):
        data = _tsv(
            ("Hello", 95, 1, 1, 1), ("world", 90, 1, 1, 1),
            ("Second", 85, 1, 1, 2), ("line", 80, 1, 1, 2),
            ("New", 70, 2, 1, 1), ("block", 60, 2, 1, 1),
        )
        result = TesseractOcrService._parse_data(data)
        assert result.text == "Hello world\nSecond line\n\nNew block"
        assert result.word_count == 6
        assert result.words[0]["left"] == 10
        assert result.words[0]["text"] == "Hello"

    def test_confidence_is_mean_of_word_confidences(self):
        data = _tsv(("a", 100, 1, 1, 1), ("b", 50, 1, 1, 1))
        result = TesseractOcrService._parse_data(data)
        assert result.confidence == 75.0
        assert result.reliable_word_count == 1  # only the 100 is >= 60

    def test_negative_confidences_and_blank_words_are_ignored(self):
        data = _tsv(("ok", 90, 1, 1, 1), ("   ", 99, 1, 1, 1), ("bad", -1, 1, 1, 1))
        result = TesseractOcrService._parse_data(data)
        assert result.text == "ok bad"
        assert result.confidence == 90.0

    def test_no_words_gives_none_confidence(self):
        result = TesseractOcrService._parse_data(_tsv())
        assert (result.text, result.confidence, result.word_count) == ("", None, 0)
        assert not result.has_text

    def test_string_confidences_from_older_pytesseract_are_handled(self):
        data = _tsv(("x", "88", 1, 1, 1))
        assert TesseractOcrService._parse_data(data).confidence == 88.0


class TestErrorMapping:
    @pytest.fixture
    def service(self):
        return TesseractOcrService(make_settings())

    def _patch(self, monkeypatch, exc):
        def boom(*args, **kwargs):
            raise exc

        monkeypatch.setattr(pytesseract, "image_to_data", boom)

    def test_timeout(self, service, monkeypatch):
        self._patch(monkeypatch, RuntimeError("Tesseract process timeout"))
        with pytest.raises(OcrTimeoutError):
            service.recognize(Image.new("L", (10, 10)))

    def test_missing_binary(self, service, monkeypatch):
        self._patch(monkeypatch, pytesseract.TesseractNotFoundError())
        with pytest.raises(OcrEngineUnavailableError):
            service.recognize(Image.new("L", (10, 10)))

    def test_engine_failure_message_does_not_leak_details(self, service, monkeypatch):
        self._patch(monkeypatch, pytesseract.TesseractError(1, "secret /internal/path failure"))
        with pytest.raises(OcrEngineError) as info:
            service.recognize(Image.new("L", (10, 10)))
        assert "secret" not in info.value.message
        assert "/internal" not in info.value.message

    def test_other_runtime_errors_are_engine_errors(self, service, monkeypatch):
        self._patch(monkeypatch, RuntimeError("something else"))
        with pytest.raises(OcrEngineError):
            service.recognize(Image.new("L", (10, 10)))

    def test_rotation_detection_failure_returns_zero(self, service, monkeypatch):
        def boom(*args, **kwargs):
            raise pytesseract.TesseractError(1, "Too few characters")

        monkeypatch.setattr(pytesseract, "image_to_osd", boom)
        assert service.detect_rotation(Image.new("L", (10, 10))) == 0

    def test_low_confidence_rotation_is_ignored(self, service, monkeypatch):
        monkeypatch.setattr(
            pytesseract, "image_to_osd", lambda *a, **k: {"rotate": 90, "orientation_conf": 0.5}
        )
        assert service.detect_rotation(Image.new("L", (10, 10))) == 0

    def test_confident_rotation_is_returned(self, service, monkeypatch):
        monkeypatch.setattr(
            pytesseract, "image_to_osd", lambda *a, **k: {"rotate": "270", "orientation_conf": "9.1"}
        )
        assert service.detect_rotation(Image.new("L", (10, 10))) == 270


class TestStatus:
    def test_unavailable_when_tesseract_is_missing(self, monkeypatch):
        def boom():
            raise pytesseract.TesseractNotFoundError()

        monkeypatch.setattr(pytesseract, "get_tesseract_version", boom)
        status = TesseractOcrService(make_settings()).status()
        assert (status.available, status.language_available) == (False, False)

    @requires_tesseract
    def test_reports_real_engine(self):
        status = TesseractOcrService(make_settings()).status()
        assert status.available and status.version
        assert status.language_available

    @requires_tesseract
    def test_unknown_language_reported_unavailable(self):
        status = TesseractOcrService(make_settings(ocr_language="zzz")).status()
        assert status.available and not status.language_available


@pytest.mark.integration
@requires_tesseract
class TestRealTesseract:
    """Runs the real engine on generated pages with known text."""

    @pytest.fixture(scope="class")
    def service(self):
        return TesseractOcrService(make_settings())

    def test_recognises_known_text_with_high_confidence(self, service):
        image = docs.render_text_image(docs.SAMPLE_LINES).convert("L")
        result = service.recognize(image)
        for expected in ("INVOICE", "Acme", "Corporation", "1250", "business"):
            assert expected in result.text
        assert result.confidence is not None and result.confidence > 85
        assert result.word_count >= 10

    def test_blank_image_returns_empty_text(self, service):
        result = service.recognize(Image.new("L", (800, 600), 255))
        assert result.text == ""
        assert result.confidence is None

    @pytest.mark.parametrize("rotation", [90, 180])
    def test_detects_upside_down_and_sideways_pages(self, service, rotation):
        upright = docs.render_text_image(docs.SAMPLE_LINES).convert("L")
        # PIL rotates counter-clockwise, so the clockwise fix equals the same angle.
        rotated = upright.rotate(rotation, expand=True, fillcolor=255)
        assert service.detect_rotation(rotated) == rotation

    def test_per_call_timeout_kills_tesseract(self):
        service = TesseractOcrService(make_settings(ocr_timeout_seconds=1))
        busy = docs.render_text_image(["word " * 8] * 30, size=(6000, 8000), font_size=30)
        with pytest.raises(OcrTimeoutError):
            service.recognize(busy.convert("L"))
