import asyncio
import io

import pytest
from starlette.datastructures import UploadFile

from app.core.exceptions import EmptyFileError, FileTooLargeError, UnsupportedFormatError
from app.utils.file_validation import (
    DocumentFormat,
    detect_format,
    read_upload_limited,
    sanitize_filename,
    validate_upload,
)
from tests import document_factory as docs

PNG = docs.image_bytes(docs.render_text_image(["x"], size=(50, 50)))
JPG = docs.image_bytes(docs.render_text_image(["x"], size=(50, 50)), "JPEG")
PDF = docs.make_blank_pdf()


class TestDetectFormat:
    def test_detects_png_jpeg_pdf(self):
        assert detect_format(PNG) is DocumentFormat.PNG
        assert detect_format(JPG) is DocumentFormat.JPEG
        assert detect_format(PDF) is DocumentFormat.PDF

    def test_pdf_header_allowed_after_leading_junk(self):
        assert detect_format(b"\r\n  junk" + PDF) is DocumentFormat.PDF

    @pytest.mark.parametrize("data", [b"GIF89a....", b"MZ\x90\x00exe", b"<html></html>", b"plain text"])
    def test_unsupported_content_returns_none(self, data):
        assert detect_format(data) is None


class TestValidateUpload:
    def test_valid_files(self):
        assert validate_upload("a.png", "image/png", PNG).format is DocumentFormat.PNG
        assert validate_upload("a.JPG", "image/jpeg", JPG).format is DocumentFormat.JPEG
        assert validate_upload("a.jpeg", "image/jpeg", JPG).format is DocumentFormat.JPEG
        assert validate_upload("a.pdf", "application/pdf", PDF).format is DocumentFormat.PDF

    def test_missing_or_generic_mime_is_accepted_when_content_is_valid(self):
        assert validate_upload("a.png", None, PNG).format is DocumentFormat.PNG
        assert validate_upload("a.png", "application/octet-stream", PNG).format is DocumentFormat.PNG

    def test_mime_parameters_are_ignored(self):
        assert validate_upload("a.png", "image/png; charset=binary", PNG)

    def test_empty_file_rejected(self):
        with pytest.raises(EmptyFileError):
            validate_upload("a.png", "image/png", b"")

    @pytest.mark.parametrize("name", ["a.gif", "a.txt", "a.exe", "noextension", "a.pdf.exe"])
    def test_unsupported_extension_rejected(self, name):
        with pytest.raises(UnsupportedFormatError):
            validate_upload(name, None, PNG)

    def test_executable_disguised_as_png_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            validate_upload("evil.png", "image/png", b"MZ\x90\x00" + b"\x00" * 100)

    def test_extension_must_match_real_content(self):
        with pytest.raises(UnsupportedFormatError, match="extension"):
            validate_upload("photo.jpg", "image/jpeg", PNG)
        with pytest.raises(UnsupportedFormatError, match="extension"):
            validate_upload("doc.pdf", "application/pdf", PNG)

    def test_mime_must_match_real_content(self):
        with pytest.raises(UnsupportedFormatError, match="content type"):
            validate_upload("a.png", "application/pdf", PNG)
        with pytest.raises(UnsupportedFormatError, match="content type"):
            validate_upload("a.pdf", "text/html", PDF)

    def test_filename_is_sanitised_in_result(self):
        assert validate_upload("../../etc/pass wd.png", "image/png", PNG).filename == "pass wd.png"


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report.pdf", "report.pdf"),
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\me\\scan.png", "scan.png"),
            ("/abs/path/file.jpg", "file.jpg"),
            ("evil\x00name.png", "evilname.png"),
            ("bad<>:\"|?*name.png", "bad_______name.png"),
            ("..", "upload"),
            ("   ", "upload"),
            ("", "upload"),
            (None, "upload"),
        ],
    )
    def test_cases(self, raw, expected):
        assert sanitize_filename(raw) == expected

    def test_long_names_are_truncated_but_keep_extension(self):
        result = sanitize_filename("a" * 500 + ".pdf")
        assert len(result) <= 120
        assert result.endswith(".pdf")

    def test_unicode_names_are_preserved(self):
        assert sanitize_filename("रिपोर्ट.pdf") == "रिपोर्ट.pdf"


class TestReadUploadLimited:
    @staticmethod
    def _upload(data: bytes) -> UploadFile:
        return UploadFile(file=io.BytesIO(data), filename="a.bin", size=None)

    def test_reads_within_limit(self):
        data = asyncio.run(read_upload_limited(self._upload(b"x" * 100), 100))
        assert data == b"x" * 100

    def test_rejects_over_limit_while_streaming(self):
        with pytest.raises(FileTooLargeError):
            asyncio.run(read_upload_limited(self._upload(b"x" * 101), 100, chunk_size=10))

    def test_rejects_when_declared_size_exceeds_limit(self):
        upload = UploadFile(file=io.BytesIO(b"x"), filename="a.bin", size=10_000)
        with pytest.raises(FileTooLargeError):
            asyncio.run(read_upload_limited(upload, 100))
