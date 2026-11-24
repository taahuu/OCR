"""Upload validation: size-capped reads, signature sniffing and safe filenames.

Uploaded files are untrusted. The *actual bytes* decide the file format; the
filename extension and the ``Content-Type`` header must agree with them but are
never trusted on their own.
"""

from __future__ import annotations

import unicodedata
import io
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFormatError,
)


class DocumentFormat(str, Enum):
    """Formats accepted by the API."""

    PDF = "pdf"
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"
    DOCX = "docx"

    @property
    def document_type(self) -> str:
        """Coarse category used in API responses: ``pdf`` or ``image``."""
        if self is DocumentFormat.PDF:
            return "pdf"
        if self is DocumentFormat.DOCX:
            return "word"
        return "image"


EXTENSION_TO_FORMAT: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".png": DocumentFormat.PNG,
    ".jpg": DocumentFormat.JPEG,
    ".jpeg": DocumentFormat.JPEG,
    ".tif": DocumentFormat.TIFF,
    ".tiff": DocumentFormat.TIFF,
    ".bmp": DocumentFormat.BMP,
    ".webp": DocumentFormat.WEBP,
    ".docx": DocumentFormat.DOCX,
}

FORMAT_TO_MIME_TYPES: dict[DocumentFormat, frozenset[str]] = {
    DocumentFormat.PDF: frozenset({"application/pdf"}),
    DocumentFormat.PNG: frozenset({"image/png"}),
    DocumentFormat.JPEG: frozenset({"image/jpeg", "image/jpg", "image/pjpeg"}),
    DocumentFormat.TIFF: frozenset({"image/tiff", "image/tif"}),
    DocumentFormat.BMP: frozenset({"image/bmp", "image/x-ms-bmp"}),
    DocumentFormat.WEBP: frozenset({"image/webp"}),
    DocumentFormat.DOCX: frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
}

# Clients that cannot determine a type send these; the signature check still
# decides the real format, so they are treated as "not specified".
_UNSPECIFIED_MIME_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_PDF_SIGNATURE = b"%PDF-"
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*")
_BMP_SIGNATURE = b"BM"
_RIFF_SIGNATURE = b"RIFF"
_PDF_SIGNATURE_WINDOW = 1024  # the PDF spec allows leading junk before the header

_MAX_FILENAME_LENGTH = 120
_FALLBACK_FILENAME = "upload"
_ALLOWED_PUNCTUATION = frozenset("._- ()[]")


class _AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class ValidatedUpload:
    """Result of a successful validation."""

    filename: str
    format: DocumentFormat
    size_bytes: int


def detect_format(data: bytes) -> DocumentFormat | None:
    """Identify the format from magic bytes, or ``None`` if unsupported."""
    if data.startswith(_PNG_SIGNATURE):
        return DocumentFormat.PNG
    if data.startswith(_JPEG_SIGNATURE):
        return DocumentFormat.JPEG
    if _PDF_SIGNATURE in data[:_PDF_SIGNATURE_WINDOW]:
        return DocumentFormat.PDF
    if data.startswith(_TIFF_SIGNATURES):
        return DocumentFormat.TIFF
    if data.startswith(_BMP_SIGNATURE):
        return DocumentFormat.BMP
    if len(data) >= 12 and data.startswith(_RIFF_SIGNATURE) and data[8:12] == b"WEBP":
        return DocumentFormat.WEBP
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" in names and "word/document.xml" in names:
                    return DocumentFormat.DOCX
        except (zipfile.BadZipFile, OSError):
            return None
    return None


def _is_safe_filename_char(char: str) -> bool:
    """Letters, digits and combining marks (needed for Indic scripts) are kept."""
    return (
        char.isalnum()
        or char in _ALLOWED_PUNCTUATION
        or unicodedata.category(char).startswith("M")
    )


def sanitize_filename(raw_name: str | None) -> str:
    """Return a safe display filename.

    Strips any directory components (POSIX and Windows), control and special
    characters, and limits the length. The result is only ever echoed back and
    never used as a filesystem path.
    """
    if not raw_name:
        return _FALLBACK_FILENAME
    name = PureWindowsPath(PurePosixPath(raw_name.replace("\x00", "")).name).name
    name = "".join(ch if _is_safe_filename_char(ch) else "_" for ch in name).strip(" .")
    if not name:
        return _FALLBACK_FILENAME
    if len(name) > _MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 10:
            name = stem[: _MAX_FILENAME_LENGTH - len(ext) - 1] + "." + ext
        else:
            name = name[:_MAX_FILENAME_LENGTH]
    return name


def _extension_of(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


async def read_upload_limited(
    upload: _AsyncReadable, max_bytes: int, chunk_size: int = 1024 * 1024
) -> bytes:
    """Read an upload in chunks, aborting as soon as ``max_bytes`` is exceeded."""
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and declared_size > max_bytes:
        raise FileTooLargeError()

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError()
        chunks.append(chunk)
    return b"".join(chunks)


def validate_upload(
    filename: str | None, content_type: str | None, data: bytes
) -> ValidatedUpload:
    """Validate extension, signature and MIME type of an uploaded file.

    Raises:
        EmptyFileError: no bytes were uploaded.
        UnsupportedFormatError: unsupported extension, unrecognised signature,
            or extension / MIME type that contradicts the real content.
    """
    if not data:
        raise EmptyFileError()

    safe_name = sanitize_filename(filename)

    extension_format = EXTENSION_TO_FORMAT.get(_extension_of(safe_name))
    if extension_format is None:
        raise UnsupportedFormatError()

    actual_format = detect_format(data)
    if actual_format is None:
        raise UnsupportedFormatError(
            "The file content is not a supported PDF, image or DOCX document."
        )
    if actual_format is not extension_format:
        raise UnsupportedFormatError(
            "The file extension does not match the file content."
        )

    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in _UNSPECIFIED_MIME_TYPES and mime not in FORMAT_TO_MIME_TYPES[actual_format]:
        raise UnsupportedFormatError(
            "The declared content type does not match the file content."
        )

    return ValidatedUpload(filename=safe_name, format=actual_format, size_bytes=len(data))
