"""Domain exceptions.

Each exception carries an HTTP status code, a stable machine-readable code and
a *safe* client-facing message. Internal details (paths, library messages) must
never be placed in these messages.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, client-reportable errors."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An internal error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class EmptyFileError(AppError):
    status_code = 400
    code = "empty_file"
    default_message = "The uploaded file is empty."


class CorruptDocumentError(AppError):
    status_code = 400
    code = "corrupt_document"
    default_message = "The uploaded file is corrupted or cannot be read."


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"
    default_message = "The uploaded file exceeds the maximum allowed size."


class UnsupportedFormatError(AppError):
    status_code = 415
    code = "unsupported_format"
    default_message = "Unsupported file format. Upload PDF, PNG, JPEG, TIFF, BMP, WEBP, or DOCX."


class PageLimitExceededError(AppError):
    status_code = 422
    code = "page_limit_exceeded"
    default_message = "The PDF has more pages than the allowed maximum."


class ImageTooLargeError(AppError):
    status_code = 422
    code = "image_too_large"
    default_message = "The image dimensions exceed the allowed maximum."


class EncryptedPdfError(AppError):
    status_code = 422
    code = "encrypted_pdf"
    default_message = "Password-protected PDFs are not supported."


class ServerBusyError(AppError):
    status_code = 503
    code = "server_busy"
    default_message = "The server is busy processing other documents. Retry shortly."


class OcrTimeoutError(AppError):
    status_code = 504
    code = "ocr_timeout"
    default_message = "OCR processing exceeded the time limit."


class OcrEngineUnavailableError(AppError):
    status_code = 503
    code = "ocr_engine_unavailable"
    default_message = "The OCR engine is not available on the server."


class OcrEngineError(AppError):
    status_code = 500
    code = "ocr_failed"
    default_message = "The OCR engine failed to process the document."


class ProcessingError(AppError):
    status_code = 500
    code = "processing_failed"
    default_message = "The document could not be processed."
