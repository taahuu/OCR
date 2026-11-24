"""Pydantic response schemas (identical for image and PDF uploads)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WordBox(BaseModel):
    text: str
    confidence: float | None = None
    left: int
    top: int
    width: int
    height: int
    block_number: int
    paragraph_number: int
    line_number: int


class PageResult(BaseModel):
    """Extraction result for a single page (images count as one page)."""

    page_number: int = Field(ge=1, description="1-based page number.")
    method: Literal["native_text", "ocr"] = Field(
        description="How the text was obtained: embedded PDF text or OCR."
    )
    status: Literal["success", "failed"]
    text: str = Field(description="Extracted text. Empty if the page is blank or failed.")
    confidence: float | None = Field(
        default=None,
        description=(
            "Mean Tesseract word confidence (0-100) for OCR pages; null for native "
            "text. A heuristic only - it does not guarantee correctness."
        ),
    )
    preprocessing_applied: list[str] = Field(
        default_factory=list,
        description="Image preprocessing steps used to produce the returned OCR text.",
    )
    error: str | None = Field(default=None, description="Safe error message if status=failed.")
    layout: dict = Field(default_factory=dict, description="Paragraphs, headings and line counts.")
    words: list[WordBox] = Field(default_factory=list, description="OCR word boxes in source-image pixels; empty for native PDF text.")


class ExtractionMetadata(BaseModel):
    """Processing details for the whole document."""

    processing_time_ms: int
    ocr_language: str
    native_pages: int
    ocr_pages: int
    failed_pages: int
    render_dpi: int | None = Field(
        default=None, description="PDF render resolution; null for image uploads."
    )


class ExtractionResponse(BaseModel):
    """Response of ``POST /api/v1/ocr/extract``."""

    status: Literal["success", "partial", "failed"] = Field(
        description="'partial' means at least one page failed but others succeeded."
    )
    filename: str = Field(description="Sanitised original filename.")
    document_type: Literal["pdf", "image", "word"]
    detected_format: Literal["pdf", "png", "jpeg", "tiff", "bmp", "webp", "docx"] = Field(
        description="Format detected from the file signature (not the filename)."
    )
    total_pages: int
    pages: list[PageResult]
    full_text: str = Field(description="Text of all successful pages joined in page order.")
    metadata: ExtractionMetadata
    tables: list[dict] = Field(default_factory=list, description="Tables detected in supported source documents.")
    structured_data: dict = Field(default_factory=dict, description="Heuristic document fields; missing values are not invented.")


class SupportedFormat(BaseModel):
    extension: str
    mime_types: list[str]
    description: str


class SupportedFormatsResponse(BaseModel):
    formats: list[SupportedFormat]
    max_upload_size_mb: int
    max_pdf_pages: int
    max_image_pixels: int
    max_image_dimension: int
    ocr_language: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
