"""OCR endpoints. Routes stay thin: validate, delegate to the service, return."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings
from app.dependencies.providers import get_app_settings, get_document_service
from app.schemas.ocr import (
    ErrorResponse,
    ExtractionResponse,
    SupportedFormat,
    SupportedFormatsResponse,
)
from app.services.document_service import DocumentService
from app.utils.file_validation import read_upload_limited, validate_upload

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.get(
    "/supported-formats",
    response_model=SupportedFormatsResponse,
    summary="List supported file formats and active limits",
)
async def supported_formats(
    settings: Settings = Depends(get_app_settings),
) -> SupportedFormatsResponse:
    return SupportedFormatsResponse(
        formats=[
            SupportedFormat(extension=".pdf", mime_types=["application/pdf"], description="Selectable-text, scanned, or mixed PDF."),
            SupportedFormat(extension=".png", mime_types=["image/png"], description="PNG image."),
            SupportedFormat(extension=".jpg / .jpeg", mime_types=["image/jpeg"], description="JPEG image."),
            SupportedFormat(extension=".tif / .tiff", mime_types=["image/tiff"], description="TIFF image (first frame/page is processed)."),
            SupportedFormat(extension=".bmp", mime_types=["image/bmp"], description="BMP image."),
            SupportedFormat(extension=".webp", mime_types=["image/webp"], description="WEBP image."),
            SupportedFormat(extension=".docx", mime_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"], description="Word document; paragraphs and tables are extracted without OCR."),
        ],
        max_upload_size_mb=settings.max_upload_size_mb,
        max_pdf_pages=settings.max_pdf_pages,
        max_image_pixels=settings.max_image_pixels,
        max_image_dimension=settings.max_image_dimension,
        ocr_language=settings.ocr_language,
    )


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract text and structured information from PDF, image, or DOCX",
    responses={
        400: {"model": ErrorResponse, "description": "Empty or corrupted file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported or mismatched file type"},
        422: {"model": ErrorResponse, "description": "Page/dimension limit exceeded, encrypted PDF, or missing field"},
        503: {"model": ErrorResponse, "description": "Server busy or OCR engine unavailable"},
        504: {"model": ErrorResponse, "description": "Processing time limit exceeded"},
    },
)
async def extract_text(
    file: UploadFile = File(..., description="PDF, PNG, JPEG, TIFF, BMP, WEBP, or DOCX file."),
    settings: Settings = Depends(get_app_settings),
    service: DocumentService = Depends(get_document_service),
) -> ExtractionResponse:
    """Upload a supported document as ``multipart/form-data`` (field name ``file``).

    Native PDF and DOCX text is extracted directly; scanned PDF pages and images
    are OCR'd locally. Additional fields are heuristic and may be empty when absent.
    """
    data = await read_upload_limited(file, settings.max_upload_bytes)
    upload = validate_upload(file.filename, file.content_type, data)
    return await service.extract(data, upload)
