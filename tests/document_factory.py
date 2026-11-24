"""Builders for test documents (images and PDFs) with known text.

Run ``python -m tests.document_factory`` to write sample files into ``samples/``.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

LETTER_POINTS = (612, 792)


def render_text_image(
    lines: list[str],
    size: tuple[int, int] = (1700, 2200),
    font_size: int = 52,
    rotate_degrees: float = 0.0,
    mode: str = "RGB",
) -> Image.Image:
    """Draw ``lines`` of black text on a white page (optionally rotated)."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    y = 150
    for line in lines:
        draw.text((120, y), line, fill="black", font=font)
        y += int(font_size * 1.8)
    if rotate_degrees:
        image = image.rotate(rotate_degrees, expand=False, fillcolor="white", resample=Image.BICUBIC)
    return image.convert(mode) if mode != "RGB" else image


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def make_native_pdf(pages: list[list[str]]) -> bytes:
    """PDF whose pages contain real, selectable text."""
    document = pymupdf.open()
    for lines in pages:
        page = document.new_page(width=LETTER_POINTS[0], height=LETTER_POINTS[1])
        y = 100
        for line in lines:
            page.insert_text((72, y), line, fontsize=14)
            y += 28
    data = document.tobytes(deflate=True, garbage=3)
    document.close()
    return data


def _add_scanned_page(document: pymupdf.Document, lines: list[str]) -> None:
    page = document.new_page(width=LETTER_POINTS[0], height=LETTER_POINTS[1])
    # 1700x2200 px grayscale = 200 DPI on letter, like a typical office scan.
    png = image_bytes(render_text_image(lines, mode="L"))
    page.insert_image(page.rect, stream=png)


def make_scanned_pdf(pages: list[list[str]]) -> bytes:
    """PDF whose pages are pure images (no text layer)."""
    document = pymupdf.open()
    for lines in pages:
        _add_scanned_page(document, lines)
    data = document.tobytes(deflate=True, garbage=3)
    document.close()
    return data


def make_mixed_pdf(pages: list[tuple[str, list[str]]]) -> bytes:
    """PDF mixing pages of kind ``"native"`` and ``"scanned"``."""
    document = pymupdf.open()
    for kind, lines in pages:
        if kind == "native":
            page = document.new_page(width=LETTER_POINTS[0], height=LETTER_POINTS[1])
            y = 100
            for line in lines:
                page.insert_text((72, y), line, fontsize=14)
                y += 28
        else:
            _add_scanned_page(document, lines)
    data = document.tobytes(deflate=True, garbage=3)
    document.close()
    return data


def make_blank_pdf(page_count: int = 1) -> bytes:
    """PDF with blank pages (no text, no images)."""
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page(width=LETTER_POINTS[0], height=LETTER_POINTS[1])
    data = document.tobytes(deflate=True, garbage=3)
    document.close()
    return data


# A structurally valid PDF whose page tree is empty (zero pages).
ZERO_PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n"
)

CORRUPT_PDF = b"%PDF-1.7\n" + b"\x00\x13garbage that is not a pdf structure" * 20


SAMPLE_LINES = [
    "INVOICE 2026-0042",
    "Customer: Acme Corporation",
    "Total due: 1250 dollars",
    "Thank you for your business",
]


def write_samples(target: Path) -> None:
    """Write demo files for manual API testing."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "sample_image.png").write_bytes(image_bytes(render_text_image(SAMPLE_LINES)))
    (target / "sample_image.jpg").write_bytes(
        image_bytes(render_text_image(SAMPLE_LINES), "JPEG")
    )
    (target / "sample_native.pdf").write_bytes(make_native_pdf([SAMPLE_LINES, ["Second page text"]]))
    (target / "sample_scanned.pdf").write_bytes(
        make_scanned_pdf([SAMPLE_LINES, ["Second scanned page"]])
    )
    (target / "sample_mixed.pdf").write_bytes(
        make_mixed_pdf(
            [
                ("native", ["This first page has selectable text and needs no OCR at all."]),
                ("scanned", SAMPLE_LINES),
            ]
        )
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "samples"
    write_samples(out)
    print(f"Wrote samples to {out}")
