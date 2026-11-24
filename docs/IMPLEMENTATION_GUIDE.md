# OCR document extraction implementation

## What this version supports

- PDF: native selectable text, scanned pages through Tesseract, per-page processing, PDF table detection where PyMuPDF can identify the table.
- Images: PNG, JPEG, TIFF, BMP, and WEBP. TIFF currently processes its first frame only.
- Word: DOCX paragraphs, headers/footers, and tables are extracted locally with `python-docx`; DOCX itself does not require OCR.
- Printed text only. Handwriting recognition is not included.
- Local heuristic classification for resume, invoice, bank statement, form, and general documents.

## File-by-file changes

### Modified files

| File | Change | Why |
|---|---|---|
| `requirements.txt` | Added `python-docx==1.2.0` | Read DOCX paragraphs and tables without a cloud service. |
| `app/utils/file_validation.py` | Added DOCX, TIFF, BMP, WEBP signature/extension/MIME validation | Keep file type validation based on bytes, not filename alone. |
| `app/services/ocr_service.py` | Preserve each recognized word's text, confidence, bounding box, and Tesseract block/paragraph/line IDs | Expose OCR metadata instead of discarding it. |
| `app/services/pdf_service.py` | Added PyMuPDF table extraction | Extract PDF tables when the installed MuPDF table detector recognizes them. |
| `app/services/document_service.py` | Added DOCX routing, layout/field extraction, word boxes, and OCR table reconstruction | Coordinate the new processing steps while retaining the existing PDF/image path. |
| `app/schemas/ocr.py` | Added `layout`, `words`, `tables`, and `structured_data`; expanded format values | Return richer results while keeping all existing response fields. |
| `app/api/routes/ocr.py` | Updated accepted format list and endpoint descriptions | Document supported uploads in Swagger. |
| `app/main.py` | Updated API description | Explain local processing and structured outputs. |
| `app/core/exceptions.py` | Updated unsupported-format message | Show the actual accepted formats. |
| `README.md` | Updated setup, supported formats, limitations | Keep developer instructions aligned with implementation. |
| `docs/sample_response.json` | Added the new response fields | Provide a current response example. |
| `tests/test_ocr_service.py`, `tests/test_ocr_api.py` | Updated response expectations and added DOCX endpoint coverage | Protect old behavior and verify new fields. |

### New files

| File | Responsibility |
|---|---|
| `app/services/docx_service.py` | Safely validates and extracts DOCX paragraphs, headers/footers, and tables. Checks expanded archive size before parsing. |
| `app/services/layout_service.py` | Produces paragraphs, heading candidates, and line counts from extracted text. |
| `app/services/information_extraction_service.py` | Local rules for document classification and common resume/invoice/bank-statement/form fields. Does not invent missing values. |
| `app/services/table_extraction_service.py` | Best-effort table reconstruction from OCR word coordinates when repeated rows/columns are visible. |
| `tests/test_docx_and_structured_extraction.py` | Tests DOCX validation/extraction and common field extraction. |

## Response additions

The existing `status`, `filename`, `document_type`, `detected_format`, `total_pages`, `pages`, `full_text`, and `metadata` fields remain. New fields are additive:

- `pages[].layout`: paragraphs, heading candidates, and line count.
- `pages[].words`: word text, confidence (null for native PDF text), pixel/page coordinates, and Tesseract line identifiers.
- `tables`: detected table headers and rows.
- `structured_data`: detected document category and type-specific fields. Missing fields remain null or empty.

`document_type` identifies the input container (`pdf`, `image`, or `word`). The semantic category is `structured_data.document_type` (`resume`, `invoice`, `bank_statement`, `form`, or `general_document`).

## Run locally

```bash
# Ubuntu / Debian: install the local OCR executable once
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd

# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # optional
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and call `POST /api/v1/ocr/extract` with multipart field `file`.

## Tests

```bash
pytest -q
pytest -m "not integration" -q
pytest -m integration -q
```

## Important accuracy boundaries

This is a local extraction pipeline, not a universal document-understanding model. OCR output depends on scan quality and installed Tesseract language packs. Table reconstruction from scanned images is heuristic; PDF table extraction depends on PyMuPDF's detector. The field extraction rules are intentionally explainable and can miss unusual layouts or labels. Always retain and review `full_text` and source documents for high-impact decisions, especially financial or identity data.

DOCX embedded images, comments, tracked changes, and legacy `.doc` files are not processed. TIFF multi-page files currently process only the first frame. PDF pages containing both a small amount of selectable text and a large scanned image may need `OCR_MODE=force` to OCR the whole page; the existing `auto` mode may choose native text based on its current threshold.
