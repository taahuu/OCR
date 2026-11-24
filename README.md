# OCR Document Extraction API

A local FastAPI document extraction service for **PDF, PNG, JPEG, TIFF, BMP, WEBP, and DOCX**.
Selectable PDF text and DOCX content are extracted directly. Scanned PDF pages and
images are OCR'd locally with **Tesseract**. A local rules-based layer classifies
common document types (resume, invoice, bank statement, form, or general document)
and returns detected fields, layout, word coordinates, and tables where supported.
No cloud service, API key, or database is required.

- Normal, scanned, multi-page and **mixed** PDFs (each page is decided independently)
- One additive JSON response schema for PDF, image, and DOCX inputs
- OCR word-level bounding boxes and confidence; native PDF word coordinates
- DOCX paragraph/table extraction and PDF table detection; best-effort table reconstruction from OCR coordinates
- Local heuristic structured extraction for resumes, invoices, bank statements, and forms
- Untrusted-upload hardening: size / page / pixel limits, signature checks, safe parsing, no storage
- Conditional image preprocessing (deskew, rotation, denoise, contrast, binarise) used only as a retry, keeping the better of the two OCR results
- Concurrency-limited, non-blocking OCR; everything configurable via environment variables

---

## 1. Quick start (local)

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Developed on 3.12/3.13 |
| Tesseract OCR 5.x | Must include the `eng` and `osd` data (see below) |

Install Tesseract:

```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd

# macOS (Homebrew)
brew install tesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# then set TESSERACT_CMD in .env, e.g.
#   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Set up and run:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime + test dependencies
cp .env.example .env                 # optional; defaults work as-is

uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### Try it

```bash
curl http://127.0.0.1:8000/api/v1/health

curl -F "file=@samples/sample_mixed.pdf" http://127.0.0.1:8000/api/v1/ocr/extract
curl -F "file=@samples/sample_image.png" http://127.0.0.1:8000/api/v1/ocr/extract
curl -F "file=@resume.docx" http://127.0.0.1:8000/api/v1/ocr/extract
```

`samples/` contains generated demo files (regenerate with `python -m tests.document_factory`).
A real response is saved in [`docs/sample_response.json`](docs/sample_response.json).

---

## 2. API

Base path: `/api/v1`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `200` when Tesseract and the configured language are usable, `503` otherwise |
| GET | `/ocr/supported-formats` | Accepted formats and the active limits |
| POST | `/ocr/extract` | Upload one file (`multipart/form-data`, field `file`) |

A batch endpoint is intentionally not included in v1 (clients can call `/extract` repeatedly).

### Response (`POST /ocr/extract`)

```json
{
  "status": "success",
  "filename": "sample_mixed.pdf",
  "document_type": "pdf",
  "detected_format": "pdf",
  "total_pages": 2,
  "pages": [
    {"page_number": 1, "method": "native_text", "status": "success",
     "text": "This first page has selectable text ...", "confidence": null,
     "preprocessing_applied": [], "error": null},
    {"page_number": 2, "method": "ocr", "status": "success",
     "text": "INVOICE 2026-0042\nCustomer: Acme Corporation ...", "confidence": 95.71,
     "preprocessing_applied": [], "error": null}
  ],
  "full_text": "...all successful pages joined in order...",
  "metadata": {"processing_time_ms": 368, "ocr_language": "eng",
               "native_pages": 1, "ocr_pages": 1, "failed_pages": 0, "render_dpi": 300}
}
```

- `status`: `success`, or `partial` if some pages failed (those have `status: "failed"` and a safe `error`).
- `method`: `native_text` (embedded PDF text) or `ocr`.
- `confidence`: mean Tesseract word confidence (0-100) for OCR pages, `null` otherwise.
  **It is a heuristic, not a guarantee of correctness.**
- `preprocessing_applied`: steps that produced the returned text (e.g. `rotate_180`, `deskew_-4.0deg`, `denoise`, `contrast`, `upscale`, `binarize`).
- `filename` is sanitised (directory parts and special characters removed).

### Errors

All errors share one shape: `{"error": {"code": "...", "message": "..."}}`.

| HTTP | `code` | When |
|---|---|---|
| 400 | `empty_file` / `corrupt_document` | Empty upload; unreadable PDF/image; PDF with no pages |
| 413 | `file_too_large` | Larger than `MAX_UPLOAD_SIZE_MB` |
| 415 | `unsupported_format` | Bad extension, content is not PDF/JPEG/PNG, or extension / MIME type contradicts the content |
| 422 | `page_limit_exceeded` / `image_too_large` / `encrypted_pdf` / `validation_error` | Limits exceeded, password-protected PDF, missing `file` field |
| 503 | `server_busy` / `ocr_engine_unavailable` | No free OCR slot within `QUEUE_TIMEOUT_SECONDS`; Tesseract missing |
| 504 | `ocr_timeout` | Tesseract or the whole document exceeded its time limit |
| 500 | `ocr_failed` / `internal_error` | Generic failure; details are only in the server log |

---

## 3. How it works

```
upload ──► size-capped read ──► signature check (magic bytes, ext, MIME must agree)
                                   │
              ┌────────────────────┴───────────────────┐
              ▼ PDF                                    ▼ image
   open with PyMuPDF (page / encryption limits)   Pillow: header size check,
   for each page, independently:                  decode, EXIF orientation,
     native text >= threshold? ──yes──► use it    flatten alpha, normalise mode
        │ no                                             │
        ▼                                                ▼
   render page to grayscale (DPI, pixel cap) ──────► OCR (Tesseract)
                                                         │ confidence < threshold?
                                                         ▼ yes
                                     detect rotation → preprocess → OCR again
                                     keep whichever result has more reliable words
```

**Normal vs scanned PDFs.** A normal PDF contains text objects, which PyMuPDF reads exactly with no OCR. A scanned PDF page is just an image, so PyMuPDF finds no text and the page is rendered and OCR'd. A page counts as *native* when it has at least `NATIVE_TEXT_MIN_CHARS` characters, or some text and no images. Mixed PDFs simply get a different decision per page, and page order and numbers are preserved. Scans that already carry an invisible OCR text layer are read as native text.

**Preprocessing.** Tesseract binarises internally and does best on clean pages, so blanket filtering can hurt. The pipeline therefore runs plain grayscale OCR first. Only if the mean confidence is below `LOW_CONFIDENCE_THRESHOLD` (default 70) does it detect rotation (Tesseract OSD), apply *only the corrections that measurements justify*, and run OCR again. It keeps whichever attempt recognised more words with confidence of 60 or higher, so an enhancement that does not help is discarded. That is a heuristic comparison, not a proof of better accuracy. Input images are never modified.

| Step | Triggered when |
|---|---|
| rotate 90/180/270 | Tesseract OSD reports a confident rotation |
| denoise (3x3 median) | Estimated Gaussian noise or salt-and-pepper ratio is high |
| contrast stretch | Paper-vs-ink level difference is small (faded scan) |
| upscale (max 3x) | Long side below `PREPROCESS_MIN_LONG_SIDE` |
| deskew (up to ±10°) | Projection-profile search finds a clear improvement |
| adaptive binarise | Background brightness varies strongly (shadows, gradients) |

Measured on synthetic degradations of one page: 90°/180° rotation 29-31% → 96% confidence; uneven lighting 0 → 14 words recognised; salt-and-pepper noise 65% → 96%; heavy noise on a low-contrast page 0 → 14 words.

**Concurrency.** Blocking work (PyMuPDF, OpenCV, Tesseract) runs in a worker thread; a semaphore (`MAX_CONCURRENT_OCR`) bounds simultaneous documents and further requests wait up to `QUEUE_TIMEOUT_SECONDS` before receiving `503`. Each Tesseract call has a hard timeout (the subprocess is killed) and the whole document has a deadline checked between pages.

### Project layout

```
app/
  main.py                   app factory, lifespan (startup/shutdown logs, engine check)
  api/router.py, routes/    thin HTTP layer: health.py, ocr.py
  core/                     config.py (settings), exceptions.py, error_handlers.py,
                            middleware.py (request size limit), logging.py
  schemas/                  Pydantic response models
  services/
    document_service.py     orchestration, native-vs-OCR decision, retry, concurrency
    pdf_service.py          PyMuPDF: safe open, native text, page rendering
    image_service.py        Pillow: safe decode, orientation, mode normalisation
    preprocessing_service.py  measurements + conditional corrections
    ocr_service.py          Pytesseract wrapper, confidence, OSD, error mapping
  utils/file_validation.py  signature sniffing, filename sanitising, capped reads
  dependencies/providers.py FastAPI dependency providers
tests/                      pytest suite + test-document generators
samples/, docs/             demo files and a real sample response
```

---

## 4. Configuration

All settings are environment variables (or `.env`); see [`.env.example`](.env.example) for the full list with comments. The most important:

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | 20 | Max file size |
| `MAX_PDF_PAGES` | 50 | Max pages per PDF |
| `MAX_IMAGE_PIXELS` / `MAX_IMAGE_DIMENSION` | 50,000,000 / 12,000 | Image limits (checked from the header before decoding) |
| `OCR_LANGUAGE` | `eng` | Tesseract language(s), e.g. `eng+hin` (packs must be installed) |
| `TESSERACT_CMD` | (PATH) | Tesseract executable, needed on Windows |
| `RENDER_DPI` | 300 | Resolution for rendering scanned PDF pages |
| `OCR_MODE` | `auto` | `auto`, `force` (OCR every PDF page), `never` (native text only) |
| `OCR_TIMEOUT_SECONDS` / `DOCUMENT_TIMEOUT_SECONDS` | 60 / 300 | Per Tesseract call / per document |
| `MAX_CONCURRENT_OCR` | 2 | Documents processed simultaneously |
| `LOG_LEVEL` | `INFO` | Logging level |

Other languages: install the pack (e.g. `tesseract-ocr-hin`) and set `OCR_LANGUAGE=hin` or `eng+hin`. `GET /health` reports whether the configured language is installed.

---

## 5. Tests

```bash
pytest                      # everything (166 tests)
pytest -m "not integration" # fast tests with a fake OCR engine (no Tesseract needed)
pytest -m integration       # real Tesseract on generated documents with known text
```

The suite covers validation (spoofed types, traversal filenames, size limits including streamed bodies), Pillow image modes / corrupt / decompression-bomb inputs, PDFs (native, scanned, mixed, blank, zero-page, corrupt, encrypted, page limit), preprocessing decisions, OCR parsing / confidence / error mapping, orchestration (retry, partial failure, deadlines, concurrency cap) and the HTTP API. Integration tests assert real recognised text from generated PNG, JPEG, native, scanned, multi-page and mixed documents, plus upside-down and unevenly lit scans.

---

## 6. Docker

```bash
docker build -t ocr-api .
docker run --rm -p 8000:8000 ocr-api

# extra OCR languages (3-letter codes), then select with OCR_LANGUAGE
docker build --build-arg TESSERACT_LANGS="deu hin" -t ocr-api .
docker run --rm -p 8000:8000 -e OCR_LANGUAGE=eng+hin -e MAX_CONCURRENT_OCR=4 ocr-api
```

The image is based on `python:3.12-slim-bookworm`, installs Tesseract, runs as a non-root user and has a health check on `/api/v1/health`.

> **Not verified yet:** this project was developed and tested in an environment without a Docker daemon, so the Dockerfile has **not been built or run**. Please run the commands above once and check `curl localhost:8000/api/v1/health`.

---

## 7. Security notes

- Uploads are treated as untrusted: the **bytes** decide the format; extension and MIME type must agree with them.
- Limits on body size (rejected from `Content-Length` and while streaming), PDF pages, image pixels/dimensions and rendered page size; decompression bombs are rejected from the image header.
- Password-protected PDFs are rejected; PDFs are opened from memory, never from a user-supplied path; nothing is executed.
- Uploads are processed in memory and never written to disk or kept.
- Logs contain only metadata (type, page counts, duration, error codes), never text, content or filenames.
- Error responses use fixed safe messages; tracebacks and paths stay in the server log.
- Tesseract language/config come from trusted environment settings only, and the language format is validated.

---

## 8. Known limitations

- **Accuracy depends on the scan.** OCR confidence is a heuristic. Handwriting is not supported (printed text only). Very low resolution, heavy skew over 10°, multi-column pages, and complex borderless tables may be read incorrectly. OCR table reconstruction is heuristic and should be reviewed.
- **Preprocessing thresholds are heuristics** tuned on synthetic degradations of one generated page. Tune them (`PreprocessingService` constants, `LOW_CONFIDENCE_THRESHOLD`) against real documents before relying on them.
- A PDF page with a small amount of text *and* a scanned body (e.g. a typed header over a scan) is treated as native when it has at least `NATIVE_TEXT_MIN_CHARS` characters, so the scanned body is not OCR'd. Use `OCR_MODE=force` for such documents.
- Processing is synchronous per request (the response is returned when OCR finishes). Very large documents need a generous client timeout.
- Concurrency limits are per process. With several Uvicorn workers or replicas, the effective limit is `MAX_CONCURRENT_OCR` times the number of processes.
- DOCX extraction reads document paragraphs, headers/footers, and tables; embedded images, comments, tracked changes, and legacy `.doc` files are not OCR-processed. TIFF currently processes its first frame only.
- Structured fields are rule-based and best-effort, not a trained document AI model. Raw text remains available when a field is not identified.
- No authentication or rate limiting; put the service behind a gateway or add them before exposing it publicly.
- Windows requires `TESSERACT_CMD` unless Tesseract is on `PATH`.

## 9. Future improvements

- Async job API (submit, poll, webhook) backed by a queue for large documents, behind the existing `DocumentService.extract` interface
- Batch endpoint with per-file results and total limits
- Hybrid native + OCR merging for pages with embedded images
- More accurate table structure recognition for scanned tables
- hOCR or searchable-PDF output
- Authentication, rate limiting and metrics (Prometheus)
- Tune preprocessing on a labelled real-document corpus; optional alternative engines (e.g. PaddleOCR) for difficult scans


---

## 10. Streamlit frontend

A Streamlit review UI is included in `streamlit_app.py`. It calls the existing FastAPI
endpoint; it does not duplicate or modify the OCR pipeline.

### Run the frontend

Keep the FastAPI backend running in one terminal:

```bash
uvicorn app.main:app --reload
```

In a second terminal, from the project root:

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually `http://localhost:8501`).

By default, the frontend connects to `http://127.0.0.1:8000`. To use another backend URL:

```bash
export OCR_API_URL="http://127.0.0.1:8000"
```

On Windows PowerShell:

```powershell
$env:OCR_API_URL="http://127.0.0.1:8000"
```

The frontend automatically submits a newly uploaded file to `POST /api/v1/ocr/extract`,
then renders editable fields from the returned `structured_data`. It also displays
page-level extraction metadata, detected tables, raw OCR text, and JSON exports.
The extracted values are heuristic and must be reviewed by the user.
