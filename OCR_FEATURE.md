# TRAM — PDF & Image OCR Ingestion

This branch extends TRAM's report ingestion with two new input types:

- **PDF files** — text-based PDFs are parsed natively, scanned PDFs fall back to
  OCR, embedded images inside pages are OCR'd individually, and tables are
  converted into natural-language sentences.
- **Image files** — `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif` screenshots of IR
  reports, threat-intel tables, or advisory pages.

The downstream ML pipeline (`pipeline.py`, sentence tokenization, technique
mapping) is **untouched** — the new serializers produce the same plain-text
report body that the existing flow already expects.

---

## What Changed

### New files
| Path | Purpose |
| --- | --- |
| `src/tram/serializers/pdf_serializer.py` | pdfplumber text extraction, table → sentences, page-level OCR fallback, embedded-image OCR |
| `src/tram/serializers/image_serializer.py` | Full-image tesseract OCR with preprocessing (RGB convert, smart resize, contrast boost, grayscale) |
| `tests/tram/test_serializers.py` | Unit tests for table conversion, noise filtering, dedup, error handling |
| `OCR_FEATURE.md` | This document |

### Modified files
| Path | Change |
| --- | --- |
| `Dockerfile` | Adds `tesseract-ocr`, `tesseract-ocr-eng`, `poppler-utils` to the apt install block |
| `requirements/requirements.txt` | Bumps `pdfplumber` to `>=0.10.0`; adds `pdf2image`, `pytesseract`, `Pillow` |
| `src/tram/templates/base.html` | Extends the Upload Report button to accept `.png/.jpg/.jpeg/.tiff` |
| `src/tram/views.py` | `upload()` view dispatches PDFs → `PDFSerializer`, images → `ImageSerializer`, then submits the extracted text as a `.txt` job |
| `src/tram/serializers.py` → `src/tram/serializers/` | Old single-file module converted to a package; all original DRF serializer classes preserved verbatim in `__init__.py` and re-exported alongside the new OCR serializers |

---

## Deploy

### 1. Build the Docker image
```bash
docker compose -f docker/docker-compose.yml build tram
```

### 2. Verify OCR tooling inside the container
```bash
docker compose -f docker/docker-compose.yml run --rm tram tesseract --version
docker compose -f docker/docker-compose.yml run --rm tram pdftoppm -v
```
Both should print version banners.

### 3. Bring up the stack
```bash
docker compose -f docker/docker-compose.yml up
```

### 4. Smoke test
```bash
# Text-based PDF
curl -X POST http://localhost:8000/upload/ \
  -H "X-CSRFToken: ..." -b cookies.txt \
  -F "file=@/path/to/report.pdf"

# Scanned PDF
curl -X POST http://localhost:8000/upload/ \
  -F "file=@/path/to/scanned-report.pdf"

# Image
curl -X POST http://localhost:8000/upload/ \
  -F "file=@/path/to/screenshot.png"
```

Or use the **Upload Report** button in the navbar — the file picker now accepts
the new extensions directly.

### 5. Run the unit tests
```bash
docker compose -f docker/docker-compose.yml run --rm tram \
  python manage.py test tram.tests.test_serializers
```

---

## Extraction Strategy

### PDF (per page)
1. `pdfplumber.Page.extract_text()` — if the result is at least
   `MIN_TEXT_LENGTH` (100 chars), keep it.
2. Otherwise rasterize that single page with `pdf2image` at 300 DPI and run
   tesseract on the image.
3. `Page.extract_tables()` — every table is converted to
   `Header: value. Header: value.` sentences so the NLP pipeline sees
   cell relationships instead of column-collapsed text.
4. `Page.images` — for every embedded image, crop the page to the image's
   bbox, rasterize the crop with `to_image()`, and OCR it. This catches text
   in diagrams, figures, and inline screenshots that plain text extraction
   misses.

### Image
1. Convert to RGB (handles RGBA / palette modes).
2. Upscale small images (< 1000px wide) by 2× — tesseract is most accurate at
   ≥ 300 DPI equivalent.
3. Downscale very large images (> 4000px on any axis) to keep tesseract from
   timing out or running out of memory.
4. Bump contrast by 1.3×, then convert to grayscale.
5. `pytesseract.image_to_string(..., config="--psm 3", lang="eng")`.
6. Filter noise lines: drop anything shorter than `MIN_LINE_LENGTH` (15 chars),
   purely numeric/punctuation lines (table borders, page numbers), and
   consecutive duplicate lines (a tesseract artifact).

---

## Known Limitations

| Limitation | Workaround |
| --- | --- |
| English-only OCR | Add e.g. `tesseract-ocr-ara` to the Dockerfile and `lang="ara+eng"` in the OCR call |
| Password-protected PDFs | Decrypt before upload — out of scope |
| Multi-column layouts | pdfplumber may merge columns; layout-aware post-processing not implemented |
| Very large PDFs (100+ pages) | Synchronous request may time out — consider routing through Celery |
| Handwritten notes | Tesseract accuracy drops sharply — out of scope |
| Image-table reconstruction | Current image OCR is whole-page; `image_to_data()` with bbox grouping would preserve cell layout |
