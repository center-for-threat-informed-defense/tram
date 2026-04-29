"""
PDF Serializer for TRAM
Handles both text-based PDFs (via pdfplumber) and scanned/image PDFs
(via pdf2image + pytesseract OCR fallback).

Table extraction converts structured rows into natural language sentences
that TRAM's NLP pipeline can process effectively.
"""

import logging
import re
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy imports (fail gracefully if deps not installed) ──────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed — text PDF extraction unavailable")

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pdf2image/pytesseract not installed — OCR fallback unavailable")


# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 100        # chars — below this, page is considered image-only
MIN_LINE_LENGTH = 15         # chars — shorter lines discarded as noise
OCR_DPI = 300                # DPI for pdf2image rasterization
TESSERACT_CONFIG = "--psm 3" # auto page segmentation, no OSD


class PDFSerializer:
    """
    Extracts clean text from PDF files for TRAM ingestion.

    Strategy per page:
      1. Try pdfplumber text extraction
      2. If text < MIN_TEXT_LENGTH, fall back to OCR via pdf2image + tesseract
      3. Extract tables separately and convert rows to NL sentences
      4. Combine: body text + table sentences → single string
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, file_bytes: bytes, filename: str = "report.pdf"):
        self.file_bytes = file_bytes
        self.filename = filename

    def serialize(self) -> str:
        """
        Main entry point. Returns extracted text ready for TRAM pipeline.
        Raises ValueError if extraction fails completely.
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF parsing. "
                "Install with: pip install pdfplumber"
            )

        all_text_parts = []

        try:
            with pdfplumber.open(BytesIO(self.file_bytes)) as pdf:
                logger.info(
                    f"Processing PDF: {self.filename} "
                    f"({len(pdf.pages)} pages)"
                )

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = self._extract_page_text(page, page_num)
                    if page_text:
                        all_text_parts.append(page_text)

        except Exception as e:
            raise ValueError(
                f"Failed to open PDF '{self.filename}': {e}"
            ) from e

        if not all_text_parts:
            raise ValueError(
                f"No extractable text found in '{self.filename}'. "
                "Ensure the file is a valid PDF."
            )

        combined = "\n\n".join(all_text_parts)
        return self._clean_text(combined)

    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text and tables from a single pdfplumber page object."""
        parts = []

        # ── Body text ─────────────────────────────────────────────────────────
        raw_text = page.extract_text() or ""

        if len(raw_text.strip()) >= MIN_TEXT_LENGTH:
            parts.append(raw_text.strip())
            logger.debug(f"Page {page_num}: text extraction OK ({len(raw_text)} chars)")
        else:
            # Insufficient text — try OCR on this page
            logger.debug(
                f"Page {page_num}: insufficient text ({len(raw_text.strip())} chars), "
                "attempting OCR"
            )
            ocr_text = self._ocr_page(page_num)
            if ocr_text:
                parts.append(ocr_text)

        # ── Tables ────────────────────────────────────────────────────────────
        try:
            tables = page.extract_tables()
            for table in tables:
                table_text = self._table_to_sentences(table)
                if table_text:
                    parts.append(table_text)
        except Exception as e:
            logger.debug(f"Page {page_num}: table extraction failed — {e}")

        # ── Embedded images ───────────────────────────────────────────────────
        embedded_text = self._ocr_embedded_images(page, page_num)
        if embedded_text:
            parts.append(embedded_text)

        return "\n\n".join(parts)

    def _ocr_embedded_images(self, page, page_num: int) -> str:
        """
        Run tesseract on each image embedded in the page.

        pdfplumber exposes embedded images via `page.images` (list of dicts with
        x0/top/x1/bottom bounding boxes). For each image we crop the page to the
        image's bbox, rasterize that crop via `to_image()`, and OCR the result.
        This picks up text in diagrams, figures, and screenshots that
        `page.extract_text()` misses.
        """
        if not OCR_AVAILABLE:
            return ""

        try:
            images = page.images or []
        except Exception as e:
            logger.debug(f"Page {page_num}: page.images access failed — {e}")
            return ""

        if not images:
            return ""

        collected = []
        page_bbox = (0, 0, page.width, page.height)

        for idx, img in enumerate(images, start=1):
            try:
                x0 = max(float(img.get("x0", 0)), page_bbox[0])
                top = max(float(img.get("top", 0)), page_bbox[1])
                x1 = min(float(img.get("x1", page.width)), page_bbox[2])
                bottom = min(float(img.get("bottom", page.height)), page_bbox[3])

                if x1 <= x0 or bottom <= top:
                    continue

                cropped = page.crop((x0, top, x1, bottom))
                pil_image = cropped.to_image(resolution=OCR_DPI).original

                raw = pytesseract.image_to_string(
                    pil_image,
                    config=TESSERACT_CONFIG,
                    lang="eng",
                )
                cleaned = self._clean_text(raw)
                if cleaned:
                    collected.append(cleaned)
                    logger.debug(
                        f"Page {page_num} image {idx}: OCR extracted "
                        f"{len(cleaned)} chars"
                    )
            except Exception as e:
                logger.debug(
                    f"Page {page_num} image {idx}: embedded OCR failed — {e}"
                )
                continue

        return "\n\n".join(collected)

    def _ocr_page(self, page_num: int) -> Optional[str]:
        """Rasterize a specific page and run tesseract OCR on it."""
        if not OCR_AVAILABLE:
            logger.warning("OCR unavailable — skipping image-only page")
            return None

        try:
            images = convert_from_bytes(
                self.file_bytes,
                dpi=OCR_DPI,
                first_page=page_num,
                last_page=page_num,
            )
            if not images:
                return None

            text = pytesseract.image_to_string(
                images[0],
                config=TESSERACT_CONFIG
            )
            cleaned = self._clean_text(text)
            logger.debug(
                f"Page {page_num}: OCR extracted {len(cleaned)} chars"
            )
            return cleaned if cleaned else None

        except Exception as e:
            logger.warning(f"OCR failed on page {page_num}: {e}")
            return None

    def _table_to_sentences(self, table: list) -> str:
        """
        Convert a pdfplumber table (list of rows, each row a list of cells)
        into natural language sentences suitable for NLP processing.

        Approach:
          - First row = headers (if it looks like a header row)
          - Subsequent rows = data
          - Output: "Header1: value1. Header2: value2. Header3: value3."
          - Preserves cell relationships that flat text extraction loses

        Example input table:
          [["Technique", "ID", "Description"],
           ["Spearphishing", "T1566.001", "Attacker sent emails..."],
           ["PowerShell", "T1059.001", "Executed encoded command..."]]

        Example output:
          "Technique: Spearphishing. ID: T1566.001. Description: Attacker sent emails...
           Technique: PowerShell. ID: T1059.001. Description: Executed encoded command..."
        """
        if not table or len(table) < 2:
            return ""

        # Detect header row: all cells are non-empty short strings (< 50 chars)
        potential_headers = table[0]
        is_header_row = all(
            cell and isinstance(cell, str) and len(cell.strip()) < 50
            for cell in potential_headers
            if cell is not None
        )

        sentences = []

        if is_header_row:
            headers = [
                str(h).strip() if h else f"Column{i}"
                for i, h in enumerate(potential_headers)
            ]
            data_rows = table[1:]
        else:
            headers = [f"Column{i}" for i in range(len(table[0]))]
            data_rows = table

        for row in data_rows:
            if not any(cell for cell in row if cell):
                continue  # skip empty rows

            cell_parts = []
            for header, cell in zip(headers, row):
                if cell and str(cell).strip():
                    cell_parts.append(f"{header}: {str(cell).strip()}")

            if cell_parts:
                sentences.append(". ".join(cell_parts) + ".")

        return "\n".join(sentences)

    def _clean_text(self, text: str) -> str:
        """
        Post-process extracted/OCR text:
          - Remove lines shorter than MIN_LINE_LENGTH (noise, page numbers, etc.)
          - Collapse multiple blank lines
          - Strip leading/trailing whitespace
          - Remove common OCR artifacts
        """
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip noise: too short, page numbers, pure punctuation
            if len(stripped) < MIN_LINE_LENGTH:
                continue
            if re.match(r"^[\d\s\-–—|]+$", stripped):
                continue
            cleaned_lines.append(stripped)

        # Collapse multiple blank lines into one
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()
