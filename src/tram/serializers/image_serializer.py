"""
Image Serializer for TRAM
Handles PNG, JPG, JPEG, TIFF screenshots of IR reports, threat intel tables,
or any image containing text relevant to ATT&CK technique extraction.
"""

import logging
import re
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract/Pillow not installed — image OCR unavailable")


# ── Constants ─────────────────────────────────────────────────────────────────
MIN_LINE_LENGTH = 15
TESSERACT_CONFIG = "--psm 3"   # fully automatic page segmentation
MAX_IMAGE_DIM = 4000           # px — resize if larger (prevents tesseract timeout)
OCR_SCALE_FACTOR = 2.0         # upscale small images for better OCR accuracy


class ImageSerializer:
    """
    Extracts text from image files using tesseract OCR.

    Pre-processing pipeline applied before OCR:
      1. Convert to RGB (handles RGBA, palette, etc.)
      2. Upscale if image is small (< 1000px wide) — improves accuracy
      3. Downscale if image is very large (> MAX_IMAGE_DIM) — prevents timeout
      4. Enhance contrast slightly
      5. Convert to grayscale
      6. Run tesseract with PSM 3

    Post-processing:
      - Remove noise lines (< MIN_LINE_LENGTH chars)
      - Remove lines that are purely numbers/punctuation (table borders, page nums)
      - Deduplicate consecutive identical lines (tesseract artifact)
      - Collapse multiple blank lines
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    SUPPORTED_MIMETYPES = {
        "image/png", "image/jpeg", "image/tiff", "image/jpg"
    }

    def __init__(self, file_bytes: bytes, filename: str = "image"):
        self.file_bytes = file_bytes
        self.filename = filename

    def serialize(self) -> str:
        """
        Main entry point. Returns OCR-extracted text ready for TRAM pipeline.
        Raises ImportError if tesseract/Pillow not available.
        Raises ValueError if OCR produces no usable text.
        """
        if not OCR_AVAILABLE:
            raise ImportError(
                "pytesseract and Pillow are required for image OCR. "
                "Install with: pip install pytesseract Pillow\n"
                "Also ensure tesseract-ocr is installed: apt install tesseract-ocr"
            )

        try:
            image = Image.open(BytesIO(self.file_bytes))
        except Exception as e:
            raise ValueError(
                f"Cannot open image '{self.filename}': {e}"
            ) from e

        processed = self._preprocess(image)
        raw_text = self._run_ocr(processed)

        if not raw_text or not raw_text.strip():
            raise ValueError(
                f"OCR produced no text from '{self.filename}'. "
                "Ensure the image contains readable text."
            )

        cleaned = self._clean_text(raw_text)

        if not cleaned:
            raise ValueError(
                f"All extracted text from '{self.filename}' was filtered as noise. "
                "The image may be too low resolution or contain only graphical elements."
            )

        logger.info(
            f"Image OCR: '{self.filename}' → {len(cleaned)} chars extracted"
        )
        return cleaned

    def _preprocess(self, image: "Image.Image") -> "Image.Image":
        """Apply pre-processing pipeline to improve OCR accuracy."""
        # Convert to RGB first (handles RGBA transparency, palette mode, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        width, height = image.size

        # Upscale small images — tesseract works best at 300+ DPI equivalent
        if width < 1000:
            scale = OCR_SCALE_FACTOR
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug(f"Upscaled image to {new_size}")

        # Downscale very large images to prevent tesseract memory/timeout issues
        width, height = image.size
        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
            logger.debug(f"Downscaled large image to {image.size}")

        # Enhance contrast slightly — helps with low-contrast screenshots
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)

        # Convert to grayscale — tesseract performs better on grayscale
        image = image.convert("L")

        return image

    def _run_ocr(self, image: "Image.Image") -> str:
        """Run tesseract OCR on the pre-processed image."""
        try:
            text = pytesseract.image_to_string(
                image,
                config=TESSERACT_CONFIG,
                lang="eng"
            )
            return text
        except Exception as e:
            raise ValueError(f"Tesseract OCR failed: {e}") from e

    def _clean_text(self, text: str) -> str:
        """
        Post-process OCR output to remove noise and improve NLP quality.
        """
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []
        prev_line = None

        for line in lines:
            stripped = line.strip()

            # Skip too-short lines (noise, stray characters)
            if len(stripped) < MIN_LINE_LENGTH:
                continue

            # Skip lines that are purely numeric/punctuation (table borders, rulers)
            if re.match(r"^[\d\s\-–—|_.,:;]+$", stripped):
                continue

            # Skip consecutive duplicate lines (tesseract artifact on borders)
            if stripped == prev_line:
                continue

            cleaned_lines.append(stripped)
            prev_line = stripped

        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
