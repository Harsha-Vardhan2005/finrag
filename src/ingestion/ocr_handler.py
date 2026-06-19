"""
src/ingestion/ocr_handler.py
=============================
OCR fallback for scanned pages using EasyOCR.

When is this used?
- Pages where PyMuPDF extracts little or no text (< 50 chars)
  but the page clearly has content → likely a scanned image page
- Some older BSE filings have scanned sections mixed with digital text

EasyOCR is chosen over pytesseract because:
- No external Tesseract binary install needed (pure Python)
- Better accuracy on mixed-language content (English + numbers + ₹ symbols)
- GPU acceleration support (uses your RTX 3050 if available)

Note: OCR is SLOW. Only triggered as fallback, not for every page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from src.utils.logger import logger
from config.settings import settings

# Lazy import — EasyOCR takes ~3s to initialize, only load when needed
_ocr_reader = None


def get_ocr_reader():
    """Lazy-initialize EasyOCR reader (GPU-aware)."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR reader (first time only, may take ~10s)...")
        import easyocr
        device = settings.get_device()
        use_gpu = device == "cuda"
        _ocr_reader = easyocr.Reader(
            lang_list=["en"],
            gpu=use_gpu,
            verbose=False,
        )
        logger.info(f"EasyOCR ready | GPU: {use_gpu}")
    return _ocr_reader


class OCRHandler:
    """
    Handles OCR for scanned pages in PDFs.
    Used as a fallback when PyMuPDF extracts insufficient text.
    """

    # Threshold: if extracted text is shorter than this, try OCR
    SPARSE_TEXT_THRESHOLD = 50  # characters

    def is_sparse_page(self, extracted_text: str) -> bool:
        """
        Determine if a page has insufficient extracted text and needs OCR.

        Args:
            extracted_text: Text already extracted by PyMuPDF for this page

        Returns:
            True if OCR fallback should be attempted
        """
        # Clean whitespace to get actual content length
        clean = extracted_text.strip().replace("\n", "").replace(" ", "")
        return len(clean) < self.SPARSE_TEXT_THRESHOLD

    def ocr_page(self, pdf_path: Path, page_number: int) -> str:
        """
        Run OCR on a specific page from a PDF.

        Process:
        1. Render the page to a high-res image via PyMuPDF
        2. Pass to EasyOCR
        3. Return extracted text sorted by vertical position

        Args:
            pdf_path: Path to the PDF file
            page_number: 1-indexed page number

        Returns:
            OCR-extracted text for the page (or empty string on failure)
        """
        try:
            import fitz

            doc = fitz.open(str(pdf_path))
            if page_number > len(doc):
                doc.close()
                return ""

            page = doc[page_number - 1]

            # Render at 2x DPI for better OCR accuracy (150 DPI → 300 DPI equivalent)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            doc.close()

            reader = get_ocr_reader()
            results = reader.readtext(img_array, detail=1, paragraph=False)

            # Sort results top-to-bottom (by y-coordinate of bounding box)
            results.sort(key=lambda r: r[0][0][1])  # top-left y coordinate

            # Extract just the text, filter low confidence results
            text_lines = []
            for bbox, text, confidence in results:
                if confidence > 0.3:  # discard very low confidence
                    text_lines.append(text)

            ocr_text = "\n".join(text_lines)
            logger.debug(f"OCR page {page_number}: extracted {len(ocr_text)} chars")
            return ocr_text

        except Exception as e:
            logger.warning(f"OCR failed on page {page_number} of {pdf_path.name}: {e}")
            return ""

    def ocr_pdf_sparse_pages(
        self,
        pdf_path: Path,
        page_texts: dict[int, str],
    ) -> dict[int, str]:
        """
        Run OCR only on pages that have sparse/missing text.

        Args:
            pdf_path: Path to the PDF
            page_texts: Dict mapping page_number → existing extracted text

        Returns:
            Dict of page_number → OCR text (only for pages that needed OCR)
        """
        ocr_results = {}
        sparse_pages = [
            page_num for page_num, text in page_texts.items()
            if self.is_sparse_page(text)
        ]

        if not sparse_pages:
            logger.debug("No sparse pages detected — OCR not needed")
            return {}

        logger.info(f"Running OCR on {len(sparse_pages)} sparse pages: {sparse_pages}")

        for page_num in sparse_pages:
            ocr_text = self.ocr_page(pdf_path, page_num)
            if ocr_text.strip():
                ocr_results[page_num] = ocr_text

        return ocr_results
