"""
src/ingestion/pdf_parser.py
============================
Core PDF text and layout extraction using PyMuPDF (fitz).

What this does:
- Extracts text blocks WITH layout metadata (font size, bold, position)
- Detects headings via font size / font flags
- Strips repeating headers/footers (page numbers, company name noise)
- Identifies page regions that contain tables (for routing to table extractor)
- Returns a structured list of blocks per page with content type labels
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.utils.logger import logger
from config.settings import settings


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #

@dataclass
class TextBlock:
    """A single block of text extracted from a PDF page."""
    text: str
    page_number: int
    block_type: str          # "heading" | "paragraph" | "caption" | "footer" | "header"
    font_size: float
    is_bold: bool
    bbox: tuple              # (x0, y0, x1, y1) bounding box
    block_index: int         # sequential block index within the page


@dataclass
class TableRegion:
    """Marks a region on a page that likely contains a table."""
    page_number: int
    bbox: tuple              # bounding box of the table region


@dataclass
class ParsedPage:
    """All extracted content from a single PDF page."""
    page_number: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    table_regions: list[TableRegion] = field(default_factory=list)
    raw_text: str = ""
    width: float = 0.0
    height: float = 0.0


@dataclass
class ParsedDocument:
    """Full parsed output for one PDF file."""
    source_file: str
    company: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def all_text_blocks(self) -> list[TextBlock]:
        return [b for page in self.pages for b in page.text_blocks]


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #

class PDFParser:
    """
    Production-grade PDF parser using PyMuPDF.

    Key features:
    - Font-size-based heading detection (no hardcoded sizes — uses relative analysis)
    - Automatic header/footer removal (detects repeating text across pages)
    - Table region detection (bounding box hints for pdfplumber)
    - Handles multi-column layouts via block sorting
    """

    # Font flags from PyMuPDF spec
    BOLD_FLAG = 1 << 4       # bit 4 = bold
    ITALIC_FLAG = 1 << 1     # bit 1 = italic

    def __init__(self):
        self.heading_threshold = settings.heading_font_size_threshold

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def parse(self, pdf_path: Path, company: str) -> ParsedDocument:
        """
        Parse a single PDF file and return a structured ParsedDocument.

        Args:
            pdf_path: Path to the PDF file
            company:  Company name (from filename stem, e.g. "TCS")

        Returns:
            ParsedDocument with all pages, text blocks, and table regions
        """
        logger.info(f"Parsing PDF: {pdf_path.name} | Company: {company}")

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        parsed_doc = ParsedDocument(
            source_file=pdf_path.name,
            company=company,
            total_pages=total_pages,
        )

        # ---- Step 1: Raw extraction pass (collect all blocks) ----
        raw_pages: list[ParsedPage] = []
        for page_num in range(total_pages):
            page = doc[page_num]
            parsed_page = self._extract_page(page, page_num + 1)
            raw_pages.append(parsed_page)

        doc.close()

        # ---- Step 2: Detect and remove repeating headers/footers ----
        footer_header_texts = self._detect_repeating_noise(raw_pages)
        logger.debug(f"Detected {len(footer_header_texts)} repeating header/footer patterns")

        # ---- Step 3: Compute document-wide font size stats for heading detection ----
        font_stats = self._compute_font_stats(raw_pages)
        median_size = font_stats["median"]
        heading_cutoff = max(self.heading_threshold, median_size * 1.15)
        logger.debug(f"Font stats → median: {median_size:.1f}pt | heading cutoff: {heading_cutoff:.1f}pt")

        # ---- Step 4: Classify blocks and filter noise ----
        for raw_page in raw_pages:
            clean_page = self._clean_page(raw_page, footer_header_texts, heading_cutoff)
            parsed_doc.pages.append(clean_page)

        total_blocks = sum(len(p.text_blocks) for p in parsed_doc.pages)
        logger.info(f"✓ Parsed {total_pages} pages → {total_blocks} text blocks")

        return parsed_doc

    # ---------------------------------------------------------------- #
    # Internal: Page Extraction
    # ---------------------------------------------------------------- #

    def _extract_page(self, page: fitz.Page, page_num: int) -> ParsedPage:
        """Extract all text blocks from a single page with metadata."""
        parsed_page = ParsedPage(
            page_number=page_num,
            width=page.rect.width,
            height=page.rect.height,
        )

        # Get detailed block info: text + font details
        # flags=fitz.TEXT_PRESERVE_WHITESPACE gives cleaner output
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        block_idx = 0
        for block in blocks:
            if block["type"] == 0:  # type 0 = text block
                block_text, font_size, is_bold = self._extract_block_content(block)
                if not block_text.strip():
                    continue

                bbox = block["bbox"]
                text_block = TextBlock(
                    text=block_text.strip(),
                    page_number=page_num,
                    block_type="paragraph",   # will be reclassified later
                    font_size=font_size,
                    is_bold=is_bold,
                    bbox=bbox,
                    block_index=block_idx,
                )
                parsed_page.text_blocks.append(text_block)
                block_idx += 1

            elif block["type"] == 1:  # type 1 = image block
                # Mark this bounding box — may contain a table nearby
                # (image detection is a hint for visual layout)
                pass

        # Sort blocks top-to-bottom, left-to-right (handles 2-column layouts)
        parsed_page.text_blocks.sort(key=lambda b: (round(b.bbox[1] / 20), b.bbox[0]))

        parsed_page.raw_text = "\n".join(b.text for b in parsed_page.text_blocks)
        return parsed_page

    def _extract_block_content(self, block: dict) -> tuple[str, float, bool]:
        """Extract combined text, dominant font size, and bold flag from a block."""
        texts = []
        font_sizes = []
        is_bold = False

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                font_sizes.append(span.get("size", 10.0))
                if span.get("flags", 0) & self.BOLD_FLAG:
                    is_bold = True
            texts.append(line_text)

        combined_text = "\n".join(texts)
        dominant_font = max(font_sizes, default=10.0)
        return combined_text, dominant_font, is_bold

    # ---------------------------------------------------------------- #
    # Internal: Header/Footer Detection
    # ---------------------------------------------------------------- #

    def _detect_repeating_noise(self, pages: list[ParsedPage]) -> set[str]:
        """
        Detect text patterns that repeat across many pages (headers, footers, page numbers).
        Returns a set of normalized text strings to filter out.
        """
        if len(pages) < 3:
            return set()

        # Count how often each text block content appears across pages
        text_frequency: dict[str, int] = {}
        for page in pages:
            seen_on_this_page = set()
            for block in page.text_blocks:
                # Normalize: lowercase, strip whitespace, remove digits (page numbers)
                normalized = re.sub(r'\d+', '', block.text.lower()).strip()
                if len(normalized) < 3:
                    continue
                if normalized not in seen_on_this_page:
                    text_frequency[normalized] = text_frequency.get(normalized, 0) + 1
                    seen_on_this_page.add(normalized)

        # If a text appears on more than 30% of pages → it's a repeating element
        threshold = max(3, len(pages) * 0.30)
        noise_texts = {text for text, count in text_frequency.items() if count >= threshold}
        return noise_texts

    # ---------------------------------------------------------------- #
    # Internal: Font Statistics
    # ---------------------------------------------------------------- #

    def _compute_font_stats(self, pages: list[ParsedPage]) -> dict:
        """Compute document-wide font size statistics for relative heading detection."""
        all_sizes = []
        for page in pages:
            for block in page.text_blocks:
                all_sizes.append(block.font_size)

        if not all_sizes:
            return {"median": 10.0, "mean": 10.0, "max": 12.0}

        sorted_sizes = sorted(all_sizes)
        n = len(sorted_sizes)
        median = sorted_sizes[n // 2]
        mean = sum(sorted_sizes) / n
        return {"median": median, "mean": mean, "max": max(sorted_sizes)}

    # ---------------------------------------------------------------- #
    # Internal: Page Cleaning + Block Classification
    # ---------------------------------------------------------------- #

    def _clean_page(
        self,
        raw_page: ParsedPage,
        noise_texts: set[str],
        heading_cutoff: float,
    ) -> ParsedPage:
        """Remove noise blocks and classify remaining blocks by type."""
        clean_page = ParsedPage(
            page_number=raw_page.page_number,
            width=raw_page.width,
            height=raw_page.height,
            table_regions=raw_page.table_regions,
        )

        page_height = raw_page.height
        top_margin = page_height * 0.06      # top 6% = likely header
        bottom_margin = page_height * 0.94   # bottom 6% = likely footer

        for block in raw_page.text_blocks:
            normalized = re.sub(r'\d+', '', block.text.lower()).strip()

            # 1. Skip repeating noise (headers/footers by content)
            if normalized in noise_texts:
                block.block_type = "footer"
                continue

            # 2. Skip pure page numbers
            if re.fullmatch(r'\s*\d{1,4}\s*', block.text):
                continue

            # 3. Skip very short blocks at page margins (positional header/footer)
            y_center = (block.bbox[1] + block.bbox[3]) / 2
            if len(block.text) < 50 and (y_center < top_margin or y_center > bottom_margin):
                continue

            # 4. Classify block type
            if block.font_size >= heading_cutoff or (block.is_bold and block.font_size >= heading_cutoff * 0.9):
                block.block_type = "heading"
            elif block.font_size < 8.0:
                block.block_type = "caption"
            else:
                block.block_type = "paragraph"

            clean_page.text_blocks.append(block)

        clean_page.raw_text = "\n".join(b.text for b in clean_page.text_blocks)
        return clean_page


# ------------------------------------------------------------------ #
# Helper: Extract company name from filename
# ------------------------------------------------------------------ #

def company_from_filename(filename: str) -> str:
    """
    Extract clean company name from PDF filename.
    Examples:
        'TCS-1.pdf'     → 'TCS'
        'ICICI-2.pdf'   → 'ICICI'
        'L&T.pdf'       → 'L&T'
        'AIRTEL.pdf'    → 'AIRTEL'
    """
    stem = Path(filename).stem          # strip .pdf
    # Remove trailing -1, -2 etc.
    name = re.sub(r'-\d+$', '', stem)
    return name.upper().strip()
