"""
src/ingestion/table_extractor.py
=================================
Table extraction from PDFs using pdfplumber.

Why pdfplumber for tables?
- It uses a different algorithm than PyMuPDF — better at detecting cell boundaries
- Handles both bordered AND borderless tables (common in BSE annual reports)
- Returns tables as Python lists of lists → easy to convert to Markdown

What this module does:
- For each page in a PDF, detect and extract all tables
- Convert each table to Markdown format (pipe syntax)
- Clean up merged cells, None values, and whitespace
- Return TableChunk objects with page metadata

Markdown tables are the best format to pass to LLMs because:
- LLMs natively understand | column | syntax
- Preserves row/column structure without confusion
- Much better than CSV (no header ambiguity) or raw text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

from src.utils.logger import logger


# ------------------------------------------------------------------ #
# Data Classes
# ------------------------------------------------------------------ #

@dataclass
class ExtractedTable:
    """A single extracted table from a PDF page."""
    markdown: str                # Table in Markdown pipe format
    page_number: int
    table_index: int             # Table number on this page (0-indexed)
    row_count: int
    col_count: int
    bbox: Optional[tuple] = None # Bounding box on the page
    has_header: bool = True


@dataclass
class PageTables:
    """All tables extracted from one page."""
    page_number: int
    tables: list[ExtractedTable] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Table Extractor
# ------------------------------------------------------------------ #

class TableExtractor:
    """
    Extracts tables from PDF files using pdfplumber and converts to Markdown.

    Table extraction settings are tuned for BSE annual report layouts:
    - snap_tolerance: handles slightly misaligned cell borders (common in BSE PDFs)
    - join_tolerance: merges text fragments within the same cell
    - edge_min_length: minimum line length to be considered a table border
    """

    # pdfplumber table settings tuned for financial reports
    TABLE_SETTINGS = {
        "vertical_strategy": "lines_strict",    # use actual drawn lines (not text-based)
        "horizontal_strategy": "lines_strict",
        "snap_tolerance": 5,                    # pixels — snap nearby lines together
        "join_tolerance": 5,                    # pixels — join nearby text in same cell
        "edge_min_length": 25,                  # ignore tiny lines (< 25px)
        "min_words_vertical": 2,                # at least 2 words to form vertical boundary
        "min_words_horizontal": 1,
        "text_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
        "intersection_tolerance": 3,
        "intersection_x_tolerance": 3,
        "intersection_y_tolerance": 3,
    }

    # Fallback settings if strict line detection finds no tables
    TABLE_SETTINGS_FALLBACK = {
        "vertical_strategy": "text",            # fall back to text-based column detection
        "horizontal_strategy": "text",
        "snap_tolerance": 8,
        "join_tolerance": 8,
        "edge_min_length": 10,
    }

    def extract_all_tables(
        self,
        pdf_path: Path,
        pages_to_check: Optional[list[int]] = None,
    ) -> dict[int, PageTables]:
        """
        Extract all tables from a PDF file.

        Args:
            pdf_path: Path to the PDF
            pages_to_check: Optional list of 1-indexed page numbers to process.
                            If None, processes all pages.

        Returns:
            Dict mapping page_number → PageTables
        """
        logger.info(f"Extracting tables from: {pdf_path.name}")
        results: dict[int, PageTables] = {}
        total_tables = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages
            if pages_to_check:
                # Convert 1-indexed to 0-indexed
                pages = [pdf.pages[i - 1] for i in pages_to_check if 0 < i <= len(pdf.pages)]

            for page in pages:
                page_num = page.page_number   # pdfplumber is 1-indexed
                page_tables = self._extract_page_tables(page, page_num)

                if page_tables.tables:
                    results[page_num] = page_tables
                    total_tables += len(page_tables.tables)

        logger.info(f"✓ Extracted {total_tables} tables from {pdf_path.name}")
        return results

    def _extract_page_tables(self, page: pdfplumber.page.Page, page_num: int) -> PageTables:
        """Extract all tables from a single page."""
        page_tables = PageTables(page_number=page_num)

        # Try strict line-based detection first
        tables = page.extract_tables(self.TABLE_SETTINGS)

        # If nothing found, try text-based fallback
        if not tables:
            tables = page.extract_tables(self.TABLE_SETTINGS_FALLBACK)

        if not tables:
            return page_tables

        # Get bounding boxes for tables on this page
        table_bboxes = []
        try:
            finder = page.find_tables(self.TABLE_SETTINGS)
            table_bboxes = [t.bbox for t in finder]
        except Exception:
            pass

        for idx, raw_table in enumerate(tables):
            if not raw_table or len(raw_table) < 2:
                continue   # Skip single-row "tables" (not real tables)

            cleaned = self._clean_table(raw_table)
            if not cleaned:
                continue

            markdown = self._table_to_markdown(cleaned)
            if not markdown.strip():
                continue

            bbox = table_bboxes[idx] if idx < len(table_bboxes) else None

            table = ExtractedTable(
                markdown=markdown,
                page_number=page_num,
                table_index=idx,
                row_count=len(cleaned),
                col_count=max(len(row) for row in cleaned),
                bbox=bbox,
                has_header=True,
            )
            page_tables.tables.append(table)

        return page_tables

    # ---------------------------------------------------------------- #
    # Table Cleaning
    # ---------------------------------------------------------------- #

    def _clean_table(self, raw_table: list[list]) -> list[list[str]]:
        """
        Clean raw table data from pdfplumber.

        Handles:
        - None values (merged cells) → forward-fill or empty string
        - Whitespace normalization
        - Empty row removal
        - Duplicate header rows
        """
        if not raw_table:
            return []

        cleaned = []
        for row in raw_table:
            # Replace None with empty string
            clean_row = []
            for cell in row:
                if cell is None:
                    clean_row.append("")
                else:
                    # Normalize whitespace, remove newlines within cells
                    cell_str = str(cell).strip()
                    cell_str = re.sub(r'\s+', ' ', cell_str)
                    cell_str = cell_str.replace('\n', ' ').replace('\r', '')
                    clean_row.append(cell_str)

            # Skip completely empty rows
            if all(c == "" for c in clean_row):
                continue

            cleaned.append(clean_row)

        # Normalize column count (handle ragged tables)
        if cleaned:
            max_cols = max(len(row) for row in cleaned)
            cleaned = [row + [""] * (max_cols - len(row)) for row in cleaned]

        return cleaned

    # ---------------------------------------------------------------- #
    # Markdown Conversion
    # ---------------------------------------------------------------- #

    def _table_to_markdown(self, table: list[list[str]]) -> str:
        """
        Convert a cleaned table (list of lists) to Markdown pipe format.

        Example output:
        | Revenue | FY2024 | FY2023 |
        |---|---|---|
        | Net Revenue | ₹240,893 Cr | ₹225,458 Cr |
        | Other Income | ₹3,212 Cr | ₹2,890 Cr |
        """
        if not table:
            return ""

        lines = []

        # Header row
        header = table[0]
        lines.append("| " + " | ".join(self._escape_cell(c) for c in header) + " |")

        # Separator row (standard Markdown)
        lines.append("|" + "|".join("---" for _ in header) + "|")

        # Data rows
        for row in table[1:]:
            lines.append("| " + " | ".join(self._escape_cell(c) for c in row) + " |")

        return "\n".join(lines)

    def _escape_cell(self, text: str) -> str:
        """Escape pipe characters inside cells so they don't break Markdown table."""
        return text.replace("|", "\\|").strip()

    # ---------------------------------------------------------------- #
    # Utility: Get table bounding boxes (for PDF parser coordination)
    # ---------------------------------------------------------------- #

    def get_table_bboxes(self, pdf_path: Path, page_num: int) -> list[tuple]:
        """
        Return bounding boxes of all tables on a given page.
        Used by the PDF parser to avoid double-extracting table text as paragraphs.
        """
        bboxes = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num > len(pdf.pages):
                return []
            page = pdf.pages[page_num - 1]
            try:
                tables = page.find_tables(self.TABLE_SETTINGS)
                bboxes = [t.bbox for t in tables]
            except Exception:
                pass
        return bboxes
