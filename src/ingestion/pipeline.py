"""
src/ingestion/pipeline.py
==========================
Master ingestion pipeline — orchestrates the full PDF → structured document flow.

This is the entry point for processing all 23 BSE financial PDFs.
It coordinates:
    1. PDF parsing (PyMuPDF) → text blocks with layout metadata
    2. Table extraction (pdfplumber) → Markdown tables
    3. OCR fallback (EasyOCR) → handles scanned pages
    4. Fiscal year detection → auto-tags FY2024 / FY2025
    5. Output → list of DocumentRecord objects (ready for chunking)

Output is also cached to disk as JSON so re-running doesn't re-parse all PDFs.

Usage:
    pipeline = IngestionPipeline()
    documents = pipeline.run_all()        # process all PDFs in raw_pdfs/
    documents = pipeline.run(pdf_path)    # process a single PDF
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config.settings import settings
from src.utils.logger import logger
from src.ingestion.pdf_parser import PDFParser, ParsedDocument, company_from_filename
from src.ingestion.table_extractor import TableExtractor, ExtractedTable
from src.ingestion.ocr_handler import OCRHandler


# ------------------------------------------------------------------ #
# Output Data Structure
# ------------------------------------------------------------------ #

@dataclass
class ContentItem:
    """
    A single piece of content (text block or table) from a document.
    This is the raw unit before chunking.
    """
    content: str                   # The actual text/markdown content
    content_type: str              # "text" | "table" | "ocr_text"
    page_number: int
    section: str                   # Detected section heading (best guess)
    subsection: str                # Detected subsection heading (best guess)
    company: str
    source_file: str
    fiscal_year: str
    is_table: bool = False


@dataclass
class DocumentRecord:
    """
    Fully processed document — output of the ingestion pipeline.
    One DocumentRecord per PDF file.
    Each record contains an ordered list of ContentItems.
    """
    company: str
    ticker: str
    source_file: str
    fiscal_year: str
    total_pages: int
    items: list[ContentItem] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(item.content for item in self.items)


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

class IngestionPipeline:
    """
    Orchestrates the full ingestion pipeline for BSE financial PDFs.
    """

    def __init__(self):
        self.parser = PDFParser()
        self.table_extractor = TableExtractor()
        self.ocr_handler = OCRHandler()
        self.cache_dir = settings.processed_docs_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("IngestionPipeline initialized")

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def run_all(self, force_reprocess: bool = False) -> list[DocumentRecord]:
        """
        Process all PDFs in the raw_pdfs directory.

        Args:
            force_reprocess: If True, ignores cache and re-processes everything

        Returns:
            List of DocumentRecord objects (one per PDF file group / company)
        """
        pdf_dir = settings.raw_pdfs_path
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Group multi-part PDFs (TCS-1, TCS-2 → same company, process separately)
        all_records: list[DocumentRecord] = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            try:
                record = self.run(pdf_path, force_reprocess=force_reprocess)
                all_records.append(record)
                logger.info(f"✓ {pdf_path.name} → {len(record.items)} content items")
            except Exception as e:
                logger.error(f"✗ Failed to process {pdf_path.name}: {e}")
                continue

        logger.info(f"Ingestion complete: {len(all_records)} documents processed")
        return all_records

    def run(self, pdf_path: Path, force_reprocess: bool = False) -> DocumentRecord:
        """
        Process a single PDF file through the full ingestion pipeline.

        Steps:
        1. Check cache (skip if already processed and not force)
        2. Parse text via PyMuPDF
        3. Extract tables via pdfplumber
        4. OCR fallback for sparse pages
        5. Detect fiscal year
        6. Merge into ordered ContentItems
        7. Cache result to disk

        Args:
            pdf_path: Path to the PDF file
            force_reprocess: Ignore cache

        Returns:
            DocumentRecord
        """
        company = company_from_filename(pdf_path.name)
        cache_file = self.cache_dir / f"{pdf_path.stem}.json"

        # Check cache
        if cache_file.exists() and not force_reprocess:
            logger.info(f"Loading cached: {pdf_path.name}")
            return self._load_from_cache(cache_file)

        logger.info(f"Processing: {pdf_path.name} | Company: {company}")

        # ---- Step 1: Parse text and layout ----
        parsed_doc: ParsedDocument = self.parser.parse(pdf_path, company)

        # ---- Step 2: Extract tables (all pages) ----
        page_tables = self.table_extractor.extract_all_tables(pdf_path)

        # ---- Step 3: OCR fallback for sparse pages ----
        page_raw_texts = {
            page.page_number: page.raw_text
            for page in parsed_doc.pages
        }
        ocr_supplements = self.ocr_handler.ocr_pdf_sparse_pages(pdf_path, page_raw_texts)

        # ---- Step 4: Detect fiscal year ----
        fiscal_year = self._detect_fiscal_year(parsed_doc)
        logger.info(f"Detected fiscal year: {fiscal_year}")

        # ---- Step 5: Get ticker ----
        ticker = settings.company_ticker_map.get(company, company)

        # ---- Step 6: Merge into ordered ContentItems ----
        items = self._merge_content(
            parsed_doc=parsed_doc,
            page_tables=page_tables,
            ocr_supplements=ocr_supplements,
            company=company,
            source_file=pdf_path.name,
            fiscal_year=fiscal_year,
        )

        record = DocumentRecord(
            company=company,
            ticker=ticker,
            source_file=pdf_path.name,
            fiscal_year=fiscal_year,
            total_pages=parsed_doc.total_pages,
            items=items,
        )

        # ---- Step 7: Cache to disk ----
        self._save_to_cache(record, cache_file)

        return record

    # ---------------------------------------------------------------- #
    # Internal: Content Merging
    # ---------------------------------------------------------------- #

    def _merge_content(
        self,
        parsed_doc: ParsedDocument,
        page_tables: dict,
        ocr_supplements: dict,
        company: str,
        source_file: str,
        fiscal_year: str,
    ) -> list[ContentItem]:
        """
        Merge text blocks, tables, and OCR text into an ordered list of ContentItems.
        Maintains document reading order (page by page, top to bottom).
        """
        items: list[ContentItem] = []
        current_section = "Introduction"
        current_subsection = ""

        for page in parsed_doc.pages:
            page_num = page.page_number

            # Track current section from heading blocks
            for block in page.text_blocks:
                if block.block_type == "heading":
                    if len(block.text) > 5:  # ignore very short headings (noise)
                        # Distinguish section vs subsection by font size
                        if block.font_size >= 14:
                            current_section = block.text.strip()
                            current_subsection = ""
                        else:
                            current_subsection = block.text.strip()

                # Add text/heading blocks as ContentItems
                if block.text.strip() and len(block.text.strip()) > 20:
                    items.append(ContentItem(
                        content=block.text.strip(),
                        content_type="text",
                        page_number=page_num,
                        section=current_section,
                        subsection=current_subsection,
                        company=company,
                        source_file=source_file,
                        fiscal_year=fiscal_year,
                        is_table=False,
                    ))

            # Add tables for this page (after text blocks, interleaved by page)
            if page_num in page_tables:
                for table in page_tables[page_num].tables:
                    if table.markdown.strip():
                        items.append(ContentItem(
                            content=table.markdown,
                            content_type="table",
                            page_number=page_num,
                            section=current_section,
                            subsection=current_subsection,
                            company=company,
                            source_file=source_file,
                            fiscal_year=fiscal_year,
                            is_table=True,
                        ))

            # Add OCR text for sparse pages
            if page_num in ocr_supplements:
                ocr_text = ocr_supplements[page_num]
                if ocr_text.strip():
                    items.append(ContentItem(
                        content=f"[OCR] {ocr_text.strip()}",
                        content_type="ocr_text",
                        page_number=page_num,
                        section=current_section,
                        subsection=current_subsection,
                        company=company,
                        source_file=source_file,
                        fiscal_year=fiscal_year,
                        is_table=False,
                    ))

        logger.debug(f"Merged {len(items)} content items for {source_file}")
        return items

    # ---------------------------------------------------------------- #
    # Internal: Fiscal Year Detection
    # ---------------------------------------------------------------- #

    def _detect_fiscal_year(self, parsed_doc: ParsedDocument) -> str:
        """
        Auto-detect fiscal year from document text.

        BSE reports typically state:
        - "for the year ended March 31, 2025" → FY2025
        - "Annual Report 2024-25" → FY2025
        - "Financial Year 2023-24" → FY2024

        Falls back to settings.default_fiscal_year if not found.
        """
        # Only scan first 10 pages (cover + ToC usually has this info)
        scan_text = ""
        for page in parsed_doc.pages[:10]:
            scan_text += page.raw_text + "\n"

        scan_text_lower = scan_text.lower()

        # Pattern 1: "year ended march 31, YYYY" — most reliable
        match = re.search(r'year\s+ended\s+march\s+31[,\s]+(\d{4})', scan_text_lower)
        if match:
            year = int(match.group(1))
            return f"FY{year}"

        # Pattern 2: "annual report YYYY-YY" (e.g. "2024-25" → FY2025)
        match = re.search(r'annual\s+report\s+(\d{4})-(\d{2})', scan_text_lower)
        if match:
            end_year_short = int(match.group(2))
            start_year = int(match.group(1))
            end_year = start_year // 100 * 100 + end_year_short
            return f"FY{end_year}"

        # Pattern 3: "financial year YYYY-YY" — anchored to avoid forward-looking mentions
        match = re.search(r'financial\s+year\s+(\d{4})-(\d{2})', scan_text_lower)
        if match:
            start_year = int(match.group(1))
            end_short = int(match.group(2))
            end_year = start_year // 100 * 100 + end_short
            if 2020 <= end_year <= 2030:
                return f"FY{end_year}"

        # Pattern 4: "FY YYYY-YY" or "FY2024-25" in original case (not forward-looking)
        match = re.search(r'\bfy\s*(\d{4})-(\d{2})\b', scan_text_lower)
        if match:
            start_year = int(match.group(1))
            end_short = int(match.group(2))
            end_year = start_year // 100 * 100 + end_short
            if 2020 <= end_year <= 2030:
                return f"FY{end_year}"

        # Pattern 5: "for the year YYYY-YY" — anchored context
        match = re.search(r'for\s+the\s+year\s+(\d{4})-(\d{2})\b', scan_text_lower)
        if match:
            start_year = int(match.group(1))
            end_short = int(match.group(2))
            end_year = start_year // 100 * 100 + end_short
            if 2020 <= end_year <= 2030:
                return f"FY{end_year}"

        logger.warning(f"Could not detect fiscal year for {parsed_doc.source_file}, defaulting to {settings.default_fiscal_year}")
        return settings.default_fiscal_year


    # ---------------------------------------------------------------- #
    # Cache helpers
    # ---------------------------------------------------------------- #

    def _save_to_cache(self, record: DocumentRecord, cache_file: Path) -> None:
        """Serialize DocumentRecord to JSON cache."""
        try:
            data = asdict(record)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cache {cache_file.name}: {e}")

    def _load_from_cache(self, cache_file: Path) -> DocumentRecord:
        """Load DocumentRecord from JSON cache."""
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = [ContentItem(**item) for item in data.pop("items", [])]
        record = DocumentRecord(**data, items=items)
        return record
