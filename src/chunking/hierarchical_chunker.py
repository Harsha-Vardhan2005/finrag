"""
src/chunking/hierarchical_chunker.py
=====================================
Production-grade hierarchical (parent-child) chunker for financial documents.

Architecture:
  - PARENT chunks: large context windows (~1800 tokens) — sent to the LLM
  - CHILD chunks: small, precise (~350 tokens) — indexed in the vector store

Why parent-child?
  Search on small child chunks → precise, low-noise retrieval
  Return the parent chunk to the LLM → rich context for accurate answers

Special handling for financial documents:
  - Tables are NEVER split (atomic chunks, always kept whole)
  - Section boundaries are respected (never chunk across headings)
  - Metadata is propagated from parent to all its children
  - Chunk IDs are deterministic (SHA256-based) for deduplication

Usage:
    chunker = HierarchicalChunker()
    all_chunks = chunker.chunk_documents(document_records)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from src.utils.logger import logger
from src.ingestion.pipeline import DocumentRecord, ContentItem


# ------------------------------------------------------------------ #
# Data Classes
# ------------------------------------------------------------------ #

@dataclass
class Chunk:
    """
    A single chunk ready for embedding and vector storage.
    Every chunk carries its full metadata for filtered retrieval.
    """
    # Content
    chunk_id: str              # Deterministic SHA256-based unique ID
    content: str               # The actual text to embed + return to LLM
    content_type: str          # "text" | "table" | "ocr_text"

    # Hierarchy
    chunk_level: str           # "child" | "parent" | "atomic"
    parent_chunk_id: Optional[str] = None   # set on child chunks

    # Source metadata (used for Qdrant payload filtering)
    company: str = ""
    ticker: str = ""
    source_file: str = ""
    fiscal_year: str = ""
    page_number: int = 0
    section: str = ""
    subsection: str = ""

    # Stats
    token_count: int = 0
    char_count: int = 0


@dataclass
class ChunkedDocument:
    """All chunks produced from one DocumentRecord."""
    company: str
    source_file: str
    fiscal_year: str
    parent_chunks: list[Chunk] = field(default_factory=list)
    child_chunks: list[Chunk] = field(default_factory=list)
    atomic_chunks: list[Chunk] = field(default_factory=list)   # tables etc.

    @property
    def all_index_chunks(self) -> list[Chunk]:
        """Chunks to put in the vector index (child + atomic)."""
        return self.child_chunks + self.atomic_chunks

    @property
    def all_chunks(self) -> list[Chunk]:
        return self.parent_chunks + self.child_chunks + self.atomic_chunks


# ------------------------------------------------------------------ #
# Chunker
# ------------------------------------------------------------------ #

class HierarchicalChunker:
    """
    Produces hierarchical parent-child chunks from ingested documents.

    Strategy per content type:
        TEXT  → split into parent chunks (respect section boundaries)
                → split each parent into child chunks (overlapping)
        TABLE → kept as atomic chunks (never split)
        OCR   → treated same as text
    """

    def __init__(self):
        self.child_size = settings.child_chunk_size
        self.child_overlap = settings.child_chunk_overlap
        self.parent_size = settings.parent_chunk_size
        self.parent_overlap = settings.parent_chunk_overlap
        self.min_tokens = settings.min_chunk_tokens
        logger.info(
            f"HierarchicalChunker | child={self.child_size}t overlap={self.child_overlap}t "
            f"| parent={self.parent_size}t overlap={self.parent_overlap}t"
        )

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def chunk_documents(self, records: list[DocumentRecord]) -> list[ChunkedDocument]:
        """
        Chunk all DocumentRecords.

        Args:
            records: Output from IngestionPipeline.run_all()

        Returns:
            List of ChunkedDocument, one per input record
        """
        results = []
        total_child = 0
        total_parent = 0
        total_atomic = 0

        for record in records:
            chunked = self.chunk_document(record)
            results.append(chunked)
            total_child += len(chunked.child_chunks)
            total_parent += len(chunked.parent_chunks)
            total_atomic += len(chunked.atomic_chunks)
            logger.info(
                f"{record.source_file}: {len(chunked.parent_chunks)} parents, "
                f"{len(chunked.child_chunks)} children, {len(chunked.atomic_chunks)} atomic"
            )

        logger.info(
            f"Chunking complete: {total_parent} parents | "
            f"{total_child} children | {total_atomic} atomic | "
            f"Total index chunks: {total_child + total_atomic}"
        )
        return results

    def chunk_document(self, record: DocumentRecord) -> ChunkedDocument:
        """Chunk a single DocumentRecord."""
        chunked = ChunkedDocument(
            company=record.company,
            source_file=record.source_file,
            fiscal_year=record.fiscal_year,
        )

        # Separate items by type
        text_items: list[ContentItem] = []
        table_items: list[ContentItem] = []

        for item in record.items:
            if item.is_table:
                table_items.append(item)
            else:
                text_items.append(item)

        # ---- Process text items → parent-child hierarchy ----
        # First aggregate text into section-aware parent chunks
        parent_chunks = self._make_parent_chunks(text_items, record)

        for parent in parent_chunks:
            chunked.parent_chunks.append(parent)
            # Split each parent into child chunks
            children = self._make_child_chunks(parent)
            chunked.child_chunks.extend(children)

        # ---- Process table items → atomic chunks ----
        for item in table_items:
            atomic = self._make_atomic_table_chunk(item, record)
            if atomic:
                chunked.atomic_chunks.append(atomic)

        return chunked

    # ---------------------------------------------------------------- #
    # Parent Chunk Creation
    # ---------------------------------------------------------------- #

    def _make_parent_chunks(
        self,
        items: list[ContentItem],
        record: DocumentRecord,
    ) -> list[Chunk]:
        """
        Group text items into parent-sized chunks respecting section boundaries.

        Algorithm:
        1. Accumulate items until parent token limit is reached
        2. On section heading boundary → always start a new parent
        3. On token limit → split at sentence boundary if possible
        """
        parents: list[Chunk] = []
        current_texts: list[str] = []
        current_tokens = 0
        current_section = ""
        current_subsection = ""
        current_page = 0

        def flush_parent():
            nonlocal current_texts, current_tokens
            if not current_texts:
                return
            combined = "\n\n".join(current_texts)
            token_count = self._count_tokens(combined)
            if token_count < self.min_tokens:
                current_texts = []
                current_tokens = 0
                return

            chunk = Chunk(
                chunk_id=self._make_id(combined),
                content=combined,
                content_type="text",
                chunk_level="parent",
                company=record.company,
                ticker=record.ticker,
                source_file=record.source_file,
                fiscal_year=record.fiscal_year,
                page_number=current_page,
                section=current_section,
                subsection=current_subsection,
                token_count=token_count,
                char_count=len(combined),
            )
            parents.append(chunk)
            current_texts = []
            current_tokens = 0

        for item in items:
            item_tokens = self._count_tokens(item.content)

            # Section boundary → start a new parent
            section_changed = (
                item.section != current_section and
                item.section and
                current_texts
            )

            if section_changed:
                flush_parent()

            # Over size limit → flush first
            if current_tokens + item_tokens > self.parent_size and current_texts:
                flush_parent()

            # Update tracking
            current_section = item.section or current_section
            current_subsection = item.subsection or current_subsection
            current_page = item.page_number

            # Add item
            current_texts.append(item.content)
            current_tokens += item_tokens

        flush_parent()
        return parents

    # ---------------------------------------------------------------- #
    # Child Chunk Creation
    # ---------------------------------------------------------------- #

    def _make_child_chunks(self, parent: Chunk) -> list[Chunk]:
        """
        Split a parent chunk into overlapping child chunks.

        Uses sentence-aware splitting:
        - Split on sentence boundaries (. ! ?) where possible
        - Apply token-based sliding window with overlap
        - Each child inherits parent metadata + parent_chunk_id
        """
        sentences = self._split_into_sentences(parent.content)
        if not sentences:
            return []

        children: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        sentence_token_counts = [self._count_tokens(s) for s in sentences]

        i = 0
        while i < len(sentences):
            stokens = sentence_token_counts[i]

            # If adding this sentence exceeds child size → flush
            if current_tokens + stokens > self.child_size and current_sentences:
                child_text = " ".join(current_sentences).strip()
                if self._count_tokens(child_text) >= self.min_tokens:
                    children.append(self._make_child_chunk(child_text, parent, len(children)))

                # Overlap: keep last N tokens worth of sentences
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_sentences):
                    st = self._count_tokens(sent)
                    if overlap_tokens + st <= self.child_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += st
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentences[i])
            current_tokens += stokens
            i += 1

        # Flush remaining
        if current_sentences:
            child_text = " ".join(current_sentences).strip()
            if self._count_tokens(child_text) >= self.min_tokens:
                children.append(self._make_child_chunk(child_text, parent, len(children)))

        return children

    def _make_child_chunk(self, text: str, parent: Chunk, index: int) -> Chunk:
        """Create a single child chunk inheriting parent metadata."""
        return Chunk(
            chunk_id=self._make_id(text),
            content=text,
            content_type=parent.content_type,
            chunk_level="child",
            parent_chunk_id=parent.chunk_id,
            company=parent.company,
            ticker=parent.ticker,
            source_file=parent.source_file,
            fiscal_year=parent.fiscal_year,
            page_number=parent.page_number,
            section=parent.section,
            subsection=parent.subsection,
            token_count=self._count_tokens(text),
            char_count=len(text),
        )

    # ---------------------------------------------------------------- #
    # Atomic Table Chunks
    # ---------------------------------------------------------------- #

    def _make_atomic_table_chunk(
        self,
        item: ContentItem,
        record: DocumentRecord,
    ) -> Optional[Chunk]:
        """
        Tables are atomic — never split, always kept whole.
        Returns a single chunk with chunk_level='atomic'.
        """
        if not item.content.strip():
            return None

        token_count = self._count_tokens(item.content)

        # Add context header to table so LLM knows what it's reading
        header = f"[TABLE] Company: {item.company} | Year: {item.fiscal_year} | Section: {item.section}"
        if item.subsection:
            header += f" > {item.subsection}"
        header += f" | Page: {item.page_number}"

        enriched_content = f"{header}\n\n{item.content}"

        return Chunk(
            chunk_id=self._make_id(enriched_content),
            content=enriched_content,
            content_type="table",
            chunk_level="atomic",
            company=record.company,
            ticker=record.ticker,
            source_file=record.source_file,
            fiscal_year=record.fiscal_year,
            page_number=item.page_number,
            section=item.section,
            subsection=item.subsection,
            token_count=token_count,
            char_count=len(enriched_content),
        )

    # ---------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------- #

    def _count_tokens(self, text: str) -> int:
        """
        Fast approximate token count.
        Rule of thumb: 1 token ≈ 4 characters (OpenAI / LLaMA tokenizers).
        Good enough for chunking decisions; no need to load a full tokenizer.
        """
        return max(1, len(text) // 4)

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.
        Handles:
        - Standard sentence endings (. ! ?)
        - Abbreviations (Rs., Ltd., No., etc.) — avoid false splits
        - Numbered list items
        - Newline-based splits (paragraph boundaries)
        """
        # First split on newlines (paragraph breaks in financial docs)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        sentences = []
        for para in paragraphs:
            # Split on sentence-ending punctuation, but avoid splitting on
            # common financial abbreviations (Rs., Cr., Ltd., No., etc.)
            parts = re.split(
                r'(?<!\bRs)(?<!\bCr)(?<!\bLtd)(?<!\bNo)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bSt)'
                r'(?<=[.!?])\s+(?=[A-Z0-9₹])',
                para
            )
            sentences.extend([s.strip() for s in parts if s.strip()])

        return sentences

    def _make_id(self, content: str) -> str:
        """Generate a deterministic chunk ID from content hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
