"""
src/vectorstore/bm25_index.py
==============================
BM25 sparse index for keyword-based retrieval.

Why BM25 alongside dense vectors?
    Dense vectors capture SEMANTIC meaning ("revenue growth" ≈ "sales increase")
    BM25 captures EXACT TERM matching ("₹4,234 crore" or "NPA 2.3%")

Financial documents are full of exact numbers and acronyms that
dense retrieval may miss. BM25 ensures we always catch exact matches.

This index is combined with Qdrant dense search via RRF in hybrid_retriever.py.

Storage:
    - BM25 index serialized to ./data/bm25_index/bm25.pkl
    - Document list serialized to ./data/bm25_index/docs.json
    - Both files must be regenerated when documents change
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.utils.logger import logger
from src.chunking.hierarchical_chunker import Chunk


@dataclass
class BM25Result:
    """A single result from BM25 keyword search."""
    chunk_id: str
    content: str
    bm25_score: float
    company: str
    ticker: str
    source_file: str
    fiscal_year: str
    page_number: int
    section: str
    subsection: str
    content_type: str
    chunk_level: str
    parent_chunk_id: Optional[str] = None


class BM25Index:
    """
    BM25 sparse index over all indexed chunks.

    Build once → persist to disk → reload on startup.
    Supports company/year filtering (post-filter after search).
    """

    INDEX_FILE = "bm25.pkl"
    DOCS_FILE = "docs.json"

    def __init__(self):
        self.index_dir = settings.bm25_index_path
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25: Optional[BM25Okapi] = None
        self._docs: list[dict] = []   # stores chunk metadata + content

        # Try to load existing index
        if self._index_exists():
            self._load()

    # ---------------------------------------------------------------- #
    # Build
    # ---------------------------------------------------------------- #

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from a list of chunks.
        Indexes the same chunks that go into Qdrant (child + atomic).

        Args:
            chunks: List of Chunk objects to index
        """
        logger.info(f"Building BM25 index over {len(chunks)} chunks...")

        self._docs = []
        tokenized_corpus = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            tokenized_corpus.append(tokens)

            # Store chunk metadata for result reconstruction
            self._docs.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "company": chunk.company,
                "ticker": chunk.ticker,
                "source_file": chunk.source_file,
                "fiscal_year": chunk.fiscal_year,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "content_type": chunk.content_type,
                "chunk_level": chunk.chunk_level,
                "parent_chunk_id": chunk.parent_chunk_id or "",
            })

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._save()
        logger.info(f"BM25 index built and saved | {len(chunks)} documents")

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #

    def search(
        self,
        query: str,
        top_k: int = 30,
        company_filter: Optional[str | list[str]] = None,
        fiscal_year_filter: Optional[str] = None,
    ) -> list[BM25Result]:
        """
        BM25 keyword search with optional post-filtering.

        Args:
            query: Search query string
            top_k: Number of results to return
            company_filter: Filter by company name or list of companies
            fiscal_year_filter: Filter by fiscal year

        Returns:
            List of BM25Result sorted by score descending
        """
        if self._bm25 is None:
            logger.warning("BM25 index not loaded — call build() first")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Pair scores with doc metadata and sort
        scored_docs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for idx, score in scored_docs:
            if score <= 0:
                continue
            if len(results) >= top_k * 3:   # over-retrieve for post-filtering
                break

            doc = self._docs[idx]

            # Post-filter by company
            if company_filter:
                if isinstance(company_filter, list):
                    if doc["company"] not in company_filter:
                        continue
                else:
                    if doc["company"] != company_filter:
                        continue

            # Post-filter by fiscal year
            if fiscal_year_filter and doc["fiscal_year"] != fiscal_year_filter:
                continue

            results.append(BM25Result(
                chunk_id=doc["chunk_id"],
                content=doc["content"],
                bm25_score=float(score),
                company=doc["company"],
                ticker=doc["ticker"],
                source_file=doc["source_file"],
                fiscal_year=doc["fiscal_year"],
                page_number=doc["page_number"],
                section=doc["section"],
                subsection=doc["subsection"],
                content_type=doc["content_type"],
                chunk_level=doc["chunk_level"],
                parent_chunk_id=doc["parent_chunk_id"] or None,
            ))

            if len(results) >= top_k:
                break

        return results

    # ---------------------------------------------------------------- #
    # Tokenizer
    # ---------------------------------------------------------------- #

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Financial document specific:
        - Preserve numbers (4234, 2.3%) as tokens
        - Preserve ₹ symbol
        - Lowercase everything
        - Remove punctuation except . % ₹ (financial symbols)
        - Preserve common financial acronyms as single tokens
        """
        text = text.lower()

        # Keep numbers with decimals and % intact
        text = re.sub(r'[^\w\s₹.%,/-]', ' ', text)

        # Split on whitespace
        tokens = text.split()

        # Remove very short tokens (< 2 chars) except numbers
        tokens = [t for t in tokens if len(t) >= 2 or t.isdigit()]

        # Remove pure punctuation tokens
        tokens = [t for t in tokens if re.search(r'[a-zA-Z0-9]', t)]

        return tokens

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    def _index_exists(self) -> bool:
        return (
            (self.index_dir / self.INDEX_FILE).exists() and
            (self.index_dir / self.DOCS_FILE).exists()
        )

    def _save(self) -> None:
        """Persist index to disk."""
        with open(self.index_dir / self.INDEX_FILE, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(self.index_dir / self.DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False)
        logger.debug("BM25 index saved to disk")

    def _load(self) -> None:
        """Load index from disk."""
        try:
            with open(self.index_dir / self.INDEX_FILE, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(self.index_dir / self.DOCS_FILE, "r", encoding="utf-8") as f:
                self._docs = json.load(f)
            logger.info(f"BM25 index loaded | {len(self._docs)} documents")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e} — will rebuild on next ingest")
            self._bm25 = None
            self._docs = []
