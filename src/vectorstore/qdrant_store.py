"""
src/vectorstore/qdrant_store.py
================================
Qdrant vector store operations — collection management, upsert, search.

Running in LOCAL mode (no Docker/server needed):
    QdrantClient(path="./data/qdrant_store") stores everything to disk.
    This is perfectly fine for our scale (~15K chunks).

HNSW Index:
    m=16 — controls graph connectivity (higher = better recall, more memory)
    ef_construct=200 — build-time search depth (higher = better index quality)
    These values are production defaults used by major RAG systems.

Payload Filtering:
    Every chunk is stored with its metadata as a Qdrant "payload".
    This allows queries like:
        "only search in TCS documents"
        "only search FY2025 data"
        "only search table chunks"
    Without payload filtering, multi-company RAG would mix up contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScoredPoint,
)

from config.settings import settings
from src.utils.logger import logger
from src.chunking.hierarchical_chunker import Chunk


@dataclass
class SearchResult:
    """A single result from vector search."""
    chunk_id: str
    content: str
    score: float
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


class QdrantStore:
    """
    Manages the Qdrant vector store for the financial RAG system.
    Handles collection creation, chunk upsert, and similarity search.
    """

    def __init__(self):
        # Connect to Cloud if URL provided, else fallback to Local disk
        if settings.qdrant_url and settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                port=443,
                api_key=settings.qdrant_api_key,
                timeout=60,
            )
            mode_msg = f"Cloud | url: {settings.qdrant_url}"
        else:
            self.client = QdrantClient(
                path=str(settings.qdrant_path),
                timeout=60,
            )
            mode_msg = f"Local | path: {settings.qdrant_path}"
            
        self.collection_name = settings.qdrant_collection
        self.dim = settings.embedding_dim
        logger.info(f"QdrantStore initialized | collection: {self.collection_name} | {mode_msg}")

    # ---------------------------------------------------------------- #
    # Collection Management
    # ---------------------------------------------------------------- #

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with HNSW index.

        Args:
            recreate: If True, drop and recreate the collection (fresh start).
                      Use this if you change embedding dims or want to re-index.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                logger.warning(f"Dropping existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' already exists — skipping creation")
                return

        logger.info(f"Creating collection '{self.collection_name}' | dim={self.dim} | HNSW")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,   # cosine similarity (L2 normalized → same as dot product)
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.hnsw_m,
                ef_construct=settings.hnsw_ef_construct,
                full_scan_threshold=10000,   # use exact search below this count
                on_disk=False,               # keep index in RAM for speed
            ),
        )

        logger.info(f"Collection created successfully | HNSW m={settings.hnsw_m}, ef_construct={settings.hnsw_ef_construct}")

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in existing

    def get_collection_info(self) -> dict:
        """Get collection stats (useful for debugging)."""
        if not self.collection_exists():
            return {"status": "does not exist", "vectors_count": 0}
        info = self.client.get_collection(self.collection_name)
        # qdrant-client API varies by version — handle gracefully
        try:
            # Newer qdrant-client uses points_count
            points = info.points_count or 0
        except AttributeError:
            points = 0
        try:
            vectors = info.vectors_count or points
        except AttributeError:
            vectors = points
        return {
            "vectors_count": vectors,
            "points_count": points,
            "status": str(info.status),
        }

    # ---------------------------------------------------------------- #
    # Upsert
    # ---------------------------------------------------------------- #

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Store chunks with their embeddings in Qdrant.

        Args:
            chunks: List of Chunk objects
            embeddings: np.ndarray of shape (len(chunks), 768)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            # Qdrant requires a numeric or UUID point ID
            # We use a hash of chunk_id converted to int
            point_id = int(chunk.chunk_id, 16) % (2**63)   # SHA256 hex → int

            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "content_type": chunk.content_type,
                "chunk_level": chunk.chunk_level,
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "company": chunk.company,
                "ticker": chunk.ticker,
                "source_file": chunk.source_file,
                "fiscal_year": chunk.fiscal_year,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            ))

        # Batch upsert in groups of 100 (Qdrant recommended batch size)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        logger.info(f"Upserted {len(points)} chunks into Qdrant")

    # ---------------------------------------------------------------- #
    # Search
    # ---------------------------------------------------------------- #

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 30,
        company_filter: Optional[str | list[str]] = None,
        fiscal_year_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        chunk_level_filter: Optional[str] = "child",   # default: search child chunks
    ) -> list[SearchResult]:
        """
        Dense vector similarity search with optional payload filters.

        Args:
            query_vector: L2-normalized query embedding (768-dim)
            top_k: Number of results to return
            company_filter: Filter by company name (str) or list of companies
            fiscal_year_filter: Filter by fiscal year (e.g. "FY2025")
            content_type_filter: Filter by "text", "table", or "ocr_text"
            chunk_level_filter: "child" | "atomic" | "parent" | None (no filter)

        Returns:
            List of SearchResult objects sorted by score descending
        """
        must_conditions = []

        # Company filter
        if company_filter:
            if isinstance(company_filter, list):
                must_conditions.append(FieldCondition(
                    key="company",
                    match=MatchAny(any=company_filter),
                ))
            else:
                must_conditions.append(FieldCondition(
                    key="company",
                    match=MatchValue(value=company_filter),
                ))

        # Fiscal year filter
        if fiscal_year_filter:
            must_conditions.append(FieldCondition(
                key="fiscal_year",
                match=MatchValue(value=fiscal_year_filter),
            ))

        # Content type filter
        if content_type_filter:
            must_conditions.append(FieldCondition(
                key="content_type",
                match=MatchValue(value=content_type_filter),
            ))

        # Chunk level filter (default: child chunks only for retrieval)
        if chunk_level_filter:
            must_conditions.append(FieldCondition(
                key="chunk_level",
                match=MatchValue(value=chunk_level_filter),
            ))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # qdrant-client >= 1.7: use query_points() instead of deprecated search()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=settings.score_threshold,
            with_payload=True,
        )
        hits = response.points  # list[ScoredPoint]

        results = []
        for hit in hits:
            p = hit.payload
            results.append(SearchResult(
                chunk_id=p.get("chunk_id", ""),
                content=p.get("content", ""),
                score=hit.score,
                company=p.get("company", ""),
                ticker=p.get("ticker", ""),
                source_file=p.get("source_file", ""),
                fiscal_year=p.get("fiscal_year", ""),
                page_number=p.get("page_number", 0),
                section=p.get("section", ""),
                subsection=p.get("subsection", ""),
                content_type=p.get("content_type", ""),
                chunk_level=p.get("chunk_level", ""),
                parent_chunk_id=p.get("parent_chunk_id") or None,
            ))

        return results

    def search_tables_only(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        company_filter: Optional[str | list[str]] = None,
    ) -> list[SearchResult]:
        """Convenience: search only table (atomic) chunks."""
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            company_filter=company_filter,
            chunk_level_filter="atomic",
        )

    def fetch_parent_chunk(self, parent_chunk_id: str) -> Optional[str]:
        """
        Fetch a parent chunk's content given a parent_chunk_id.
        Used in parent-child retrieval: find child → return parent content to LLM.
        """
        # qdrant-client >= 1.7: scroll() uses query_filter (not scroll_filter)
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=parent_chunk_id),
                    )]
                ),
                limit=1,
                with_payload=True,
            )
            points, _ = results
        except TypeError:
            # Newer API uses query_filter instead of scroll_filter
            results = self.client.scroll(
                collection_name=self.collection_name,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=parent_chunk_id),
                    )]
                ),
                limit=1,
                with_payload=True,
            )
            points, _ = results

        if points:
            return points[0].payload.get("content", "")
        return None
