"""
src/retrieval/hybrid_retriever.py
===================================
Hybrid retrieval combining dense (Qdrant) + sparse (BM25) search via RRF.

Reciprocal Rank Fusion (RRF):
    A score fusion algorithm that combines ranked lists without needing
    to normalize their scores (since BM25 and cosine scores are on different scales).

    Formula: RRF(d) = Σ 1 / (k + rank(d))
    where k=60 (standard constant that prevents high ranks from dominating)

    Why RRF over simple score addition?
    - BM25 scores and cosine scores are NOT comparable (different scales)
    - RRF is rank-based → immune to scale differences
    - Empirically shown to outperform weighted score combination

Also implements:
    - Multi-query expansion: generates 3 paraphrased queries for better recall
    - Parent chunk promotion: returns parent context when a child is found
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from src.utils.logger import logger
from src.vectorstore.qdrant_store import QdrantStore, SearchResult
from src.vectorstore.bm25_index import BM25Index, BM25Result
from src.embeddings.embedder import Embedder


@dataclass
class HybridResult:
    """A single result after RRF fusion — combines dense + sparse signals."""
    chunk_id: str
    content: str
    rrf_score: float
    dense_rank: Optional[int] = None     # rank in dense results (None if not found)
    sparse_rank: Optional[int] = None    # rank in BM25 results (None if not found)
    company: str = ""
    ticker: str = ""
    source_file: str = ""
    fiscal_year: str = ""
    page_number: int = 0
    section: str = ""
    subsection: str = ""
    content_type: str = ""
    chunk_level: str = ""
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None  # fetched parent chunk text (if promoted)


class HybridRetriever:
    """
    Production hybrid retriever:
        1. Optionally expand query into multiple paraphrased versions
        2. Run dense search (Qdrant HNSW) for each query variant
        3. Run BM25 sparse search for each query variant
        4. Fuse all ranked lists using RRF
        5. (Optionally) promote child chunks to parent context
        6. Return top-K fused results for reranking

    This class does NOT do reranking — that's in reranker.py.
    """

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        bm25_index: Optional[BM25Index] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.qdrant = qdrant_store or QdrantStore()
        self.bm25 = bm25_index or BM25Index()
        self.embedder = embedder or Embedder()
        self.rrf_k = settings.rrf_k
        logger.info("HybridRetriever initialized")

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        company_filter: Optional[str | list[str]] = None,
        fiscal_year_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        expand_query: bool = True,
        promote_to_parent: bool = True,
    ) -> list[HybridResult]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User's question
            top_k: Number of fused results to return
            company_filter: Restrict search to specific company/companies
            fiscal_year_filter: Restrict to specific fiscal year
            content_type_filter: "text" | "table" | None
            expand_query: Whether to generate paraphrased query variants
            promote_to_parent: Whether to fetch and attach parent chunk content

        Returns:
            List of HybridResult sorted by RRF score descending
        """
        # Step 1: Query variants
        queries = [query]
        if expand_query and settings.num_expanded_queries > 1:
            expanded = self._expand_query_simple(query)
            queries = [query] + expanded
            logger.debug(f"Expanded to {len(queries)} query variants")

        # Step 2: Embed all query variants
        query_embeddings = self.embedder.embed_queries(queries)

        # Step 3: Dense retrieval for each query variant
        all_dense_results: list[list[SearchResult]] = []
        for q_emb in query_embeddings:
            dense = self.qdrant.search(
                query_vector=q_emb,
                top_k=settings.dense_top_k,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
                content_type_filter=content_type_filter,
                chunk_level_filter=None,   # search both child and atomic
            )
            all_dense_results.append(dense)

        # Step 4: BM25 sparse retrieval for each query variant
        all_sparse_results: list[list[BM25Result]] = []
        for q in queries:
            sparse = self.bm25.search(
                query=q,
                top_k=settings.sparse_top_k,
                company_filter=company_filter,
                fiscal_year_filter=fiscal_year_filter,
            )
            all_sparse_results.append(sparse)

        # Step 5: RRF fusion
        fused = self._rrf_fusion(all_dense_results, all_sparse_results)

        # Sort and truncate
        fused = sorted(fused.values(), key=lambda r: r.rrf_score, reverse=True)[:top_k]

        # Step 6: Promote children to parent context
        if promote_to_parent:
            fused = self._promote_to_parent(fused)

        logger.debug(f"Hybrid retrieval: {len(fused)} results after RRF fusion")
        return fused

    # ---------------------------------------------------------------- #
    # RRF Fusion
    # ---------------------------------------------------------------- #

    def _rrf_fusion(
        self,
        dense_lists: list[list[SearchResult]],
        sparse_lists: list[list[BM25Result]],
    ) -> dict[str, HybridResult]:
        """
        Apply Reciprocal Rank Fusion across all result lists.

        For each document, accumulates:
            RRF_score += 1 / (k + rank)
        across all ranked lists it appears in.
        """
        k = self.rrf_k
        fused: dict[str, HybridResult] = {}   # chunk_id → HybridResult

        # Process dense results
        for dense_list in dense_lists:
            for rank, result in enumerate(dense_list):
                cid = result.chunk_id
                rrf_increment = 1.0 / (k + rank + 1)

                if cid not in fused:
                    fused[cid] = HybridResult(
                        chunk_id=cid,
                        content=result.content,
                        rrf_score=0.0,
                        company=result.company,
                        ticker=result.ticker,
                        source_file=result.source_file,
                        fiscal_year=result.fiscal_year,
                        page_number=result.page_number,
                        section=result.section,
                        subsection=result.subsection,
                        content_type=result.content_type,
                        chunk_level=result.chunk_level,
                        parent_chunk_id=result.parent_chunk_id,
                    )

                fused[cid].rrf_score += rrf_increment
                if fused[cid].dense_rank is None or rank < fused[cid].dense_rank:
                    fused[cid].dense_rank = rank

        # Process sparse results
        for sparse_list in sparse_lists:
            for rank, result in enumerate(sparse_list):
                cid = result.chunk_id
                rrf_increment = 1.0 / (k + rank + 1)

                if cid not in fused:
                    fused[cid] = HybridResult(
                        chunk_id=cid,
                        content=result.content,
                        rrf_score=0.0,
                        company=result.company,
                        ticker=result.ticker,
                        source_file=result.source_file,
                        fiscal_year=result.fiscal_year,
                        page_number=result.page_number,
                        section=result.section,
                        subsection=result.subsection,
                        content_type=result.content_type,
                        chunk_level=result.chunk_level,
                        parent_chunk_id=result.parent_chunk_id,
                    )

                fused[cid].rrf_score += rrf_increment
                if fused[cid].sparse_rank is None or rank < fused[cid].sparse_rank:
                    fused[cid].sparse_rank = rank

        return fused

    # ---------------------------------------------------------------- #
    # Parent Promotion
    # ---------------------------------------------------------------- #

    def _promote_to_parent(self, results: list[HybridResult]) -> list[HybridResult]:
        """
        For child chunks: fetch the parent chunk content and attach it.
        The reranker will use the parent content for scoring.
        The LLM will also see the parent context (richer).
        """
        parent_ids_seen = set()

        for result in results:
            if result.chunk_level == "child" and result.parent_chunk_id:
                pid = result.parent_chunk_id

                if pid in parent_ids_seen:
                    continue   # already fetched

                parent_content = self.qdrant.fetch_parent_chunk(pid)
                if parent_content:
                    result.parent_content = parent_content
                    parent_ids_seen.add(pid)

        return results

    # ---------------------------------------------------------------- #
    # Simple Query Expansion (no LLM needed)
    # ---------------------------------------------------------------- #

    def _expand_query_simple(self, query: str) -> list[str]:
        """
        Generate simple query variants without calling an LLM.
        Used as a fallback if LLM-based expansion is not configured.

        Strategies:
        - Abbreviation expansion (PAT → Profit After Tax)
        - Reordering key terms
        """
        expansions = []
        q = query.strip()

        # Financial abbreviation expansion
        abbrev_map = {
            "PAT": "profit after tax",
            "PBT": "profit before tax",
            "EBITDA": "earnings before interest taxes depreciation amortization",
            "NPA": "non performing assets",
            "GNPA": "gross non performing assets",
            "NNPA": "net non performing assets",
            "ROE": "return on equity",
            "ROA": "return on assets",
            "EPS": "earnings per share",
            "P/E": "price to earnings ratio",
            "CAGR": "compound annual growth rate",
            "CASA": "current account savings account",
            "NIM": "net interest margin",
            "CRR": "cash reserve ratio",
            "SLR": "statutory liquidity ratio",
        }

        for abbrev, full in abbrev_map.items():
            if abbrev in q.upper():
                expanded = q.replace(abbrev, full).replace(abbrev.lower(), full)
                if expanded != q:
                    expansions.append(expanded)
                    break   # one expansion is enough

        return expansions[:settings.num_expanded_queries - 1]
