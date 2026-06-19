"""
src/retrieval/reranker.py
==========================
Cross-encoder reranker using BAAI/bge-reranker-base.

Why rerank?
    Bi-encoder (embedding model) retrieves fast but coarsely.
    Cross-encoder sees the FULL (query, document) pair jointly → much
    more accurate relevance scoring, but too slow to run on 10K+ chunks.

    Solution (standard production pattern):
        1. Bi-encoder retrieves top-50 candidates (fast, approximate)
        2. Cross-encoder reranks just those 50 (accurate, manageable cost)
        3. Return top-5 to LLM

    This 2-stage approach gives both speed AND accuracy.

BAAI/bge-reranker-base:
    - Free, runs locally on RTX 3050
    - ~550MB download (cached after first load)
    - Significantly better than bi-encoder ranking on financial Q&A
"""

from __future__ import annotations

from typing import Optional

# config first to suppress TF
from config.settings import settings
from src.utils.logger import logger
from src.retrieval.hybrid_retriever import HybridResult

# Lazy import — only load when first reranking call is made
_reranker = None


def get_reranker():
    """Lazy-initialize the cross-encoder reranker (loads model on first call)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        model_name = settings.reranker_model
        device = settings.get_device()
        logger.info(f"Loading reranker model: {model_name} → {device}")
        _reranker = CrossEncoder(model_name, device=device, max_length=512)
        logger.info("Reranker model loaded")
    return _reranker


class Reranker:
    """
    Cross-encoder reranker — takes hybrid retrieval candidates and re-scores them.
    """

    def __init__(self):
        # Don't load model here — lazy load on first use
        pass

    def rerank(
        self,
        query: str,
        candidates: list[HybridResult],
        top_n: int = 5,
        use_parent_content: bool = True,
    ) -> list[HybridResult]:
        """
        Rerank hybrid retrieval candidates using the cross-encoder.

        Args:
            query: The original user query
            candidates: Top-K results from hybrid retrieval
            top_n: How many reranked results to return (to the LLM)
            use_parent_content: If True and parent_content is available,
                                 rerank using the richer parent context

        Returns:
            Top-N reranked HybridResult objects sorted by cross-encoder score
        """
        if not candidates:
            return []

        reranker = get_reranker()

        # Build (query, passage) pairs for the cross-encoder
        pairs = []
        for candidate in candidates:
            # Use parent content if available (richer context)
            if use_parent_content and candidate.parent_content:
                passage = candidate.parent_content
            else:
                passage = candidate.content

            # Truncate very long passages to avoid exceeding cross-encoder max length
            # Cross-encoder max is 512 tokens — rough character estimate: ~1800 chars
            if len(passage) > 1800:
                passage = passage[:1800] + "..."

            pairs.append([query, passage])

        # Score all pairs
        logger.debug(f"Reranking {len(pairs)} candidates...")
        scores = reranker.predict(pairs, show_progress_bar=False)

        # Attach scores to candidates
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-N with scores attached (store in rrf_score for display)
        result = []
        for candidate, score in scored[:top_n]:
            # Preserve the original candidate, just update the displayed score
            candidate.rrf_score = float(score)   # reuse field to store reranker score
            result.append(candidate)

        logger.debug(f"Reranking complete: returning top {len(result)} chunks")
        return result

    def rerank_with_scores(
        self,
        query: str,
        candidates: list[HybridResult],
        top_n: int = 5,
    ) -> list[tuple[HybridResult, float]]:
        """
        Rerank and return (result, score) tuples (useful for evaluation/debugging).
        """
        reranked = self.rerank(query, candidates, top_n=top_n)
        return [(r, r.rrf_score) for r in reranked]
