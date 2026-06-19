"""
src/comparison/company_comparator.py
=====================================
Multi-company retrieval and comparison engine.

Handles the "Compare HDFC vs ICICI vs AXIS on NPA ratios" use case by:
1. Running hybrid retrieval separately for each company
2. Feeding all results into the comparison prompt
3. Returning structured comparison data + Plotly chart config
"""

from __future__ import annotations

from typing import Optional

from config.settings import settings
from src.utils.logger import logger
from src.retrieval.hybrid_retriever import HybridRetriever, HybridResult
from src.retrieval.reranker import Reranker
from src.generation.groq_client import GroqClient
from src.generation.prompts import build_comparison_prompt


class CompanyComparator:
    """
    Handles cross-company comparison queries.
    Retrieves data per-company then synthesizes a comparison.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        groq_client: Optional[GroqClient] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.llm = groq_client or GroqClient()

    def compare(
        self,
        query: str,
        companies: list[str],
        fiscal_year: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Run a comparison query across multiple companies.

        Args:
            query: The comparison question (e.g. "Compare revenue growth")
            companies: List of company names to compare
            fiscal_year: Optional fiscal year filter
            stream: If True, yields text chunks for streaming UI

        Returns:
            (company_results_dict, answer_text) if stream=False
            Generator of text chunks if stream=True
        """
        logger.info(f"Comparison query: '{query}' | Companies: {companies}")

        # Retrieve + rerank for each company separately
        company_results: dict[str, list[HybridResult]] = {}

        for company in companies:
            candidates = self.retriever.retrieve(
                query=query,
                top_k=50,
                company_filter=company,
                fiscal_year_filter=fiscal_year,
                expand_query=True,
                promote_to_parent=True,
            )

            # Rerank to get top 4 per company
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=4,
            )

            company_results[company] = reranked
            logger.info(f"  {company}: {len(reranked)} relevant chunks")

        # Build comparison prompt
        system_prompt, user_message = build_comparison_prompt(query, company_results)

        if stream:
            return company_results, self.llm.generate_stream(system_prompt, user_message)
        else:
            answer = self.llm.generate(system_prompt, user_message)
            return company_results, answer

    def get_available_companies(self) -> list[str]:
        """Return list of all companies in the vector store."""
        return list(settings.company_ticker_map.keys())
