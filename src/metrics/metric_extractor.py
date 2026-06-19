"""
src/metrics/metric_extractor.py
================================
Extracts structured financial KPIs from documents using the Groq LLM.

Returns clean Python dicts with numeric values — ready for Plotly charts.
Handles both banking metrics (NPA, NIM, CASA) and non-banking metrics (EBITDA, etc.)
"""

from __future__ import annotations

import json
import re
from typing import Optional

from config.settings import settings
from src.utils.logger import logger
from src.retrieval.hybrid_retriever import HybridResult
from src.generation.groq_client import GroqClient
from src.generation.prompts import build_metric_extraction_prompt


# Financial metrics we try to extract
BANKING_COMPANIES = {"HDFC", "ICICI", "AXIS", "SBI", "KOTAK", "KVB"}

KEY_METRICS_DISPLAY = {
    "revenue_crore": "Revenue (₹ Cr)",
    "net_profit_crore": "Net Profit (₹ Cr)",
    "net_profit_margin_pct": "Net Profit Margin (%)",
    "total_assets_crore": "Total Assets (₹ Cr)",
    "equity_crore": "Shareholders' Equity (₹ Cr)",
    "eps_inr": "EPS (₹)",
    "roe_pct": "Return on Equity (%)",
    "ebitda_crore": "EBITDA (₹ Cr)",
    "ebitda_margin_pct": "EBITDA Margin (%)",
    "debt_to_equity": "Debt / Equity",
    # Banking specific
    "npa_gross_pct": "Gross NPA (%)",
    "npa_net_pct": "Net NPA (%)",
    "nim_pct": "Net Interest Margin (%)",
}


class MetricExtractor:
    """
    Extracts structured financial KPIs using LLM JSON mode.
    Results are ready for Plotly chart rendering.
    """

    def __init__(self, groq_client: Optional[GroqClient] = None):
        self.llm = groq_client or GroqClient()

    def extract(
        self,
        company: str,
        results: list[HybridResult],
    ) -> dict:
        """
        Extract financial metrics from retrieved chunks.

        Args:
            company: Company name (e.g. "TCS")
            results: Retrieved + reranked chunks for this company

        Returns:
            Dict with 'company', 'fiscal_year', 'metrics', 'source_citation'
        """
        system_prompt, user_message = build_metric_extraction_prompt(company, results)

        try:
            extracted = self.llm.generate_json(system_prompt, user_message)
            logger.info(f"Extracted metrics for {company}: {list(extracted.get('metrics', {}).keys())}")
            return extracted
        except Exception as e:
            logger.error(f"Metric extraction failed for {company}: {e}")
            return {"company": company, "metrics": {}, "error": str(e)}

    def extract_multi(
        self,
        company_results: dict[str, list[HybridResult]],
    ) -> list[dict]:
        """
        Extract metrics for multiple companies (for comparison dashboard).

        Args:
            company_results: Dict mapping company → retrieved chunks

        Returns:
            List of metric dicts, one per company
        """
        all_metrics = []
        for company, results in company_results.items():
            metrics = self.extract(company, results)
            all_metrics.append(metrics)
        return all_metrics

    def to_comparison_table(self, metrics_list: list[dict]) -> dict:
        """
        Convert a list of company metric dicts into a comparison-ready structure.

        Returns:
            {
              "companies": ["TCS", "INFY"],
              "metric_name": {"TCS": 240893, "INFY": 153670},
              ...
            }
        """
        companies = [m.get("company", "Unknown") for m in metrics_list]
        comparison = {"companies": companies}

        for metric_key, display_name in KEY_METRICS_DISPLAY.items():
            row = {}
            for m in metrics_list:
                company = m.get("company", "Unknown")
                value = m.get("metrics", {}).get(metric_key)
                row[company] = value
            # Only include metric if at least one company has data
            if any(v is not None for v in row.values()):
                comparison[display_name] = row

        return comparison
