"""
src/generation/prompts.py
==========================
All prompt templates for the financial RAG system.

Keeping prompts in one place makes them easy to iterate on without
touching business logic. Each prompt is a function that takes
context + query and returns a formatted string.

Prompt Design Principles:
1. System prompt establishes the "financial analyst" persona
2. Context is clearly delimited (prevents prompt injection)
3. Citations are mandatory — LLM must always reference the source
4. Explicit fallback: if data not in context, say so (reduce hallucination)
5. Format instructions: use Markdown tables for comparisons
"""

from __future__ import annotations

from src.retrieval.hybrid_retriever import HybridResult


# ------------------------------------------------------------------ #
# System Prompts
# ------------------------------------------------------------------ #

FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a senior financial analyst AI specializing in Indian listed companies on BSE/NSE.

You answer questions based EXCLUSIVELY on the provided context from official BSE annual reports and financial filings.

## Core Rules & Tone:
1. **Be Conversational & Comprehensive**: Write like a professional, helpful financial advisor (like ChatGPT). Do NOT give a 1-sentence answer or just a raw list of numbers. Always formulate a rich, flowing, conversational response.
2. **Structure Your Answer**: 
   - Start with a clear, confident opening sentence.
   - Present the requested numbers clearly (using bullet points or bold text).
   - Add 2-3 sentences of helpful surrounding context, insights, or trends to flesh out the answer and provide value.
3. **Expert Persona (No Filler)**: Speak naturally and authoritatively. Do NOT use AI filler phrases like "based on the provided context", "according to the documents", or "can be assessed by looking at". Just state the facts as if you know them.
4. **Financial Terminology**: Treat "Total Income" as "Revenue". Treat "Profit for the period" or "Profit after tax" as "Net Profit". Treat "Year ended March 31, 2025" as FY2025. You are an expert—do NOT add pedantic disclaimers saying the exact word wasn't found if a standard synonym is present. Answer smoothly and confidently.
5. **Data Integrity**: Only use the provided context. If data is genuinely missing, say "This information is not available in the provided documents."
6. **No Citations**: Do NOT add inline bracket citations or a "Sources" block at the end. The UI handles this automatically.

## Formatting Guidelines:
- **Currency Format**: Always format Indian currency properly as `₹[Amount] Crore` (e.g., **₹259,286 Crore**). Never write "259,286 (in crore)".
- Use **bold** for key financial figures.
- Use Markdown tables for comparisons.
- Always write in full, engaging paragraphs rather than just robotic bullet points.
"""

COMPARISON_SYSTEM_PROMPT = """You are a senior financial analyst AI specializing in comparative analysis of Indian listed companies.

You compare companies based EXCLUSIVELY on the provided context from official BSE annual reports.

## Your Rules:
1. Present comparisons as Markdown tables whenever possible
2. Always state the fiscal year for each data point
3. Note which company performs better and why, based on the numbers
4. If data for a company is missing for a specific metric, indicate "N/A (not in documents)"
5. Do NOT include inline bracket citations or a "Sources" block at the end. The UI handles this automatically.

## Comparison Format:
Always structure your response as:
1. Summary table (key metrics side by side)
2. Detailed analysis (key differences and insights)
"""

METRIC_EXTRACTION_SYSTEM_PROMPT = """You are a financial data extraction AI. Extract specific financial metrics from the provided context.

Return your response as valid JSON only. No prose, no explanation — just the JSON.
If a metric is not found in the context, use null for its value.

Always include the fiscal year and source citation for each metric.
"""


# ------------------------------------------------------------------ #
# Context Formatting
# ------------------------------------------------------------------ #

def format_context_block(results: list[HybridResult], use_parent: bool = True) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM.
    """
    context_parts = []

    for i, result in enumerate(results, 1):
        content = result.content
        if use_parent and result.parent_content:
            content = result.parent_content

        # Clean section name — fallback if empty or just punctuation
        section = result.section.strip().strip("'\"") or "Financial Data"
        subsection = result.subsection.strip().strip("'\"")

        citation = f"[SOURCE {i}] {result.company} | {result.fiscal_year} | {section}"
        if subsection:
            citation += f" > {subsection}"
        citation += f" | Page {result.page_number} | File: {result.source_file}"
        if result.content_type == "table":
            citation += " | TYPE: TABLE"

        context_parts.append(f"{citation}\n{content}")

    return "\n\n---\n\n".join(context_parts)



# ------------------------------------------------------------------ #
# QA Prompt
# ------------------------------------------------------------------ #

def build_qa_prompt(
    query: str,
    results: list[HybridResult],
    company_context: str = "",
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_message) pair for single-company Q&A.

    Returns:
        (system_prompt, user_message) tuple for Groq API
    """
    context = format_context_block(results, use_parent=True)

    user_message = f"""## Context from Financial Documents
{context}

---

## Question
{query}

{f"## Company Context: {company_context}" if company_context else ""}

Please answer based solely on the context provided above.
"""
    return FINANCIAL_ANALYST_SYSTEM_PROMPT, user_message


# ------------------------------------------------------------------ #
# Comparison Prompt
# ------------------------------------------------------------------ #

def build_comparison_prompt(
    query: str,
    company_results: dict[str, list[HybridResult]],
) -> tuple[str, str]:
    """
    Build prompt for cross-company comparison.

    Args:
        query: The comparison question
        company_results: Dict mapping company_name → retrieved results

    Returns:
        (system_prompt, user_message) tuple
    """
    context_sections = []

    for company, results in company_results.items():
        company_context = format_context_block(results, use_parent=True)
        context_sections.append(f"### {company} Documents\n{company_context}")

    full_context = "\n\n".join(context_sections)

    user_message = f"""## Financial Documents Context

{full_context}

---

## Comparison Request
{query}

Please provide:
1. A Markdown comparison table with key metrics
2. Analysis of the differences
3. Which company performs better on this metric and why
"""
    return COMPARISON_SYSTEM_PROMPT, user_message


# ------------------------------------------------------------------ #
# Metric Extraction Prompt
# ------------------------------------------------------------------ #

METRIC_EXTRACTION_SCHEMA = {
    "company": "string",
    "fiscal_year": "string",
    "metrics": {
        "revenue_crore": "number or null",
        "net_profit_crore": "number or null",
        "net_profit_margin_pct": "number or null",
        "total_assets_crore": "number or null",
        "total_debt_crore": "number or null",
        "equity_crore": "number or null",
        "eps_inr": "number or null",
        "roe_pct": "number or null",
        "debt_to_equity": "number or null",
        "npa_gross_pct": "number or null (banks only)",
        "npa_net_pct": "number or null (banks only)",
        "nim_pct": "number or null (banks only)",
        "ebitda_crore": "number or null",
        "ebitda_margin_pct": "number or null",
    },
    "source_citation": "string"
}

def build_metric_extraction_prompt(
    company: str,
    results: list[HybridResult],
) -> tuple[str, str]:
    """
    Build prompt for structured metric extraction.
    Uses compact child content (not parent) to stay under Groq token limits.
    """
    # Use child content only (short ~350 tokens each), truncated — avoids 12K token limit
    context = format_context_block(results, use_parent=False)
    # Hard truncate to ~6000 chars to stay safely under token limit
    if len(context) > 6000:
        context = context[:6000] + "\n...[truncated]"

    schema_str = str(METRIC_EXTRACTION_SCHEMA)

    user_message = f"""Extract financial metrics for {company} from the following context.

## Context
{context}

---

Return ONLY a valid JSON object matching this schema:
{schema_str}

Use null for any metric not found in the context.
All monetary values should be in Indian Rupees Crore (Rs. Crore).
All percentage values should be numeric (e.g., 19.5 not "19.5%").
"""
    return METRIC_EXTRACTION_SYSTEM_PROMPT, user_message


# ------------------------------------------------------------------ #
# Query Expansion Prompt (LLM-based)
# ------------------------------------------------------------------ #

QUERY_EXPANSION_SYSTEM = """Generate 3 different phrasings of the given financial question.
Each rephrasing should capture the same intent but use different terminology.
Return ONLY a JSON array of 3 strings. No explanation."""

def build_query_expansion_prompt(query: str) -> tuple[str, str]:
    """Build prompt for LLM-based query expansion."""
    user_message = f"""Original question: {query}

Generate 3 alternative phrasings. Return as JSON array:
["rephrasing 1", "rephrasing 2", "rephrasing 3"]"""
    return QUERY_EXPANSION_SYSTEM, user_message
