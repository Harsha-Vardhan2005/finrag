"""
src/generation/response_parser.py
===================================
Parses LLM responses to extract structured citation info.
"""

from __future__ import annotations
import re


def extract_citations(answer_text: str) -> list[dict]:
    """
    Extract [SOURCE N] citation markers from LLM answer text.

    Returns list of dicts with citation info parsed from the answer.
    """
    citations = []
    pattern = r'\[SOURCE\s+(\d+)\]'
    matches = re.finditer(pattern, answer_text)
    for match in matches:
        idx = int(match.group(1))
        if not any(c["index"] == idx for c in citations):
            citations.append({"index": idx, "marker": match.group(0)})
    return citations


def clean_answer(answer_text: str) -> str:
    """Remove redundant whitespace from LLM output."""
    lines = answer_text.split("\n")
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned).strip()
