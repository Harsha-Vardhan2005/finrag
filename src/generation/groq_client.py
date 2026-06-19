"""
src/generation/groq_client.py
==============================
Groq LLM client with retry logic, streaming support, and JSON mode.

Groq specifics:
- Runs on custom LPU hardware → ~10x faster than OpenAI (500+ tokens/sec)
- Free tier has generous rate limits (14,400 requests/day)
- Model: llama-3.3-70b-versatile (128K context window)
- JSON mode: pass response_format={"type": "json_object"} for structured output

Rate limit handling:
- Groq free tier: 30 req/min, 14,400 req/day
- We use tenacity for exponential backoff retry on rate limit errors
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Generator

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from src.utils.logger import logger


class GroqClient:
    """
    Wrapper around the Groq API client.
    Handles retries, streaming, and JSON-mode generation.
    """

    def __init__(self):
        api_key = settings.groq_api_key
        if not api_key:
            # Try environment directly as fallback
            api_key = os.environ.get("GROQ_API_KEY", "")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set! Add it to your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )

        self.client = Groq(api_key=api_key)
        self.model = settings.groq_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        logger.info(f"GroqClient initialized | model: {self.model}")

    # ---------------------------------------------------------------- #
    # Standard Generation
    # ---------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response from Groq LLM.

        Args:
            system_prompt: System instruction
            user_message: User question + context
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, forces JSON output (for metric extraction)

        Returns:
            Generated text response
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - start_time

        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.debug(f"Groq response: {tokens_used} tokens in {elapsed:.2f}s")
        return content

    # ---------------------------------------------------------------- #
    # Streaming Generation
    # ---------------------------------------------------------------- #

    def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response (for real-time UI display).

        Yields text chunks as they arrive from Groq.
        Perfect for Streamlit's st.write_stream().

        Args:
            system_prompt: System instruction
            user_message: User question + context

        Yields:
            String chunks of the response as they stream in
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ---------------------------------------------------------------- #
    # JSON Extraction
    # ---------------------------------------------------------------- #

    def generate_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> dict:
        """
        Generate structured JSON output.
        Uses Groq's JSON mode to guarantee valid JSON response.

        Args:
            system_prompt: System instruction (should specify JSON schema)
            user_message: User message with context

        Returns:
            Parsed Python dict from the JSON response
        """
        raw = self.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            json_mode=True,
            temperature=0.0,    # zero temperature for deterministic JSON
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Groq response: {e}\nRaw: {raw[:200]}")
            return {}

    # ---------------------------------------------------------------- #
    # Query Expansion (LLM-based)
    # ---------------------------------------------------------------- #

    def expand_query(self, query: str) -> list[str]:
        """
        Use the LLM to generate paraphrased query variants.
        Falls back to empty list on failure.
        """
        from src.generation.prompts import build_query_expansion_prompt
        system_prompt, user_message = build_query_expansion_prompt(query)

        try:
            result = self.generate_json(system_prompt, user_message)
            if isinstance(result, list):
                return [q for q in result if isinstance(q, str)]
            # Some models return {"queries": [...]}
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, list):
                        return [q for q in v if isinstance(q, str)]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        return []
