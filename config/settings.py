"""
config/settings.py
==================
Central configuration for the entire RAG pipeline.
All tunable hyperparameters live here — no magic numbers scattered in code.
Uses pydantic-settings to load from .env file automatically.

IMPORTANT: This file sets USE_TF=0 / USE_TORCH=1 at import time to prevent
the broken TensorFlow installation on this machine from crashing sentence_transformers.
Always import config.settings BEFORE importing sentence_transformers or transformers.
"""

import os
# Must be set BEFORE importing transformers / sentence_transformers
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Production RAG Configuration.
    Values are loaded from .env file (override defaults by editing .env).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    raw_pdfs_path: Path = Field(default=Path("./raw_pdfs"))
    qdrant_path: Path = Field(default=Path("./data/qdrant_store"))
    bm25_index_path: Path = Field(default=Path("./data/bm25_index"))
    processed_docs_path: Path = Field(default=Path("./data/processed"))

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")
    embedding_dim: int = Field(default=1024)          # bge-large output dimension (NOT 768 — that's bge-base)

    reranker_model: str = Field(default="BAAI/bge-reranker-base")

    device: str = Field(default="auto")              # auto | cuda | cpu

    # ------------------------------------------------------------------ #
    # Qdrant / Vector Store
    # ------------------------------------------------------------------ #
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="financial_rag")

    # HNSW index params
    hnsw_m: int = Field(default=16)                  # connections per node
    hnsw_ef_construct: int = Field(default=200)      # build accuracy

    # ------------------------------------------------------------------ #
    # Chunking Hyperparameters
    # ------------------------------------------------------------------ #
    # Child chunks (used for retrieval / indexing)
    child_chunk_size: int = Field(default=350)        # tokens
    child_chunk_overlap: int = Field(default=50)      # tokens (~14% overlap)

    # Parent chunks (returned to LLM as context)
    parent_chunk_size: int = Field(default=1800)      # tokens
    parent_chunk_overlap: int = Field(default=100)    # tokens

    # Minimum chunk size — discard orphan fragments
    min_chunk_tokens: int = Field(default=80)

    # Font size threshold to detect headings in PyMuPDF
    heading_font_size_threshold: float = Field(default=12.0)

    # ------------------------------------------------------------------ #
    # Retrieval Hyperparameters
    # ------------------------------------------------------------------ #
    # How many candidates to retrieve at each stage
    dense_top_k: int = Field(default=30)             # dense retrieval candidates
    sparse_top_k: int = Field(default=30)            # BM25 candidates
    rrf_k: int = Field(default=60)                   # RRF constant (standard = 60)
    rerank_top_n: int = Field(default=5)             # final chunks after reranking

    # Query expansion: number of paraphrased queries to generate
    num_expanded_queries: int = Field(default=3)

    # Score threshold — discard chunks below this similarity
    score_threshold: float = Field(default=0.35)

    # ------------------------------------------------------------------ #
    # LLM Generation
    # ------------------------------------------------------------------ #
    llm_temperature: float = Field(default=0.1)      # low = more factual
    llm_max_tokens: int = Field(default=2048)
    llm_context_window: int = Field(default=8192)

    # ------------------------------------------------------------------ #
    # Embedding Batch Processing
    # ------------------------------------------------------------------ #
    embed_batch_size: int = Field(default=32)

    # ------------------------------------------------------------------ #
    # Metadata / Fiscal Year
    # ------------------------------------------------------------------ #
    # BSE India: Financial year = April to March
    # Reports filed Apr–Jun 2025 → FY2024-25
    default_fiscal_year: str = Field(default="FY2025")

    # Known company name → ticker mapping (for metadata tagging)
    company_ticker_map: dict = Field(default={
        "AIRTEL": "BHARTIARTL",
        "AXIS": "AXISBANK",
        "BAJAJ": "BAJFINANCE",
        "HCL": "HCLTECH",
        "HDFC": "HDFCBANK",
        "HU": "HINDUNILVR",
        "ICICI": "ICICIBANK",
        "INFOSYS": "INFY",
        "ITC": "ITC",
        "KOTAK": "KOTAKBANK",
        "KVB": "KARURVYSYA",
        "L&T": "LT",
        "MARUTI": "MARUTI",
        "MRF": "MRF",
        "ONGC": "ONGC",
        "REL": "RELIANCE",
        "SBI": "SBIN",
        "TCS": "TCS",
    })

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")

    def get_device(self) -> str:
        """Resolve 'auto' to actual device string."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.qdrant_path,
            self.bm25_index_path,
            self.processed_docs_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton settings instance — import this everywhere
settings = Settings()
