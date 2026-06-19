"""
src/embeddings/embedder.py
===========================
Embedding engine using BAAI/bge-large-en-v1.5.

Why BGE-large?
- Free, runs locally on your RTX 3050
- Top of MTEB leaderboard for retrieval tasks (among free models)
- 768-dimensional output — good balance of quality vs. memory
- Trained with specific instruction prefix for queries vs. documents

BGE-specific usage:
    Documents  → embed as-is (no prefix)
    Queries    → add "Represent this sentence: " prefix
    This asymmetric approach improves retrieval accuracy significantly.

L2 Normalization:
    All embeddings are L2-normalized before storage.
    This makes cosine similarity = dot product → faster ANN search in Qdrant.
"""

from __future__ import annotations

import numpy as np
from typing import Union

from tqdm import tqdm

# config import first — sets USE_TF=0 before sentence_transformers loads
from config.settings import settings
from src.utils.logger import logger

# Now safe to import
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Singleton-pattern embedding engine.
    Loads BAAI/bge-large-en-v1.5 once and reuses it.

    GPU-accelerated on RTX 3050 automatically (via settings.get_device()).
    """

    _instance: "Embedder | None" = None

    def __new__(cls):
        """Enforce singleton — only load the model once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Load the embedding model onto GPU/CPU."""
        model_name = settings.embedding_model
        device = settings.get_device()

        logger.info(f"Loading embedding model: {model_name} → device: {device}")
        logger.info("(First load downloads ~1.3GB — subsequent loads use cache)")

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512    # BGE-large max input length
        self.dim = settings.embedding_dim  # 768

        logger.info(f"Embedding model loaded | dim={self.dim} | device={device}")

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document chunks (corpus side).
        No instruction prefix needed for documents with BGE.

        Args:
            texts: List of text strings to embed

        Returns:
            np.ndarray of shape (len(texts), 768), L2-normalized
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} documents in batches of {settings.embed_batch_size}...")

        embeddings = []
        batch_size = settings.embed_batch_size

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
            batch = texts[i: i + batch_size]
            batch_emb = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # L2 normalize in-place
                convert_to_numpy=True,
            )
            embeddings.append(batch_emb)

        result = np.vstack(embeddings)
        logger.info(f"Embedding complete: shape={result.shape}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query (query side).
        BGE models benefit from an instruction prefix on queries.

        Args:
            query: The user's question

        Returns:
            np.ndarray of shape (768,), L2-normalized
        """
        # BGE instruction prefix — improves retrieval accuracy
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """
        Embed multiple queries (for multi-query retrieval expansion).

        Args:
            queries: List of question strings

        Returns:
            np.ndarray of shape (len(queries), 768)
        """
        prefixed = [
            f"Represent this sentence for searching relevant passages: {q}"
            for q in queries
        ]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.dim
