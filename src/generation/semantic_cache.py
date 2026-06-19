"""
src/generation/semantic_cache.py
=================================
In-memory Semantic Cache for RAG query results.
Uses cosine similarity over NumPy matrices to find semantically identical questions.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from src.utils.logger import logger

class SemanticCache:
    """
    Singleton Semantic Cache.
    Max 500 overall queries. FIFO eviction.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_size: int = 500, threshold: float = 0.95):
        if self._initialized:
            return
            
        self.max_size = max_size
        self.threshold = threshold
        
        # We store queries in a parallel array structure
        # embeddings: np.ndarray of shape (N, 768)
        self.embeddings: np.ndarray = np.empty((0, 768), dtype=np.float32)
        
        # payloads parallel to embeddings rows
        # [{"company": str, "fy": str, "text": str, "sources": list}, ...]
        self.payloads: List[Dict] = []
        
        self._initialized = True
        logger.info(f"Semantic Cache initialized (max_size={max_size}, threshold={threshold})")

    def find_match(self, query_emb: np.ndarray, company: str, fiscal_year: Optional[str]) -> Optional[Tuple[str, list]]:
        """
        Finds a cached response using cosine similarity.
        Query embedding must be L2 normalized (BGE embedder does this).
        """
        if len(self.payloads) == 0:
            return None

        # query_emb shape: (768,)
        # self.embeddings shape: (N, 768)
        # Cosine similarity is just the dot product since vectors are L2 normalized
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get the index of the highest similarity
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        
        if best_score >= self.threshold:
            # Check hard filters (company and FY must match exactly)
            p = self.payloads[best_idx]
            if p["company"] == company and p["fy"] == fiscal_year:
                logger.info(f"Semantic Cache HIT (score={best_score:.4f})")
                return p["text"], p["sources"]
                
        return None

    def add(self, query_emb: np.ndarray, company: str, fiscal_year: Optional[str], text: str, sources: list):
        """Adds a new response to the cache, evicting the oldest if full."""
        # Check size and apply FIFO eviction
        if len(self.payloads) >= self.max_size:
            # Remove oldest (index 0)
            self.embeddings = self.embeddings[1:]
            self.payloads.pop(0)
            
        # Append new embedding
        query_emb_2d = query_emb.reshape(1, -1)
        if self.embeddings.shape[0] == 0:
            self.embeddings = query_emb_2d
        else:
            self.embeddings = np.vstack([self.embeddings, query_emb_2d])
            
        # Append payload
        self.payloads.append({
            "company": company,
            "fy": fiscal_year,
            "text": text,
            "sources": sources
        })
        logger.info(f"Added to Semantic Cache (size={len(self.payloads)}/{self.max_size})")
