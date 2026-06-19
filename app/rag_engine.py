"""
app/rag_engine.py
==================
Shared cached resource loaders for all Streamlit pages.
Import from here instead of app.main to avoid page_config conflicts.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

import streamlit as st


@st.cache_resource(show_spinner="Loading embedding model (first time ~30s)...")
def load_embedder():
    from src.embeddings.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_retriever():
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.vectorstore.qdrant_store import QdrantStore
    from src.vectorstore.bm25_index import BM25Index
    qdrant = QdrantStore()
    bm25 = BM25Index()
    embedder = load_embedder()
    return HybridRetriever(qdrant_store=qdrant, bm25_index=bm25, embedder=embedder)


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker():
    from src.retrieval.reranker import Reranker
    return Reranker()


@st.cache_resource
def load_groq():
    from src.generation.groq_client import GroqClient
    return GroqClient()


@st.cache_resource
def load_comparator():
    from src.comparison.company_comparator import CompanyComparator
    return CompanyComparator(
        retriever=load_retriever(),
        reranker=load_reranker(),
        groq_client=load_groq(),
    )


@st.cache_resource
def load_metric_extractor():
    from src.metrics.metric_extractor import MetricExtractor
    return MetricExtractor(groq_client=load_groq())
