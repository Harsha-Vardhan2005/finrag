"""
scripts/build_bm25.py
======================
Standalone script to verify Qdrant is healthy and build the BM25 index.
Run this after ingestion is complete.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

from src.utils.logger import logger
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_index import BM25Index

# --- Step 1: Verify Qdrant ---
print("Checking Qdrant collection...")
qdrant = QdrantStore()
info = qdrant.get_collection_info()
print(f"Qdrant status  : {info['status']}")
print(f"Vectors stored : {info.get('vectors_count', info.get('points_count', '?'))}")

# --- Step 2: Scroll all index chunks from Qdrant to build BM25 ---
print("\nScrolling all index chunks from Qdrant for BM25 build...")

from qdrant_client.models import Filter, FieldCondition, MatchAny

# We need Chunk-like objects. Let's just rebuild from the JSON cache.
from src.ingestion.pipeline import IngestionPipeline
from src.chunking.hierarchical_chunker import HierarchicalChunker

pipeline = IngestionPipeline()
documents = pipeline.run_all(force_reprocess=False)   # loads from cache instantly

chunker = HierarchicalChunker()
chunked_docs = chunker.chunk_documents(documents)

all_index_chunks = []
for cdoc in chunked_docs:
    all_index_chunks.extend(cdoc.all_index_chunks)

print(f"Index chunks for BM25: {len(all_index_chunks)}")

# --- Step 3: Build BM25 ---
print("\nBuilding BM25 sparse index...")
bm25 = BM25Index()
bm25.build(all_index_chunks)

print(f"\nBM25 index built over {len(all_index_chunks)} chunks")
print("\nAll done! Run: streamlit run app/main.py")
