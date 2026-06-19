"""
scripts/ingest_all.py
======================
Master ingestion script — run this ONCE to process all BSE PDFs.

What it does:
    1. Parses all 23 PDFs (text + tables + OCR fallback)
    2. Chunks them hierarchically (parent-child + atomic tables)
    3. Embeds all index chunks using BAAI/bge-large (GPU-accelerated)
    4. Stores in Qdrant HNSW vector index
    5. Builds BM25 sparse keyword index
    6. Prints a summary report

Estimated time on RTX 3050:
    - Parsing: ~3-5 minutes (23 PDFs)
    - Embedding: ~5-10 minutes (GPU accelerated)
    - Total: ~10-15 minutes

Run with:
    python scripts/ingest_all.py

To force re-process (ignore cache):
    python scripts/ingest_all.py --force
"""

import sys
import os
import argparse
import time

# FORCE UTF-8 output on Windows (fixes Rich Unicode encoding errors)
os.environ["PYTHONUTF8"] = "1"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: import config FIRST to set USE_TF=0 before any ML imports
from config.settings import settings

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import logger
from src.ingestion.pipeline import IngestionPipeline
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_index import BM25Index

console = Console(force_terminal=True)


def parse_args():
    parser = argparse.ArgumentParser(description="BSE Financial RAG — Ingestion Pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all PDFs (ignore cache)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop and recreate Qdrant collection (use when changing embedding model)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR for scanned pages (faster, may miss some content)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold cyan]BSE Financial RAG -- Ingestion Pipeline[/bold cyan]\n"
        "[dim]Production-grade PDF -> Vector DB pipeline[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Ensure all data directories exist
    settings.ensure_dirs()

    start_time = time.time()

    # ---------------------------------------------------------------- #
    # Step 1: Ingest PDFs
    # ---------------------------------------------------------------- #
    console.rule("[bold]Step 1/5: PDF Ingestion[/bold]")
    console.print(f"[dim]Processing PDFs from: {settings.raw_pdfs_path}[/dim]\n")

    pipeline = IngestionPipeline()
    documents = pipeline.run_all(force_reprocess=args.force)

    console.print(f"\n✅ Ingested [bold green]{len(documents)}[/bold green] documents")

    # Show ingestion summary table
    summary_table = Table(title="Ingested Documents", show_lines=True)
    summary_table.add_column("Company", style="cyan")
    summary_table.add_column("Fiscal Year", style="yellow")
    summary_table.add_column("Source File", style="dim")
    summary_table.add_column("Pages", justify="right")
    summary_table.add_column("Content Items", justify="right")

    for doc in documents:
        summary_table.add_row(
            doc.company,
            doc.fiscal_year,
            doc.source_file,
            str(doc.total_pages),
            str(len(doc.items)),
        )
    console.print(summary_table)
    console.print()

    # ---------------------------------------------------------------- #
    # Step 2: Chunk Documents
    # ---------------------------------------------------------------- #
    console.rule("[bold]Step 2/5: Hierarchical Chunking[/bold]")

    chunker = HierarchicalChunker()
    chunked_docs = chunker.chunk_documents(documents)

    # Collect all index chunks (child + atomic)
    all_index_chunks = []
    all_parent_chunks = []
    for cdoc in chunked_docs:
        all_index_chunks.extend(cdoc.all_index_chunks)
        all_parent_chunks.extend(cdoc.parent_chunks)

    console.print(f"\n✅ Chunking complete:")
    console.print(f"   Parent chunks : [bold]{len(all_parent_chunks)}[/bold]")
    console.print(f"   Index chunks  : [bold]{len(all_index_chunks)}[/bold] (child + atomic tables)")
    console.print()

    # ---------------------------------------------------------------- #
    # Step 3: Embed Index Chunks
    # ---------------------------------------------------------------- #
    console.rule("[bold]Step 3/5: Embedding (GPU: RTX 3050)[/bold]")
    console.print(f"[dim]Model: {settings.embedding_model} | Batch size: {settings.embed_batch_size}[/dim]\n")

    embedder = Embedder()

    texts_to_embed = [chunk.content for chunk in all_index_chunks]
    embeddings = embedder.embed_documents(texts_to_embed)

    console.print(f"\n✅ Embedded [bold green]{len(embeddings)}[/bold green] chunks | shape: {embeddings.shape}")
    console.print()

    # ---------------------------------------------------------------- #
    # Step 4: Store in Qdrant
    # ---------------------------------------------------------------- #
    console.rule("[bold]Step 4/5: Qdrant Vector Store[/bold]")

    qdrant = QdrantStore()
    qdrant.create_collection(recreate=args.recreate_index)

    # Also upsert parent chunks (for parent lookup during retrieval)
    # Parents are stored with chunk_level="parent" — not searched, just fetched
    console.print("[dim]Upserting parent chunks for context promotion...[/dim]")
    parent_texts = [chunk.content for chunk in all_parent_chunks]
    if parent_texts:
        parent_embeddings = embedder.embed_documents(parent_texts)
        qdrant.upsert_chunks(all_parent_chunks, parent_embeddings)

    console.print("[dim]Upserting index chunks (child + atomic)...[/dim]")
    qdrant.upsert_chunks(all_index_chunks, embeddings)

    info = qdrant.get_collection_info()
    console.print(f"\n✅ Qdrant store: [bold green]{info['vectors_count']}[/bold green] vectors indexed")
    console.print()

    # ---------------------------------------------------------------- #
    # Step 5: Build BM25 Index
    # ---------------------------------------------------------------- #
    console.rule("[bold]Step 5/5: BM25 Sparse Index[/bold]")

    bm25 = BM25Index()
    bm25.build(all_index_chunks)

    console.print(f"\n✅ BM25 index built over [bold green]{len(all_index_chunks)}[/bold green] chunks")
    console.print()

    # ---------------------------------------------------------------- #
    # Final Summary
    # ---------------------------------------------------------------- #
    elapsed = time.time() - start_time

    console.print(Panel(
        f"[bold green]Ingestion Complete![/bold green]\n\n"
        f"  Documents ingested : {len(documents)}\n"
        f"  Parent chunks      : {len(all_parent_chunks)}\n"
        f"  Index chunks       : {len(all_index_chunks)}\n"
        f"  Total vectors      : {info['vectors_count']}\n"
        f"  Time elapsed       : {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n"
        f"[dim]Next: run [cyan]streamlit run app/main.py[/cyan] to start the app![/dim]",
        title="Summary",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
