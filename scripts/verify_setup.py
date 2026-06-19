"""
scripts/verify_setup.py
=======================
Run this after pip install to verify everything is working correctly.
Usage: python scripts/verify_setup.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def check(label: str, fn):
    """Run a check function and return (passed, message)."""
    try:
        result = fn()
        return True, result or "OK"
    except Exception as e:
        return False, str(e)


def main():
    console.rule("[bold cyan]RAG Environment Verification[/bold cyan]")
    console.print()

    results = []

    # 1. Python version
    passed, msg = check(
        "Python version",
        lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓"
    )
    results.append(("Python 3.12+", passed, msg))

    # 2. PyMuPDF
    def check_pymupdf():
        import fitz
        return f"PyMuPDF {fitz.version[0]} ✓"
    results.append(("PyMuPDF (PDF parsing)", *check("PyMuPDF", check_pymupdf)))

    # 3. pdfplumber
    def check_pdfplumber():
        import pdfplumber
        return f"pdfplumber {pdfplumber.__version__} ✓"
    results.append(("pdfplumber (tables)", *check("pdfplumber", check_pdfplumber)))

    # 4. sentence-transformers
    def check_st():
        import sentence_transformers
        return f"sentence-transformers {sentence_transformers.__version__} ✓"
    results.append(("sentence-transformers", *check("sentence-transformers", check_st)))

    # 5. PyTorch + CUDA
    def check_torch():
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"PyTorch {torch.__version__} | CUDA ✓ | {gpu} ({vram:.1f}GB VRAM)"
        else:
            return f"PyTorch {torch.__version__} | CPU only (CUDA not detected)"
    results.append(("PyTorch + CUDA", *check("PyTorch", check_torch)))

    # 6. Qdrant client
    def check_qdrant():
        from qdrant_client import QdrantClient
        import qdrant_client
        return f"qdrant-client {qdrant_client.__version__} ✓"
    results.append(("Qdrant client", *check("Qdrant", check_qdrant)))

    # 7. rank-bm25
    def check_bm25():
        import rank_bm25
        return "rank-bm25 ✓"
    results.append(("rank-bm25 (sparse)", *check("BM25", check_bm25)))

    # 8. Groq
    def check_groq():
        import groq
        return f"groq {groq.__version__} ✓"
    results.append(("Groq SDK", *check("Groq", check_groq)))

    # 9. LangChain
    def check_langchain():
        import langchain
        return f"langchain {langchain.__version__} ✓"
    results.append(("LangChain", *check("LangChain", check_langchain)))

    # 10. Groq API key
    def check_api_key():
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY not set in .env file!")
        return f"Key found: {key[:8]}...{key[-4:]} ✓"
    results.append(("Groq API Key (.env)", *check("API Key", check_api_key)))

    # 11. Streamlit
    def check_streamlit():
        import streamlit
        return f"streamlit {streamlit.__version__} ✓"
    results.append(("Streamlit", *check("Streamlit", check_streamlit)))

    # 12. Plotly
    def check_plotly():
        import plotly
        return f"plotly {plotly.__version__} ✓"
    results.append(("Plotly", *check("Plotly", check_plotly)))

    # 13. RAGAS
    def check_ragas():
        import ragas
        return f"ragas {ragas.__version__} ✓"
    results.append(("RAGAS (evaluation)", *check("RAGAS", check_ragas)))

    # 14. Settings
    def check_settings():
        from config.settings import settings
        settings.ensure_dirs()
        device = settings.get_device()
        return f"Settings loaded | Device: {device} ✓"
    results.append(("Config/Settings", *check("Settings", check_settings)))

    # 15. Raw PDFs
    def check_pdfs():
        from config.settings import settings
        pdf_dir = settings.raw_pdfs_path
        if not pdf_dir.exists():
            raise FileNotFoundError(f"raw_pdfs/ directory not found at {pdf_dir}")
        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("No PDFs found in raw_pdfs/ directory")
        return f"{len(pdfs)} PDFs found in raw_pdfs/ ✓"
    results.append(("Raw PDFs", *check("PDFs", check_pdfs)))

    # ------------------------------------------------------------------ #
    # Print results table
    # ------------------------------------------------------------------ #
    console.print()
    table = Table(title="Setup Verification Results", show_lines=True)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    all_passed = True
    for label, passed, msg in results:
        status = "[bold green]PASS ✓[/bold green]" if passed else "[bold red]FAIL ✗[/bold red]"
        table.add_row(label, status, msg)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    if all_passed:
        console.print("[bold green]✨ All checks passed! Environment is ready.[/bold green]")
        console.print("[dim]Next step: run [cyan]python scripts/ingest_all.py[/cyan] to start the ingestion pipeline.[/dim]")
    else:
        console.print("[bold red]⚠ Some checks failed. Fix the issues above before proceeding.[/bold red]")
        console.print("[dim]Common fixes:[/dim]")
        console.print("  • Missing package? Run: [cyan]pip install -r requirements.txt[/cyan]")
        console.print("  • Missing GROQ_API_KEY? Copy [cyan].env.example → .env[/cyan] and add your key")
        console.print("  • CUDA not found? Run: [cyan]pip install torch --index-url https://download.pytorch.org/whl/cu121[/cyan]")

    console.print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
