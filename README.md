---
title: FinRAG
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

<div align="center">
  
# FinRAG — Production Financial AI Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud_Vector_DB-fd1e49?style=for-the-badge&logo=qdrant)
![Llama3](https://img.shields.io/badge/Llama_3.3-70B-orange?style=for-the-badge&logo=meta)
![Groq](https://img.shields.io/badge/Powered_by-Groq-black?style=for-the-badge&logo=groq)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/Harshavard21/FinRAG)

### 🔴 [Live Production Demo on Hugging Face Spaces](https://harshavard21-finrag.hf.space/) 🔴

> **An enterprise-grade Retrieval-Augmented Generation (RAG) system built over the official BSE Annual Reports of 17 major Indian companies.**
> FinRAG leverages advanced hybrid retrieval, cross-encoder reranking, and semantic caching to deliver lightning-fast, 100% grounded financial insights.

---
</div>

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Core Features & UI](#-core-features--ui)
- [Production RAG Architecture](#-production-rag-architecture)
- [Tech Stack](#-tech-stack)
- [Companies Covered](#-companies-covered)
- [Local Setup & Running](#-local-setup--running)

---

## 🎯 Project Overview

FinRAG solves the hallucination problem in financial AI. It processes massive, complex PDF annual reports and transforms them into an interactive, highly-accurate AI assistant. Whether you need deep-dive qualitative analysis or exact quantitative metrics, every single claim the AI makes is backed by a direct, clickable citation linking directly to the source page of the official financial report.

---

## 🚀 Core Features & UI

### Main Interface
![FinRAG Main UI](docs/main_ui.png)

### Interactive Chat (Grounded Q&A)
Ask natural language questions about any company's financial performance. Features a **Semantic Cache** for blazing-fast 50ms responses on repeated/similar queries, and clickable source badges that instantly open the exact PDF page where the AI found the data.
![Chat UI](docs/chat_screenshot.png)

#### 💡 Natural Language Explanations
![Natural Language Explanation](docs/chat_screenshot2.png)

### Automated KPI Dashboard
Automatically extracts and displays key financial metrics (Revenue, Net Profit, EPS, ROE, NPA) into a beautiful, color-coded dashboard. Includes dynamically split Plotly charts (P&L vs Balance Sheet) that accurately represent data magnitude.
![KPI Dashboard](docs/dashboard_screenshot.png)

### Cross-Document Compare Mode
A powerhouse analytical workspace capable of running parallel retrievals across different documents. You can instantly compare multiple companies (e.g., "HDFC vs ICICI Gross NPA") or track Year-over-Year trends for a single company (e.g., "TCS FY24 vs FY25 Revenue").
![Compare View](docs/compare_screenshot.png)

---

## 🧠 Production RAG Architecture

FinRAG implements state-of-the-art information retrieval techniques to ensure enterprise-grade accuracy.

```mermaid
flowchart TD
    A[BSE PDFs] -->|PyMuPDF + pdfplumber| B(Smart Ingestion & OCR)
    B --> C(Hierarchical Chunking)
    C -->|Parent/Child Nodes| D{Embedding & Indexing}
    D -->|Dense Vectors| E[(Qdrant HNSW)]
    D -->|Sparse Terms| F[(BM25 Index)]
    
    G[User Query] --> H(Query Expansion)
    H --> I[Hybrid Retrieval]
    E --> I
    F --> I
    I -->|Rank Fusion RRF| J(BGE Cross-Encoder Reranker)
    J -->|Top-K Chunks| K(Llama 3.3 70B via Groq)
    K --> L[Streaming Response with Citations]
    
    G -.->|If >95% Match| M(In-Memory Semantic Cache)
    M -.->|Instant 50ms Hit| L
```

### Advanced Concepts Used:
- **Hierarchical Chunking:** Splits documents into small chunks for precise searching, but passes the larger surrounding "parent" context to the LLM to prevent data fragmentation.
- **Hybrid Search (Dense + Sparse):** Combines Semantic vector search (Qdrant) with exact keyword matching (BM25) and fuses the scores using **Reciprocal Rank Fusion (RRF)**.
- **Cross-Encoder Reranking:** The initial search pulls 30-50 candidates. A powerful `BAAI/bge-reranker-base` model then heavily scores and re-orders them to find the absolute top 3-5 most relevant chunks.
- **Semantic Caching:** A NumPy-powered in-memory vector cache that short-circuits the entire pipeline if a user asks a semantically similar question, saving expensive API tokens.

---

## 🛠️ Tech Stack

- **Backend / API:** Python 3.12, FastAPI, Uvicorn
- **Frontend UI:** Vanilla JS, HTML, CSS (Custom Glassmorphism UI)
- **Vector Database:** Qdrant (Local via Docker)
- **Embeddings:** `BAAI/bge-large-en-v1.5`
- **Reranker:** `BAAI/bge-reranker-base`
- **LLM Inference:** Llama 3.3 70B (Powered by Groq LPUs for ultra-low latency)

---

## 🏢 Companies Covered

Data includes **FY2024–FY2025** BSE Annual Reports for 17 major entities across IT, Banking, FMCG, and Infrastructure:

*Airtel, Axis Bank, Bajaj Finance, HCL, HDFC Bank, HUL, ICICI Bank, Infosys, ITC, Kotak Mahindra Bank, Karur Vysya Bank, L&T, Maruti Suzuki, MRF, ONGC, Reliance Industries, SBI, TCS.*

---

## ⚙️ Local Setup & Running

1. **Clone & Install Dependencies**
```bash
git clone https://github.com/yourusername/finrag.git
cd finrag
pip install -r requirements.txt
```

2. **Set Environment Variables**
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
QDRANT_URL=https://your-cluster-url.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key
```

3. **Run the Application Locally**
Since the vectors are hosted on Qdrant Cloud, no local Docker container is needed for the database!
```bash
streamlit run app/main.py
```

4. **Open the UI**
Navigate to `http://localhost:8501/` in your browser.

## ☁️ Cloud Deployment (Hugging Face Spaces)
This application is fully containerized and currently deployed on **Hugging Face Spaces** using Docker.
- **Git LFS**: Used to efficiently store and serve the 23 heavy PDF Annual Reports without bloating the Git history.
- **Secrets Management**: API keys (Groq & Qdrant) are securely injected into the Docker container via HF Secrets.
- **CI/CD**: Pushing to the HF remote triggers an automatic Docker rebuild and zero-downtime deployment.
