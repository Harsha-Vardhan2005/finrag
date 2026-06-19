"""
app/api/server.py
==================
FastAPI backend — serves the RAG pipeline via REST + SSE streaming.
Run with: uvicorn app.api.server:app --port 8000 --reload
"""

import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["PYTHONUTF8"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings

# ── lazy-loaded singletons ──────────────────────────────────────────
_retriever = None
_reranker  = None
_groq      = None
_extractor = None
_comparator = None
_embedder  = None
_semantic_cache = None
_executor  = ThreadPoolExecutor(max_workers=4)

def get_embedder():
    global _embedder
    if _embedder is None:
        from src.embeddings.embedder import Embedder
        _embedder = Embedder()
    return _embedder

def get_retriever():
    global _retriever
    if _retriever is None:
        from src.vectorstore.qdrant_store import QdrantStore
        from src.vectorstore.bm25_index import BM25Index
        from src.retrieval.hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever(QdrantStore(), BM25Index(), get_embedder())
    return _retriever

def get_reranker():
    global _reranker
    if _reranker is None:
        from src.retrieval.reranker import Reranker
        _reranker = Reranker()
    return _reranker

def get_groq():
    global _groq
    if _groq is None:
        from src.generation.groq_client import GroqClient
        _groq = GroqClient()
    return _groq

def get_extractor():
    global _extractor
    if _extractor is None:
        from src.metrics.metric_extractor import MetricExtractor
        _extractor = MetricExtractor(groq_client=get_groq())
    return _extractor

def get_semantic_cache():
    global _semantic_cache
    if _semantic_cache is None:
        from src.generation.semantic_cache import SemanticCache
        _semantic_cache = SemanticCache(max_size=500, threshold=0.95)
    return _semantic_cache


# ── app ─────────────────────────────────────────────────────────────
app = FastAPI(title="FinRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (HTML/CSS/JS)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount raw_pdfs directory for serving PDFs
PDF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw_pdfs"))
if os.path.exists(PDF_DIR):
    app.mount("/raw_pdfs", StaticFiles(directory=PDF_DIR), name="raw_pdfs")


# ── request models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    company: str
    fiscal_year: Optional[str] = "FY2025"
    conversation_id: Optional[str] = None

class CompareRequest(BaseModel):
    query: str
    companies: list[str]
    fiscal_year: Optional[str] = "FY2025"
    conversation_id: Optional[str] = None

class DashboardRequest(BaseModel):
    company: str
    fiscal_year: Optional[str] = "FY2025"

class ConversationCreate(BaseModel):
    mode: str = "chat"

# ── routes ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/companies")
async def companies():
    return {"companies": list(settings.company_ticker_map.keys()),
            "tickers": settings.company_ticker_map}


# ── Small-talk / greeting detection ─────────────────────────────────
_SMALL_TALK = {
    'hi','hello','hey','hii','heya','howdy','sup','yo',
    'bye','goodbye','see you','take care','cya',
    'thanks','thank you','thank you so much','thx','ty',
    'ok','okay','got it','understood','sure','alright',
    'how are you','how r u','whats up',"what's up",
    'who are you','what are you','what can you do','help me','help',
    'good morning','good afternoon','good evening','good night',
    'nice','cool','great','awesome','wow','amazing',
}

SMALL_TALK_SYSTEM = (
    "You are FinRAG, an AI assistant specialised in Indian company financials. "
    "Respond briefly and warmly to greetings or small talk. "
    "Let the user know you can answer questions about major BSE-listed companies "
    "(TCS, HDFC Bank, Infosys, Reliance, SBI etc.) using their official annual reports. "
    "Keep it to 1-3 sentences."
)

def is_small_talk(query: str) -> bool:
    q = query.lower().strip().rstrip('?!.,')
    if q in _SMALL_TALK:
        return True
    words = q.split()
    return len(words) <= 4 and any(w in _SMALL_TALK for w in words)


# ── Chat (SSE streaming) ─────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    from src.generation.prompts import build_qa_prompt
    
    # 1. Check Semantic Cache first
    semantic_cache = get_semantic_cache()
    embedder = get_embedder()
    
    # Run embedding in thread pool to not block async loop
    loop = asyncio.get_event_loop()
    query_emb = await loop.run_in_executor(_executor, embedder.embed_query, req.query)
    
    fy = None if req.fiscal_year == "All" else req.fiscal_year
    cached_result = semantic_cache.find_match(query_emb, req.company, fy)
    
    if cached_result:
        cached_text, cached_sources = cached_result
        async def cached_stream():
            yield f"data: {json.dumps({'type': 'chunk', 'text': cached_text})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached_sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(cached_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _run():
        if is_small_talk(req.query):
            groq = get_groq()
            return groq.generate_stream(SMALL_TALK_SYSTEM, req.query), []

        retriever = get_retriever()
        reranker  = get_reranker()
        groq      = get_groq()
        fy = None if req.fiscal_year == "All" else req.fiscal_year

        candidates = retriever.retrieve(
            query=req.query, top_k=50,
            company_filter=req.company,
            fiscal_year_filter=fy,
            expand_query=True, promote_to_parent=True,
        )
        if not candidates:
            return None, []

        reranked = reranker.rerank(query=req.query, candidates=candidates, top_n=5)
        system_p, user_p = build_qa_prompt(
            query=req.query, results=reranked,
            company_context=f"{req.company} ({settings.company_ticker_map.get(req.company, '')})",
        )
        sources = [
            {"company": c.company, "fiscal_year": c.fiscal_year,
             "page": c.page_number, "is_table": c.content_type == "table",
             "section": c.section or "Financial Data", "file": c.source_file}
            for c in reranked
        ]
        return groq.generate_stream(system_p, user_p), sources

    loop = asyncio.get_event_loop()
    stream_gen, sources = await loop.run_in_executor(_executor, _run)

    async def event_stream():
        if stream_gen is None:
            yield f"data: {json.dumps({'type': 'error', 'text': 'No relevant documents found.'})}\n\n"
            return

        # Stream text chunks
        full_text = ""
        def _collect():
            return list(stream_gen)

        chunks = await loop.run_in_executor(_executor, _collect)
        for chunk in chunks:
            full_text += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

        # Cache the result for future semantically similar queries
        semantic_cache.add(query_emb, req.company, fy, full_text, sources)
        
        # Send sources after text
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Dashboard (JSON) ────────────────────────────────────────────────
@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    def _run():
        retriever = get_retriever()
        reranker  = get_reranker()
        extractor = get_extractor()
        fy = None if req.fiscal_year == "All" else req.fiscal_year

        all_c = []
        for q in ["revenue net profit financial performance", "balance sheet assets equity"]:
            all_c.extend(retriever.retrieve(query=q, top_k=20,
                company_filter=req.company, fiscal_year_filter=fy))
        seen = set(); unique = []
        for c in all_c:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id); unique.append(c)

        reranked = reranker.rerank(
            query="financial metrics revenue profit assets equity EPS",
            candidates=unique, top_n=4,
        )
        return extractor.extract(req.company, reranked)

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _run)
    return JSONResponse(data)


# ── Compare (SSE streaming) ─────────────────────────────────────────
@app.post("/api/compare")
async def compare(req: CompareRequest):
    def _run():
        retriever = get_retriever()
        reranker  = get_reranker()
        groq      = get_groq()
        from src.generation.prompts import build_comparison_prompt
        import re

        # Detect cross-year comparisons. If multiple years are mentioned, search across all years
        years_mentioned = set(re.findall(r'202\d', req.query))
        if len(years_mentioned) > 1:
            fy = None
        else:
            fy = None if req.fiscal_year == "All" else req.fiscal_year
            
        company_results = {}
        top_n = max(1, 6 // len(req.companies)) if req.companies else 3
        for company in req.companies:
            cands = retriever.retrieve(
                query=req.query, top_k=30,
                company_filter=company, fiscal_year_filter=fy,
            )
            if cands:
                company_results[company] = reranker.rerank(
                    query=req.query, candidates=cands, top_n=top_n,
                )

        if not company_results:
            return None, {}

        system_p, user_p = build_comparison_prompt(
            query=req.query,
            company_results=company_results,
        )
        return groq.generate_stream(system_p, user_p), company_results

    loop = asyncio.get_event_loop()
    stream_gen, company_results = await loop.run_in_executor(_executor, _run)

    async def event_stream():
        if stream_gen is None:
            yield f"data: {json.dumps({'type': 'error', 'text': 'No data found.'})}\n\n"
            return

        def _collect():
            return list(stream_gen)

        chunks = await loop.run_in_executor(_executor, _collect)
        for chunk in chunks:
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

        sources = []
        for company, results in company_results.items():
            for r in results:
                sources.append({"company": company, "page": r.page_number,
                                "is_table": False, "section": r.section or ""})
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Conversation history ─────────────────────────────────────────────
@app.get("/api/conversations")
async def list_convs():
    from app.db.chat_store import list_conversations, group_conversations_by_date
    convs = list_conversations(40)
    grouped = group_conversations_by_date(convs)
    return {"grouped": grouped}

@app.post("/api/conversations")
async def create_conv(body: ConversationCreate):
    from app.db.chat_store import new_conversation
    cid = new_conversation(body.mode)
    return {"id": cid}

@app.get("/api/conversations/{cid}/messages")
async def get_conv_messages(cid: str):
    from app.db.chat_store import get_messages
    msgs = get_messages(cid)
    return {"messages": msgs}

@app.delete("/api/conversations/{cid}")
async def delete_conv(cid: str):
    from app.db.chat_store import delete_conversation
    delete_conversation(cid)
    return {"ok": True}

@app.post("/api/conversations/{cid}/messages")
async def save_msg(cid: str, body: dict):
    from app.db.chat_store import add_message, update_conversation_title
    add_message(cid, body["role"], body["content"], body.get("metadata"))
    if body["role"] == "user":
        update_conversation_title(cid, body["content"])
    return {"ok": True}
