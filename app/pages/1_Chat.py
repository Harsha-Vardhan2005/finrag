"""
app/pages/1_Chat.py
====================
Single company Q&A page with streaming answers and citation display.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

import streamlit as st

st.set_page_config(page_title="Q&A | FinRAG", page_icon="💬", layout="wide")

# Shared resource loaders (cached across all pages)
from app.rag_engine import load_retriever, load_reranker, load_groq

# Apply same CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg-card: rgba(255,255,255,0.04);
    --accent-cyan: #06b6d4;
    --accent-blue: #3b82f6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: rgba(255,255,255,0.08);
}
.stApp { background: linear-gradient(135deg, #0a0e1a, #0f1628); font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
.citation-chip {
    display: inline-block; background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3); border-radius: 20px;
    padding: 0.2rem 0.7rem; font-size: 0.72rem; color: #93c5fd; margin: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}
.fin-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ---- Header ----
st.markdown("""
<div style="padding:1.5rem 0 1rem 0;">
    <h1 style="font-size:1.8rem;font-weight:700;margin:0;
               background:linear-gradient(135deg,#3b82f6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        💬 Company Q&A
    </h1>
    <p style="color:#64748b;font-size:0.9rem;margin-top:0.3rem;">
        Ask anything about a company's financials. Answers are grounded in BSE annual reports.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Controls ----
col_company, col_fy = st.columns([2, 1])

with col_company:
    companies = list(settings.company_ticker_map.keys())
    selected_company = st.selectbox(
        "Select Company",
        options=companies,
        index=companies.index("TCS") if "TCS" in companies else 0,
        key="qa_company",
    )

with col_fy:
    fy_options = ["All Years", "FY2025", "FY2024"]
    selected_fy = st.selectbox("Fiscal Year", fy_options, key="qa_fy")
    fy_filter = None if selected_fy == "All Years" else selected_fy

# ---- Sample questions ----
sample_questions = {
    "TCS": [
        "What was TCS's total revenue and net profit for FY2025?",
        "What are TCS's key business segments and their revenue contributions?",
        "What is TCS's headcount and employee attrition rate?",
    ],
    "HDFC": [
        "What is HDFC Bank's gross NPA and net NPA ratio?",
        "What was HDFC Bank's net interest income for FY2025?",
        "What is HDFC Bank's CASA ratio?",
    ],
    "INFOSYS": [
        "What was Infosys's operating margin for FY2025?",
        "What are Infosys's key geographies and their revenue split?",
        "What is Infosys's guidance for the next fiscal year?",
    ],
}

if selected_company in sample_questions:
    with st.expander("Sample questions for " + selected_company, expanded=False):
        for q in sample_questions[selected_company]:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state["qa_prefill"] = q

# ---- Chat Input ----
default_q = st.session_state.pop("qa_prefill", "")
user_query = st.text_area(
    "Your question",
    value=default_q,
    placeholder="e.g. What was the company's net profit margin for FY2025?",
    height=80,
    key="qa_input",
)

col_btn, col_opts = st.columns([1, 3])
with col_btn:
    ask_btn = st.button("Ask Question", type="primary", key="qa_ask")
with col_opts:
    show_chunks = st.checkbox("Show retrieved chunks", value=False, key="qa_show_chunks")

# ---- Answer ----
if ask_btn and user_query.strip():
    retriever = load_retriever()
    llm = load_groq()

    from src.generation.prompts import build_qa_prompt

    # Stage 1: Hybrid retrieval (fast ~1-2s)
    with st.spinner(f"Step 1/3 — Searching {selected_company} documents (hybrid dense+sparse)..."):
        candidates = retriever.retrieve(
            query=user_query,
            top_k=50,
            company_filter=selected_company,
            fiscal_year_filter=fy_filter,
            expand_query=True,
            promote_to_parent=True,
        )

    if not candidates:
        st.warning("No relevant documents found. Try a different question or company.")
        st.stop()

    # Stage 2: Reranking (first time: downloads ~550MB model, ~2-5 min; subsequent: ~3s)
    reranker = load_reranker()
    with st.spinner("Step 2/3 — Reranking with cross-encoder (first run downloads BGE model ~550MB)..."):
        reranked = reranker.rerank(
            query=user_query,
            candidates=candidates,
            top_n=5,
        )

    if not reranked:
        st.warning("No relevant documents found. Try a different question or company.")
        st.stop()

    # Build prompt and stream answer
    system_prompt, user_message = build_qa_prompt(
        query=user_query,
        results=reranked,
        company_context=f"{selected_company} ({settings.company_ticker_map.get(selected_company, '')})",
    )

    # Stage 3: LLM generation via Groq (streaming)
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;">
        Step 3/3 — Generating answer from {len(reranked)} chunks | {selected_company} {fy_filter or 'all years'}
    </div>
    """, unsafe_allow_html=True)

    # Stream the answer
    with st.chat_message("assistant", avatar="📈"):
        answer = st.write_stream(llm.generate_stream(system_prompt, user_message))

    # Citations
    st.markdown("**Sources:**")
    citation_html = ""
    for chunk in reranked:
        label = f"{chunk.company} | {chunk.fiscal_year} | Pg.{chunk.page_number}"
        if chunk.content_type == "table":
            label += " [TABLE]"
        citation_html += f'<span class="citation-chip">{label}</span>'
    st.markdown(citation_html, unsafe_allow_html=True)

    # Retrieved chunks expander
    if show_chunks:
        with st.expander(f"Retrieved chunks ({len(reranked)})", expanded=False):
            for i, chunk in enumerate(reranked, 1):
                st.markdown(f"""
                <div class="fin-card">
                    <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.4rem;">
                        Chunk {i} | {chunk.company} | {chunk.section} | Page {chunk.page_number} |
                        Score: {chunk.rrf_score:.3f} | Type: {chunk.content_type}
                    </div>
                    <div style="font-size:0.82rem;color:#cbd5e1;white-space:pre-wrap;">{chunk.content[:600]}{"..." if len(chunk.content) > 600 else ""}</div>
                </div>
                """, unsafe_allow_html=True)

elif ask_btn:
    st.warning("Please enter a question.")
