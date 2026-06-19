"""
app/main.py  —  FinRAG  |  Production Financial Intelligence UI
================================================================
Single-page Streamlit app. No sub-pages.

Layout:
    Sidebar  : logo · new chat · conversation history · quick actions
    Main     : mode pills · chat thread · company context bar · chat input
"""

import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUTF8"] = "1"

from config.settings import settings

import streamlit as st
import plotly.graph_objects as go

# ── page config (MUST be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="FinRAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports ──────────────────────────────────────────────────────────
from app.rag_engine import (
    load_embedder, load_retriever, load_reranker,
    load_groq, load_comparator, load_metric_extractor,
)
from app.db.chat_store import (
    new_conversation, list_conversations, get_messages,
    add_message, update_conversation_title,
    delete_conversation, group_conversations_by_date,
)

# ════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #0d0f17;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0c13 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Sidebar collapse button ─────────────────────────────────── */
[data-testid="collapsedControl"] {
    color: #64748b !important;
    background: #0a0c13 !important;
}

/* ── Remove default padding from main content ────────────────── */
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
}

/* ── Chat messages ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
}

/* ── User bubble ─────────────────────────────────────────────── */
[data-testid="stChatMessage"][data-testid*="user"] {
    flex-direction: row-reverse !important;
}

/* ── Chat input ──────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #161929 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* ── Selectbox / radio ───────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #161929 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stRadio > div { gap: 0.4rem; }
.stRadio label { color: #94a3b8 !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.15s !important;
    border: none !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* ── Expander ────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-top: none !important;
}

/* ── Divider ─────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Plotly charts ───────────────────────────────────────────── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Custom components ───────────────────────────────────────── */
.fin-brand {
    padding: 1.2rem 1rem 0.8rem 1rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.fin-brand-icon { font-size: 1.4rem; }
.fin-brand-name {
    font-size: 1.1rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.fin-brand-sub {
    font-size: 0.65rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: -0.15rem;
}

.history-group-label {
    font-size: 0.68rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.08em;
    padding: 0.6rem 0.8rem 0.3rem 0.8rem;
}
.history-item {
    padding: 0.45rem 0.8rem; border-radius: 7px; cursor: pointer;
    font-size: 0.82rem; color: #94a3b8; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
    transition: background 0.1s, color 0.1s;
}
.history-item:hover { background: rgba(255,255,255,0.05); color: #e2e8f0; }
.history-item.active { background: rgba(59,130,246,0.12); color: #93c5fd; }

.mode-bar {
    display: flex; gap: 0.3rem; padding: 0.8rem 1.2rem 0;
}
.mode-pill {
    padding: 0.35rem 1rem; border-radius: 20px; font-size: 0.8rem;
    font-weight: 500; cursor: pointer; border: 1px solid rgba(255,255,255,0.08);
    color: #64748b; background: transparent; transition: all 0.15s;
}
.mode-pill.active {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.1));
    border-color: rgba(59,130,246,0.4); color: #93c5fd;
}

.context-bar {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 1.2rem; flex-wrap: wrap;
}
.ctx-tag {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px; padding: 0.2rem 0.65rem; font-size: 0.75rem;
    color: #93c5fd; font-family: 'JetBrains Mono', monospace;
}
.ctx-hint { font-size: 0.72rem; color: #475569; }

.kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem;
    margin: 0.8rem 0;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.85rem 1rem; text-align: center;
}
.kpi-card.has-value { border-color: rgba(59,130,246,0.2); }
.kpi-val {
    font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    color: #06b6d4; line-height: 1.2;
}
.kpi-val.na { color: #334155; font-size: 0.9rem; }
.kpi-lbl { font-size: 0.68rem; color: #64748b; margin-top: 0.25rem;
    text-transform: uppercase; letter-spacing: 0.05em; }

.source-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.6rem; }
.chip {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 5px; padding: 0.15rem 0.5rem; font-size: 0.68rem;
    color: #64748b; font-family: 'JetBrains Mono', monospace;
}
.chip.table-chip { border-color: rgba(16,185,129,0.3); color: #6ee7b7; }

.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 60vh; gap: 1.2rem; text-align: center;
    padding: 2rem;
}
.welcome-logo { font-size: 3rem; }
.welcome-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.welcome-sub { font-size: 1rem; color: #64748b; max-width: 460px; line-height: 1.6; }
.suggestion-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem; max-width: 560px; width: 100%;
}
.suggestion-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.82rem;
    color: #94a3b8; cursor: pointer; text-align: left;
    transition: all 0.15s;
}
.suggestion-card:hover {
    background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.3);
    color: #e2e8f0;
}
.suggestion-icon { font-size: 1rem; margin-bottom: 0.3rem; display: block; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Session State Init
# ════════════════════════════════════════════════════════════════════
COMPANIES = list(settings.company_ticker_map.keys())

def _init_state():
    defaults = {
        "conversation_id": None,
        "messages": [],           # list of dicts: role, content, metadata
        "mode": "chat",           # "chat" | "compare" | "dashboard"
        "company": COMPANIES[0],  # primary company
        "companies_compare": [],  # for compare mode
        "fy": "FY2025",
        "pending_suggestion": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def start_new_conversation():
    cid = new_conversation(mode=st.session_state.mode)
    st.session_state.conversation_id = cid
    st.session_state.messages = []


def load_conversation(cid: str):
    st.session_state.conversation_id = cid
    st.session_state.messages = get_messages(cid)


def save_message(role: str, content: str, metadata: dict = None):
    cid = st.session_state.conversation_id
    if not cid:
        cid = new_conversation()
        st.session_state.conversation_id = cid
    add_message(cid, role, content, metadata)
    # Set title from first user message
    if role == "user" and len(st.session_state.messages) == 0:
        update_conversation_title(cid, content)


def parse_at_mentions(text: str) -> tuple[str, list[str]]:
    """Extract @COMPANY mentions; return (cleaned_text, [companies])."""
    found = []
    cleaned = text
    for c in COMPANIES:
        if f"@{c}" in text or f"@{c.lower()}" in text.lower():
            found.append(c)
            cleaned = re.sub(f"@{re.escape(c)}", c, cleaned, flags=re.IGNORECASE)
    return cleaned.strip(), found


def detect_intent(query: str, mentioned: list[str]) -> str:
    """Auto-detect mode from query text."""
    ql = query.lower()
    if any(k in ql for k in ["dashboard", "kpi", "extract metrics", "show metrics", "key metrics"]):
        return "dashboard"
    if len(mentioned) > 1 or any(k in ql for k in ["compare", " vs ", "versus", "better than", "difference between"]):
        return "compare"
    return "chat"


def fmt_crore(v):
    if v is None: return "N/A"
    if v >= 100000: return f"₹{v/100000:.1f}L Cr"
    return f"₹{v:,.0f} Cr"

def fmt_pct(v):
    return f"{v:.2f}%" if v is not None else "N/A"

def fmt_inr(v):
    return f"₹{v:.2f}" if v is not None else "N/A"


# ════════════════════════════════════════════════════════════════════
# Render helpers
# ════════════════════════════════════════════════════════════════════

def render_source_chips(sources: list[dict]):
    if not sources:
        return
    chips_html = '<div class="source-chips">'
    for s in sources:
        cls = "chip table-chip" if s.get("is_table") else "chip"
        chips_html += f'<span class="{cls}">{s["label"]}</span>'
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


def render_kpi_cards(metrics: dict):
    """Render 2 rows of KPI cards from extracted metrics dict."""
    rows = [
        [
            ("Revenue", fmt_crore(metrics.get("revenue_crore")), metrics.get("revenue_crore") is not None),
            ("Net Profit", fmt_crore(metrics.get("net_profit_crore")), metrics.get("net_profit_crore") is not None),
            ("Net Margin", fmt_pct(metrics.get("net_profit_margin_pct")), metrics.get("net_profit_margin_pct") is not None),
            ("EBITDA Margin", fmt_pct(metrics.get("ebitda_margin_pct")), metrics.get("ebitda_margin_pct") is not None),
        ],
        [
            ("Total Assets", fmt_crore(metrics.get("total_assets_crore")), metrics.get("total_assets_crore") is not None),
            ("Equity", fmt_crore(metrics.get("equity_crore")), metrics.get("equity_crore") is not None),
            ("EPS", fmt_inr(metrics.get("eps_inr")), metrics.get("eps_inr") is not None),
            ("ROE", fmt_pct(metrics.get("roe_pct")), metrics.get("roe_pct") is not None),
        ],
    ]
    banking_row = None
    if any(metrics.get(k) is not None for k in ["npa_gross_pct", "npa_net_pct", "nim_pct"]):
        banking_row = [
            ("Gross NPA", fmt_pct(metrics.get("npa_gross_pct")), metrics.get("npa_gross_pct") is not None),
            ("Net NPA", fmt_pct(metrics.get("npa_net_pct")), metrics.get("npa_net_pct") is not None),
            ("NIM", fmt_pct(metrics.get("nim_pct")), metrics.get("nim_pct") is not None),
            ("EBITDA", fmt_crore(metrics.get("ebitda_crore")), metrics.get("ebitda_crore") is not None),
        ]

    for row in ([*rows] + ([banking_row] if banking_row else [])):
        cols = st.columns(4)
        for col, (label, value, has_val) in zip(cols, row):
            card_cls = "kpi-card has-value" if has_val else "kpi-card"
            val_cls = "kpi-val" if has_val else "kpi-val na"
            col.markdown(
                f'<div class="{card_cls}"><div class="{val_cls}">{value}</div>'
                f'<div class="kpi-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )


def render_kpi_chart(metrics: dict, company: str):
    chart_map = {
        "Revenue": metrics.get("revenue_crore"),
        "Net Profit": metrics.get("net_profit_crore"),
        "EBITDA": metrics.get("ebitda_crore"),
        "Total Assets": metrics.get("total_assets_crore"),
        "Equity": metrics.get("equity_crore"),
    }
    data = {k: v for k, v in chart_map.items() if v is not None}
    if not data:
        return

    colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b"]
    fig = go.Figure(go.Bar(
        y=list(data.keys()), x=list(data.values()), orientation="h",
        marker=dict(color=colors[:len(data)], opacity=0.85),
        text=[f"₹{v:,.0f} Cr" for v in data.values()],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#94a3b8", family="Inter"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
        margin=dict(t=10, b=10, l=10, r=70), height=220,
        title=dict(text=f"{company} — Financial Overview", font=dict(color="#94a3b8", size=12)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compare_chart(company_data: dict, metric_label: str):
    """Bar chart for one metric across multiple companies."""
    companies = list(company_data.keys())
    values = list(company_data.values())
    colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981"]

    fig = go.Figure(go.Bar(
        x=companies, y=values,
        marker_color=colors[:len(companies)],
        text=[f"{v:,.1f}" if v else "N/A" for v in values],
        textposition="outside", textfont=dict(color="#e2e8f0", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#94a3b8", family="Inter"),
        title=dict(text=metric_label, font=dict(color="#94a3b8", size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
        margin=dict(t=40, b=10, l=10, r=10), height=260,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_chat_history():
    """Render all messages in the current conversation."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        meta = msg.get("metadata") or {}

        with st.chat_message(role, avatar="🧑" if role == "user" else "📈"):
            st.markdown(content)

            # KPI cards (dashboard mode results)
            if meta.get("kpi_metrics"):
                render_kpi_cards(meta["kpi_metrics"])
                render_kpi_chart(meta["kpi_metrics"], meta.get("company", ""))

            # Comparison chart
            if meta.get("compare_chart"):
                cc = meta["compare_chart"]
                render_compare_chart(cc["data"], cc["label"])

            # Sources
            if meta.get("sources"):
                with st.expander(f"Sources ({len(meta['sources'])})", expanded=False):
                    render_source_chips(meta["sources"])


# ════════════════════════════════════════════════════════════════════
# Pipeline runners
# ════════════════════════════════════════════════════════════════════

def run_chat(query: str, company: str, fy: str | None) -> tuple[str, dict]:
    from src.generation.prompts import build_qa_prompt

    retriever = load_retriever()
    reranker  = load_reranker()
    llm       = load_groq()

    with st.spinner("Retrieving relevant passages…"):
        candidates = retriever.retrieve(
            query=query, top_k=50,
            company_filter=company,
            fiscal_year_filter=fy,
            expand_query=True, promote_to_parent=True,
        )

    if not candidates:
        return "No relevant documents found for this query.", {}

    with st.spinner("Reranking results…"):
        reranked = reranker.rerank(query=query, candidates=candidates, top_n=5)

    system_prompt, user_msg = build_qa_prompt(
        query=query, results=reranked,
        company_context=f"{company} ({settings.company_ticker_map.get(company, '')})",
    )

    with st.chat_message("assistant", avatar="📈"):
        answer = st.write_stream(llm.generate_stream(system_prompt, user_msg))

    sources = [
        {"label": f"{c.company} | {c.fiscal_year} | Pg.{c.page_number}",
         "is_table": c.content_type == "table"}
        for c in reranked
    ]
    meta = {"sources": sources, "company": company}
    with st.chat_message("assistant", avatar="📈"):
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            render_source_chips(sources)

    return answer, meta


def run_dashboard(company: str, fy: str | None) -> tuple[str, dict]:
    retriever = load_retriever()
    reranker  = load_reranker()
    extractor = load_metric_extractor()

    with st.spinner(f"Retrieving financial data for {company}…"):
        all_cands = []
        for q in ["revenue net profit financial performance", "balance sheet assets equity"]:
            all_cands.extend(retriever.retrieve(
                query=q, top_k=20, company_filter=company,
                fiscal_year_filter=fy, expand_query=False,
            ))
        seen = set(); unique = []
        for c in all_cands:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id); unique.append(c)

        reranked = reranker.rerank(
            query="financial metrics revenue profit assets equity EPS",
            candidates=unique, top_n=4,
        )

    with st.spinner("Extracting KPIs with AI…"):
        data = extractor.extract(company, reranked)

    metrics = data.get("metrics", {})
    fy_label = data.get("fiscal_year", fy or "FY2025")
    ticker = settings.company_ticker_map.get(company, company)

    answer = f"**{company}** ({ticker}) — KPI Summary for **{fy_label}**"

    with st.chat_message("assistant", avatar="📈"):
        st.markdown(answer)
        render_kpi_cards(metrics)
        render_kpi_chart(metrics, company)

    meta = {"kpi_metrics": metrics, "company": company, "fy": fy_label}
    return answer, meta


def run_compare(query: str, companies: list[str], fy: str | None) -> tuple[str, dict]:
    comparator = load_comparator()
    extractor  = load_metric_extractor()

    with st.spinner(f"Comparing {' vs '.join(companies)}…"):
        company_results, answer_gen = comparator.compare(
            query=query, companies=companies, fiscal_year=fy, stream=True,
        )

    meta = {}
    with st.chat_message("assistant", avatar="📈"):
        answer = st.write_stream(answer_gen)

        # Try to extract one plottable metric
        try:
            with st.spinner("Generating comparison chart…"):
                mlist = extractor.extract_multi(company_results)
                comp_table = extractor.to_comparison_table(mlist)
                comp_table.pop("companies", None)

                for metric_label, vals in comp_table.items():
                    numeric = {c: v for c, v in vals.items() if v is not None and isinstance(v, (int, float))}
                    if len(numeric) >= 2:
                        render_compare_chart(numeric, metric_label)
                        meta["compare_chart"] = {"data": numeric, "label": metric_label}
                        break
        except Exception:
            pass

    sources = []
    for company, results in company_results.items():
        for r in results:
            sources.append({"label": f"{company} Pg.{r.page_number}", "is_table": False})
    meta["sources"] = sources

    return answer, meta


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="fin-brand">
        <span class="fin-brand-icon">📈</span>
        <div>
            <div class="fin-brand-name">FinRAG</div>
            <div class="fin-brand-sub">Indian Markets Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New chat button
    if st.button("＋  New Chat", use_container_width=True,
                 type="secondary", key="new_chat_btn"):
        start_new_conversation()
        st.rerun()

    st.markdown("<hr style='margin:0.6rem 0'>", unsafe_allow_html=True)

    # Quick actions
    st.markdown('<div class="history-group-label">Quick Actions</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚖️ Compare", use_container_width=True, key="sb_compare"):
            st.session_state.mode = "compare"
            if not st.session_state.conversation_id:
                start_new_conversation()
            st.rerun()
    with col_b:
        if st.button("📊 Dashboard", use_container_width=True, key="sb_dashboard"):
            st.session_state.mode = "dashboard"
            if not st.session_state.conversation_id:
                start_new_conversation()
            st.rerun()

    st.markdown("<hr style='margin:0.6rem 0'>", unsafe_allow_html=True)

    # Conversation history
    all_convs = list_conversations(40)
    if all_convs:
        grouped = group_conversations_by_date(all_convs)
        current_cid = st.session_state.conversation_id

        for group_label, convs in grouped.items():
            st.markdown(f'<div class="history-group-label">{group_label}</div>', unsafe_allow_html=True)
            for conv in convs:
                is_active = conv["id"] == current_cid
                active_cls = "history-item active" if is_active else "history-item"
                mode_icon = {"chat": "💬", "compare": "⚖️", "dashboard": "📊"}.get(conv["mode"], "💬")
                label = f"{mode_icon} {conv['title']}"

                # Use a button styled as a list item
                if st.button(label, key=f"hist_{conv['id']}", use_container_width=True,
                             help=conv["title"]):
                    load_conversation(conv["id"])
                    st.rerun()
    else:
        st.markdown('<div style="padding:0.5rem 0.8rem;font-size:0.78rem;color:#334155;">No history yet</div>',
                    unsafe_allow_html=True)

    # Sidebar footer
    st.markdown("<hr style='margin:0.6rem 0'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="padding:0.3rem 0.8rem;font-size:0.68rem;color:#334155;line-height:1.6;">'
        f'17 companies · FY2024–25<br>'
        'Hybrid RAG · BGE-large · Groq LPU'
        '</div>', unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════
main = st.container()

with main:
    # ── Mode pills (top bar) ────────────────────────────────────────
    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    mode_col, spacer = st.columns([3, 7])
    with mode_col:
        mode_labels = {"chat": "💬 Chat", "compare": "⚖️ Compare", "dashboard": "📊 Dashboard"}
        new_mode = st.radio(
            "Mode", list(mode_labels.values()),
            index=list(mode_labels.keys()).index(st.session_state.mode),
            horizontal=True, label_visibility="collapsed", key="mode_radio",
        )
        selected_mode_key = list(mode_labels.keys())[list(mode_labels.values()).index(new_mode)]
        if selected_mode_key != st.session_state.mode:
            st.session_state.mode = selected_mode_key
            st.rerun()

    st.markdown("<hr style='margin:0.3rem 0 0.6rem 0'>", unsafe_allow_html=True)

    # ── Context selectors ───────────────────────────────────────────
    if st.session_state.mode == "compare":
        ctx1, ctx2, ctx3 = st.columns([3, 1.5, 5])
        with ctx1:
            sel_companies = st.multiselect(
                "Companies", COMPANIES,
                default=st.session_state.companies_compare or ["TCS", "INFOSYS"],
                max_selections=4, label_visibility="collapsed", key="cmp_sel",
            )
            st.session_state.companies_compare = sel_companies
        with ctx2:
            fy_sel = st.selectbox("FY", ["FY2025", "FY2024", "All"], index=0,
                                  label_visibility="collapsed", key="cmp_fy")
    elif st.session_state.mode == "dashboard":
        ctx1, ctx2, ctx3 = st.columns([2.5, 1.5, 6])
        with ctx1:
            sel_company = st.selectbox("Company", COMPANIES, label_visibility="collapsed",
                                       key="dash_company_sel")
            st.session_state.company = sel_company
        with ctx2:
            fy_sel = st.selectbox("FY", ["FY2025", "FY2024"], index=0,
                                  label_visibility="collapsed", key="dash_fy")
        with ctx3:
            if st.button("Extract KPIs →", type="primary", key="dash_extract_btn"):
                fy_filter = None if fy_sel == "All" else fy_sel

                if not st.session_state.conversation_id:
                    start_new_conversation()

                user_text = f"Show KPI dashboard for {st.session_state.company} {fy_sel}"
                st.session_state.messages.append({"role": "user", "content": user_text, "metadata": {}})
                save_message("user", user_text)

                answer, meta = run_dashboard(st.session_state.company, fy_filter)

                st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": meta})
                save_message("assistant", answer, meta)
                st.rerun()
    else:
        # Chat mode context bar
        ctx1, ctx2, ctx3 = st.columns([2.5, 1.5, 6])
        with ctx1:
            sel_company = st.selectbox(
                "Company", COMPANIES,
                index=COMPANIES.index(st.session_state.company) if st.session_state.company in COMPANIES else 0,
                label_visibility="collapsed", key="chat_company_sel",
            )
            st.session_state.company = sel_company
        with ctx2:
            fy_sel = st.selectbox("FY", ["FY2025", "FY2024", "All"],
                                  index=0, label_visibility="collapsed", key="chat_fy_sel")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── Chat thread ─────────────────────────────────────────────────
    if not st.session_state.messages:
        # Welcome screen
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-logo">📈</div>
            <div class="welcome-title">FinRAG</div>
            <div class="welcome-sub">
                Ask anything about 17 major Indian companies.<br>
                Grounded in official BSE annual reports.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggestion cards
        suggestions = [
            ("💰", "What was TCS's revenue and net profit for FY2025?"),
            ("🏦", "What is HDFC Bank's gross NPA ratio?"),
            ("⚖️", "Compare Infosys vs HCL Technologies on revenue growth"),
            ("📊", "Show KPI dashboard for Reliance Industries"),
        ]
        cols = st.columns(2)
        for i, (icon, text) in enumerate(suggestions):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="suggestion-card"><span class="suggestion-icon">{icon}</span>{text}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(text, key=f"sug_{i}", help=text,
                             use_container_width=True):
                    st.session_state.pending_suggestion = text
                    st.rerun()
    else:
        render_chat_history()

    # ── Chat input ──────────────────────────────────────────────────
    mode = st.session_state.mode
    placeholders = {
        "chat": "Ask anything… use @CompanyName to specify (e.g. @TCS what was the revenue?)",
        "compare": "Compare selected companies… (e.g. Compare revenue and margins)",
        "dashboard": "Dashboard mode — use Extract KPIs button above, or type a question",
    }

    # Handle pending suggestion (clicked welcome card)
    prefill = st.session_state.pop("pending_suggestion", None) or ""

    user_input = st.chat_input(
        placeholder=placeholders.get(mode, "Ask anything…"),
        key="main_chat_input",
    )

    # Use suggestion if no direct input
    if not user_input and prefill:
        user_input = prefill

    if user_input:
        raw_query = user_input.strip()
        if not raw_query:
            st.stop()

        # Parse @ mentions
        query, mentioned_companies = parse_at_mentions(raw_query)

        # Override company from @ mention
        if mentioned_companies:
            st.session_state.company = mentioned_companies[0]
            if len(mentioned_companies) > 1:
                st.session_state.companies_compare = mentioned_companies

        # Auto-detect mode
        intent = detect_intent(query, mentioned_companies)
        if intent != "chat":
            st.session_state.mode = intent

        # Ensure conversation exists
        if not st.session_state.conversation_id:
            start_new_conversation()

        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(raw_query)

        # Save user message
        st.session_state.messages.append({"role": "user", "content": raw_query, "metadata": {}})
        save_message("user", raw_query)

        fy_filter = None if fy_sel == "All" else fy_sel

        # Route to correct pipeline
        try:
            if st.session_state.mode == "dashboard":
                answer, meta = run_dashboard(st.session_state.company, fy_filter)

            elif st.session_state.mode == "compare":
                compare_cos = st.session_state.companies_compare or [st.session_state.company]
                if len(compare_cos) < 2:
                    with st.chat_message("assistant", avatar="📈"):
                        st.warning("Please select at least 2 companies from the Compare selector above.")
                    answer, meta = "Please select at least 2 companies.", {}
                else:
                    answer, meta = run_compare(query, compare_cos, fy_filter)

            else:  # chat
                answer, meta = run_chat(query, st.session_state.company, fy_filter)

        except Exception as e:
            with st.chat_message("assistant", avatar="📈"):
                st.error(f"Something went wrong: {e}")
            answer, meta = str(e), {}

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": meta})
        save_message("assistant", answer, meta)
        st.rerun()
