"""
app/pages/3_Dashboard.py
=========================
KPI financial dashboard with automatic metric extraction and Plotly visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard | FinRAG", page_icon="📊", layout="wide")

from app.rag_engine import load_retriever, load_reranker, load_metric_extractor

# Shared CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: linear-gradient(135deg, #0a0e1a, #0f1628); font-family: 'Inter', sans-serif; color: #f1f5f9; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
.metric-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(6,182,212,0.05));
    border: 1px solid rgba(59,130,246,0.2); border-radius: 10px; padding: 1rem; text-align: center;
}
.metric-value { font-size:1.5rem; font-weight:700; color:#06b6d4; font-family:'JetBrains Mono',monospace; }
.metric-label { font-size:0.72rem; color:#64748b; margin-top:0.25rem; text-transform:uppercase; letter-spacing:0.05em; }
.na-metric { color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div style="padding:1.5rem 0 1rem 0;">
    <h1 style="font-size:1.8rem;font-weight:700;margin:0;
               background:linear-gradient(135deg,#10b981,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        📊 KPI Dashboard
    </h1>
    <p style="color:#64748b;font-size:0.9rem;margin-top:0.3rem;">
        Auto-extract financial KPIs from annual reports and visualize with interactive charts.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Controls ----
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    companies = list(settings.company_ticker_map.keys())
    selected_company = st.selectbox("Select Company", companies, key="dash_company")
with col2:
    fy_options = ["FY2025", "FY2024", "All Years"]
    selected_fy = st.selectbox("Fiscal Year", fy_options, key="dash_fy")
    fy_filter = None if selected_fy == "All Years" else selected_fy
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    extract_btn = st.button("Extract KPIs", type="primary", key="dash_extract")

# ---- Extraction & Display ----
if extract_btn:
    retriever = load_retriever()
    reranker = load_reranker()
    extractor = load_metric_extractor()

    # Retrieve financial statement chunks
    with st.spinner(f"Retrieving financial data for {selected_company}..."):
        # Search specifically for financial metrics
        metric_queries = [
            "revenue net profit income financial performance",
            "balance sheet total assets equity",
            "earnings per share EPS return on equity",
        ]

        all_candidates = []
        for q in metric_queries:
            candidates = retriever.retrieve(
                query=q,
                top_k=20,
                company_filter=selected_company,
                fiscal_year_filter=fy_filter,
                expand_query=False,
            )
            all_candidates.extend(candidates)

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique_candidates.append(c)

        reranked = reranker.rerank(
            query="financial metrics revenue profit assets equity EPS NPA",
            candidates=unique_candidates,
            top_n=8,
        )

    with st.spinner("Extracting KPIs using AI..."):
        metrics_data = extractor.extract(selected_company, reranked[:4])  # max 4 chunks to stay under token limit

    metrics = metrics_data.get("metrics", {})
    fiscal_year = metrics_data.get("fiscal_year", selected_fy or "FY2025")
    ticker = settings.company_ticker_map.get(selected_company, selected_company)

    # ---- Company header ----
    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;">
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px;padding:0.8rem 1.2rem;">
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;">{selected_company}</div>
            <div style="font-size:0.8rem;color:#3b82f6;font-family:monospace;">{ticker}</div>
        </div>
        <div>
            <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Fiscal Year</div>
            <div style="font-size:1.1rem;font-weight:600;color:#06b6d4;">{fiscal_year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- KPI Metric Cards ----
    def fmt_crore(v):
        if v is None: return "N/A"
        if v >= 100000: return f"₹{v/100000:.2f}L Cr"
        if v >= 1000: return f"₹{v:,.0f} Cr"
        return f"₹{v:.1f} Cr"

    def fmt_pct(v):
        return f"{v:.2f}%" if v is not None else "N/A"

    def fmt_plain(v):
        return f"{v:.2f}" if v is not None else "N/A"

    # Row 1: Revenue metrics
    st.markdown("**Revenue & Profitability**")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        v = metrics.get("revenue_crore")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_crore(v)}</div><div class="metric-label">Revenue</div></div>', unsafe_allow_html=True)
    with r1c2:
        v = metrics.get("net_profit_crore")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_crore(v)}</div><div class="metric-label">Net Profit</div></div>', unsafe_allow_html=True)
    with r1c3:
        v = metrics.get("net_profit_margin_pct")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_pct(v)}</div><div class="metric-label">Net Margin</div></div>', unsafe_allow_html=True)
    with r1c4:
        v = metrics.get("ebitda_margin_pct")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_pct(v)}</div><div class="metric-label">EBITDA Margin</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Balance sheet
    st.markdown("**Balance Sheet**")
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        v = metrics.get("total_assets_crore")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_crore(v)}</div><div class="metric-label">Total Assets</div></div>', unsafe_allow_html=True)
    with r2c2:
        v = metrics.get("equity_crore")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_crore(v)}</div><div class="metric-label">Equity</div></div>', unsafe_allow_html=True)
    with r2c3:
        v = metrics.get("eps_inr")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{"₹"+fmt_plain(v) if v is not None else "N/A"}</div><div class="metric-label">EPS</div></div>', unsafe_allow_html=True)
    with r2c4:
        v = metrics.get("roe_pct")
        st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_pct(v)}</div><div class="metric-label">ROE</div></div>', unsafe_allow_html=True)

    # Banking metrics (if applicable)
    if selected_company in {"HDFC", "ICICI", "AXIS", "SBI", "KOTAK", "KVB"}:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Banking Metrics**")
        rb1, rb2, rb3 = st.columns(3)
        with rb1:
            v = metrics.get("npa_gross_pct")
            st.markdown(f'<div class="metric-card" style="border-color:rgba(239,68,68,0.3)"><div class="metric-value {"na-metric" if v is None else ""}" style="{"color:#f87171" if v and v > 3 else ""}">{fmt_pct(v)}</div><div class="metric-label">Gross NPA</div></div>', unsafe_allow_html=True)
        with rb2:
            v = metrics.get("npa_net_pct")
            st.markdown(f'<div class="metric-card" style="border-color:rgba(239,68,68,0.3)"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_pct(v)}</div><div class="metric-label">Net NPA</div></div>', unsafe_allow_html=True)
        with rb3:
            v = metrics.get("nim_pct")
            st.markdown(f'<div class="metric-card"><div class="metric-value {"na-metric" if v is None else ""}">{fmt_pct(v)}</div><div class="metric-label">Net Interest Margin</div></div>', unsafe_allow_html=True)

    # ---- Plotly Chart ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Financial Overview Chart**")

    chart_data = {}
    label_map = {
        "revenue_crore": "Revenue",
        "net_profit_crore": "Net Profit",
        "ebitda_crore": "EBITDA",
        "total_assets_crore": "Total Assets",
        "equity_crore": "Equity",
    }

    for key, label in label_map.items():
        v = metrics.get(key)
        if v is not None:
            chart_data[label] = v

    if chart_data:
        colors_list = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b"]
        fig = go.Figure(go.Bar(
            y=list(chart_data.keys()),
            x=list(chart_data.values()),
            orientation="h",
            marker=dict(
                color=colors_list[:len(chart_data)],
                opacity=0.85,
            ),
            text=[f"₹{v:,.0f} Cr" for v in chart_data.values()],
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#f1f5f9", size=12)),
            margin=dict(t=10, b=20, l=20, r=80),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monetary metrics not available for charting. Check the KPI cards above.")

    # ---- Source citation ----
    st.markdown("---")
    st.caption(f"Source: {metrics_data.get('source_citation', f'{selected_company} BSE Annual Report {fiscal_year}')}")
    with st.expander("Retrieved document chunks used for extraction"):
        for chunk in reranked:
            st.markdown(f"**Page {chunk.page_number}** | {chunk.section} | {chunk.content_type}")
            st.code(chunk.content[:400], language=None)
