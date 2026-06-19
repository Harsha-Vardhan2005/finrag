"""
app/pages/2_Compare.py
========================
Cross-company comparison page with streaming answers and auto-generated Plotly charts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Compare | FinRAG", page_icon="⚖️", layout="wide")

from app.rag_engine import load_comparator, load_metric_extractor

# Shared CSS (same dark theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: linear-gradient(135deg, #0a0e1a, #0f1628); font-family: 'Inter', sans-serif; color: #f1f5f9; }
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
.citation-chip {
    display: inline-block; background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3); border-radius: 20px;
    padding: 0.2rem 0.7rem; font-size: 0.72rem; color: #93c5fd; margin: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div style="padding:1.5rem 0 1rem 0;">
    <h1 style="font-size:1.8rem;font-weight:700;margin:0;
               background:linear-gradient(135deg,#8b5cf6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⚖️ Compare Companies
    </h1>
    <p style="color:#64748b;font-size:0.9rem;margin-top:0.3rem;">
        Side-by-side comparison of 2–4 companies from their BSE annual reports.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- Company Selection ----
companies_all = list(settings.company_ticker_map.keys())

col1, col2 = st.columns([2, 1])
with col1:
    selected_companies = st.multiselect(
        "Select 2–4 Companies to Compare",
        options=companies_all,
        default=["TCS", "INFOSYS"],
        max_selections=4,
        key="cmp_companies",
    )
with col2:
    fy_options = ["All Years", "FY2025", "FY2024"]
    selected_fy = st.selectbox("Fiscal Year", fy_options, key="cmp_fy")
    fy_filter = None if selected_fy == "All Years" else selected_fy

# Sample comparison queries
sample_comparisons = [
    "Compare revenue and net profit margins",
    "Which company has better return on equity (ROE)?",
    "Compare employee count and revenue per employee",
    "Compare NPA ratios (for banks)",
    "Compare dividend payout and EPS",
    "Which company has stronger balance sheet?",
]

st.markdown("**Quick comparison templates:**")
cols = st.columns(3)
for i, q in enumerate(sample_comparisons):
    with cols[i % 3]:
        if st.button(q, key=f"cmp_template_{i}"):
            st.session_state["cmp_prefill"] = q

# ---- Query Input ----
default_q = st.session_state.pop("cmp_prefill", "")
compare_query = st.text_area(
    "Comparison question",
    value=default_q,
    placeholder="e.g. Compare revenue growth and net profit margin for the latest fiscal year",
    height=80,
    key="cmp_query",
)

compare_btn = st.button("Run Comparison", type="primary", key="cmp_run")

# ---- Comparison Results ----
if compare_btn:
    if len(selected_companies) < 2:
        st.warning("Please select at least 2 companies to compare.")
        st.stop()
    if not compare_query.strip():
        st.warning("Please enter a comparison question.")
        st.stop()

    comparator = load_comparator()
    metric_extractor = load_metric_extractor()

    with st.spinner(f"Retrieving data for {', '.join(selected_companies)}..."):
        company_results, answer_gen = comparator.compare(
            query=compare_query,
            companies=selected_companies,
            fiscal_year=fy_filter,
            stream=True,
        )

    st.markdown("---")

    # Tabs: Answer | Chart | Raw Data
    tab_answer, tab_chart, tab_data = st.tabs(["📝 Analysis", "📊 Chart", "🔍 Retrieved Data"])

    with tab_answer:
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;">
            Comparing: {' vs '.join(selected_companies)} | {fy_filter or 'All Years'}
        </div>
        """, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="📈"):
            answer_text = st.write_stream(answer_gen)

        # Citations per company
        st.markdown("**Sources:**")
        for company, results in company_results.items():
            chips = "".join(
                f'<span class="citation-chip">{company} Pg.{r.page_number}</span>'
                for r in results
            )
            st.markdown(f"**{company}:** {chips}", unsafe_allow_html=True)

    with tab_chart:
        st.markdown("*Extracting financial metrics for visualization...*")

        with st.spinner("Extracting KPIs with LLM..."):
            metrics_list = metric_extractor.extract_multi(company_results)
            comparison_data = metric_extractor.to_comparison_table(metrics_list)

        companies_in_data = comparison_data.pop("companies", selected_companies)

        # Find numeric metrics to plot
        plottable = {}
        for metric, values in comparison_data.items():
            numeric_vals = {c: v for c, v in values.items() if v is not None and isinstance(v, (int, float))}
            if len(numeric_vals) >= 2:
                plottable[metric] = numeric_vals

        if plottable:
            metric_to_plot = st.selectbox(
                "Select metric to chart",
                options=list(plottable.keys()),
                key="cmp_metric_chart",
            )

            chart_data = plottable[metric_to_plot]

            # Color palette
            colors = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(chart_data.keys()),
                y=list(chart_data.values()),
                marker_color=colors[:len(chart_data)],
                text=[f"{v:,.1f}" for v in chart_data.values()],
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=12),
            ))

            fig.update_layout(
                title=dict(
                    text=metric_to_plot,
                    font=dict(color="#f1f5f9", size=16),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#94a3b8"),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#f1f5f9", size=12),
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="#94a3b8"),
                ),
                hoverlabel=dict(
                    bgcolor="#1e293b",
                    font_color="#f1f5f9",
                ),
                margin=dict(t=50, b=20, l=20, r=20),
                height=380,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Radar chart if multiple metrics available
            if len(plottable) >= 3:
                st.markdown("**Multi-metric Radar Chart**")
                radar_metrics = list(plottable.keys())[:6]

                fig_radar = go.Figure()
                for i, company in enumerate(companies_in_data):
                    values = []
                    for m in radar_metrics:
                        v = plottable.get(m, {}).get(company)
                        values.append(v if v is not None else 0)

                    # Normalize to 0-100 scale per metric
                    maxv = max(max(plottable[m].values(), default=1) for m in radar_metrics)
                    norm_values = [v / maxv * 100 if maxv > 0 else 0 for v in values]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=radar_metrics + [radar_metrics[0]],
                        fill='toself',
                        name=company,
                        opacity=0.7,
                        line_color=colors[i % len(colors)],
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(255,255,255,0.02)",
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#64748b")),
                        angularaxis=dict(tickfont=dict(color="#f1f5f9")),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    height=380,
                    legend=dict(font=dict(color="#f1f5f9")),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Could not extract numeric metrics for charting from the available context. The text analysis above still provides the comparison.")

    with tab_data:
        for company, results in company_results.items():
            with st.expander(f"{company} — {len(results)} chunks retrieved"):
                for chunk in results:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                                border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;">
                        <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.3rem;">
                            {chunk.section} | Page {chunk.page_number} | {chunk.content_type} | Score: {chunk.rrf_score:.3f}
                        </div>
                        <div style="font-size:0.8rem;color:#cbd5e1;white-space:pre-wrap;">{chunk.content[:500]}{"..." if len(chunk.content) > 500 else ""}</div>
                    </div>
                    """, unsafe_allow_html=True)
