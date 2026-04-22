"""
main.py — AI Ticket Classification Model
Streamlit application.

Run with:  streamlit run main.py
"""

from __future__ import annotations

import os
import sys
import traceback
from threading import Lock

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure local imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from models.llm_labeler import LLMLabeler
from pipeline import PipelineConfig, PipelineResult, run_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Ticket Classification Model",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean dark theme, monospace type, teal accent
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4             { font-family: 'JetBrains Mono', monospace !important; }

.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem; font-weight: 700;
    color: #00e5c0;
    border-bottom: 2px solid #00e5c033;
    padding-bottom: 0.5rem; margin-bottom: 0.3rem;
}
.app-subtitle {
    color: #9ca3af; font-size: 0.93rem; margin-bottom: 1.8rem;
}
.pipeline-step {
    background: #111827; border: 1px solid #1f2937;
    border-left: 3px solid #00e5c0;
    border-radius: 6px; padding: 0.5rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #00e5c0;
    margin: 0.3rem 0;
}
.metric-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 8px; padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem; font-weight: 700; color: #00e5c0;
}
.metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.warn-box {
    background: #1c1200; border-left: 3px solid #f59e0b;
    padding: 0.6rem 1rem; border-radius: 4px;
    color: #fbbf24; font-size: 0.87rem; margin: 0.3rem 0;
}
.err-box {
    background: #1c0000; border-left: 3px solid #ef4444;
    padding: 0.6rem 1rem; border-radius: 4px;
    color: #f87171; font-size: 0.87rem; margin: 0.3rem 0;
}
.free-badge {
    background: #064e3b; color: #6ee7b7;
    font-size: 0.72rem; border-radius: 9999px;
    padding: 1px 8px; margin-left: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.paid-badge {
    background: #1e3a5f; color: #93c5fd;
    font-size: 0.72rem; border-radius: 9999px;
    padding: 1px 8px; margin-left: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00e5c0, #0ea5e9) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe Streamlit progress helper
# ─────────────────────────────────────────────────────────────────────────────

class StepProgress:
    def __init__(self, bar, label_slot, step_name: str,
                 weight: float, offset: float):
        self.bar       = bar
        self.label     = label_slot
        self.step_name = step_name
        self.weight    = weight
        self.offset    = offset
        self._lock     = Lock()

    def __call__(self, current: int, total: int):
        if total == 0:
            return
        frac      = current / total
        global_pct = min(self.offset + frac * self.weight, 1.0)
        with self._lock:
            self.bar.progress(global_pct)
            pct_str = f"{int(frac * 100)}%"
            self.label.markdown(
                f'<div class="pipeline-step">{self.step_name} — {pct_str}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> PipelineConfig:
    st.sidebar.markdown("## ⚙️ Configuration")

    # ── Column mapping ────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📋 Column Names")
    text_col = st.sidebar.text_input(
        "Text column *", value="ticket_context",
        help="Column that contains the ticket description / conversation."
    )
    cat_col = st.sidebar.text_input(
        "Category column (optional)", value="category",
        help="Existing category column — used as extra context for embeddings."
    )
    sub_col = st.sidebar.text_input(
        "Subcategory column (optional)", value="Subcategory"
    )

    # ── Embedding model ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔢 Embedding Model")
    embed_model = st.sidebar.selectbox(
        "Sentence-transformer model",
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ],
        help=(
            "all-MiniLM-L6-v2 — fastest (384-dim)\n"
            "all-mpnet-base-v2 — higher quality (768-dim)\n"
            "multilingual — for non-English tickets"
        ),
    )

    # ── UMAP ──────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📐 UMAP")
    umap_neighbors = st.sidebar.slider("n_neighbors", 5, 50, 15,
        help="Higher = more global structure; lower = more local clusters.")
    umap_min_dist  = st.sidebar.slider("min_dist", 0.0, 0.5, 0.1, step=0.05,
        help="Higher = more spread out in 2D visualization.")

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔵 HDBSCAN")
    min_cluster = st.sidebar.slider("min_cluster_size", 2, 20, 3,
        help="Minimum tickets to form a cluster. Lower = more clusters.")
    min_samples = st.sidebar.slider("min_samples", 1, 10, 2,
        help="Higher = stricter core point definition (more noise points).")

    # ── LLM Backend ───────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 LLM for Label Assignment")

    st.sidebar.markdown(
        "The LLM reads the ticket conversations and assigns the correct "
        "**technology category** (Python, React, Azure, SQL…) and "
        "**subcategory** (Runtime Error, Build Issue, Performance…)."
    )

    backend_display = st.sidebar.selectbox(
        "Select LLM backend",
        options=list(LLMLabeler.BACKEND_OPTIONS.keys()),
        index=5,   # default: Flan-T5 (free, no token)
        format_func=lambda x: x,
    )
    backend_key = LLMLabeler.BACKEND_OPTIONS[backend_display]

    # Determine if token needed
    is_paid = backend_key in LLMLabeler.PAID_BACKENDS
    is_free = backend_key in LLMLabeler.FREE_BACKENDS

    api_key  = ""
    hf_token = ""

    if is_paid:
        st.sidebar.markdown(
            f'<span class="paid-badge">💳 PAID — API key required</span>',
            unsafe_allow_html=True,
        )
        key_label = {
            "chatgpt": "OpenAI API Key",
            "claude":  "Anthropic API Key",
            "gemini":  "Google API Key",
        }.get(backend_key, "API Key")
        api_key = st.sidebar.text_input(key_label, type="password",
            help="Your API key is used only for this session and never stored.")

        model_options = {
            "chatgpt": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "claude":  ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
            "gemini":  ["gemini-1.5-flash", "gemini-1.5-pro"],
        }
        if backend_key in model_options:
            chosen_model = st.sidebar.selectbox(
                "Model variant", model_options[backend_key]
            )
        else:
            chosen_model = ""

    else:
        st.sidebar.markdown(
            '<span class="free-badge">✅ FREE — no token needed</span>',
            unsafe_allow_html=True,
        )
        hf_token = st.sidebar.text_input(
            "HuggingFace Token (optional)",
            type="password",
            help="Not required. Adding an HF token gives higher API rate limits.",
        )
        chosen_model = ""

    llm_samples = st.sidebar.slider(
        "Tickets per cluster sent to LLM", 3, 10, 5,
        help="More samples = better context for the LLM; slightly slower."
    )

    return PipelineConfig(
        text_column        = text_col,
        category_column    = cat_col,
        subcategory_column = sub_col,
        embedding_model    = embed_model,
        umap_2d_n_neighbors    = umap_neighbors,
        umap_2d_min_dist       = umap_min_dist,
        umap_cluster_n_neighbors = umap_neighbors,
        hdbscan_min_cluster_size = min_cluster,
        hdbscan_min_samples      = min_samples,
        llm_backend              = backend_key,
        llm_api_key              = api_key,
        llm_hf_token             = hf_token,
        llm_samples_per_cluster  = llm_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cluster scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def render_cluster_plot(result: PipelineResult, df_raw: pd.DataFrame,
                        text_col: str):
    """Interactive 2D UMAP scatter plot coloured by cluster & category."""
    if result.reduced_2d is None or result.cluster_labels is None:
        st.info("Cluster plot unavailable — UMAP data missing.")
        return

    coords = result.reduced_2d
    labels = result.cluster_labels

    id_to_cat = {s["cluster_id"]: s["category"]    for s in result.cluster_summary}
    id_to_sub = {s["cluster_id"]: s["subcategory"]  for s in result.cluster_summary}

    unique_labels = sorted(set(labels))
    palette = [
        "#00e5c0","#0ea5e9","#f59e0b","#ef4444","#a855f7",
        "#ec4899","#22d3ee","#84cc16","#f97316","#6366f1",
        "#14b8a6","#eab308","#8b5cf6","#06b6d4","#d946ef",
    ]
    color_map = {
        lbl: ("#555" if lbl == -1 else palette[i % len(palette)])
        for i, lbl in enumerate(unique_labels)
    }

    hover_col = text_col if text_col in df_raw.columns else df_raw.columns[0]
    hover = [
        (
            f"<b>Cluster {lbl}</b><br>"
            f"Category: {id_to_cat.get(int(lbl), '?')}<br>"
            f"Subcategory: {id_to_sub.get(int(lbl), '?')}<br>"
            f"<i>{str(df_raw[hover_col].iloc[i])[:100]}…</i>"
        )
        for i, lbl in enumerate(labels)
    ]

    traces = []
    for lbl in unique_labels:
        mask = labels == lbl
        name = f"Noise" if lbl == -1 else f"C{lbl}: {id_to_cat.get(int(lbl), '?')}"
        traces.append(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=name,
            marker=dict(
                color=color_map[lbl], size=8 if lbl != -1 else 5,
                opacity=0.85 if lbl != -1 else 0.4,
                line=dict(width=0.4, color="#000"),
            ),
            text=[hover[i] for i, m in enumerate(mask) if m],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="UMAP Cluster Map — coloured by CorrectCategory",
                   font=dict(family="JetBrains Mono", size=14, color="#e5e7eb")),
        template="plotly_dark",
        paper_bgcolor="#0b1120", plot_bgcolor="#0b1120",
        font=dict(family="JetBrains Mono", color="#9ca3af"),
        legend=dict(bgcolor="#111827", bordercolor="#1f2937", font_size=11),
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="app-title">🎫 AI Ticket Classification Model</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Upload support tickets → clean → embed → UMAP → HDBSCAN → LLM labels '
        '→ <b>CorrectCategory</b> (Python / React / Azure / SQL…) + '
        '<b>CorrectSubcategory</b> (Runtime Error / Build / Performance…)'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar config ────────────────────────────────────────────────────────
    cfg = render_sidebar()

    # ── Pipeline diagram (static) ─────────────────────────────────────────────
    with st.expander("📊 Pipeline Architecture", expanded=False):
        steps = [
            "① User Upload (Excel / CSV)",
            "② Data Validation",
            "③ Text Cleaning  (lowercase · tech-term normalization · stopwords)",
            "④ Embedding Layer  (sentence-transformers)",
            "⑤ Dimensionality Reduction  (UMAP)",
            "⑥ HDBSCAN Clustering",
            "⑦ LLM Model — reads ticket conversation → assigns CorrectCategory + CorrectSubcategory",
            "⑧ Output  (augmented dataset download)",
        ]
        for s in steps:
            st.markdown(f'<div class="pipeline-step">{s}</div>', unsafe_allow_html=True)

    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Step 1 — Upload Your Dataset")
    col_up, col_sample = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Drop a CSV or Excel file here",
            type=["csv", "xlsx", "xls", "xlsb"],
            help=(
                f"Required column: **{cfg.text_column}** (your ticket text).\n"
                "Optional: category, Subcategory, ID."
            ),
        )

    with col_sample:
        st.markdown("&nbsp;")
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "sample_tickets.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    "⬇ sample_tickets.csv",
                    data=f.read(),
                    file_name="sample_tickets.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="40 realistic tech support tickets across Python, Java, React, Azure, etc.",
                )

    if uploaded is None:
        st.info("👆 Upload a dataset to begin. Or download the sample CSV to try it out.")
        return

    # ── Load + preview ────────────────────────────────────────────────────────
    try:
        df = utils.load_dataset(uploaded, uploaded.name)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return

    st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")

    with st.expander("👁 Preview (first 5 rows)", expanded=True):
        st.dataframe(df.head(5), use_container_width=True)

    # Quick guard: text column must exist
    if cfg.text_column not in df.columns:
        st.error(
            f"❌ Text column **'{cfg.text_column}'** not found in your file.\n\n"
            f"Available columns: `{list(df.columns)}`\n\n"
            "→ Update **Text column** in the sidebar to match your file."
        )
        return

    # ── LLM config validation before run ──────────────────────────────────────
    labeler_check = LLMLabeler(
        backend=cfg.llm_backend,
        api_key=cfg.llm_api_key,
        hf_token=cfg.llm_hf_token,
    )
    cfg_error = labeler_check.validate()
    if cfg_error:
        st.warning(f"⚠️ LLM config: {cfg_error}")

    st.divider()

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("### 🚀 Step 2 — Run Classification Pipeline")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)
    with col_info:
        is_free_lbl = "✅ Free" if labeler_check.is_free else "💳 Paid"
        st.caption(
            f"Model: `{cfg.embedding_model}` · "
            f"LLM: `{cfg.llm_backend}` ({is_free_lbl}) · "
            f"HDBSCAN min_cluster={cfg.hdbscan_min_cluster_size}"
        )

    if not run_btn:
        return

    # ── Progress UI ───────────────────────────────────────────────────────────
    progress_bar  = st.progress(0.0)
    status_slot   = st.empty()

    # Step weights (fraction of total bar)
    W = {"clean": 0.08, "embed": 0.50, "umap_hdb": 0.20, "label": 0.22}

    cb_clean = StepProgress(progress_bar, status_slot,
                            "③ Text Cleaning", W["clean"], 0.0)
    cb_embed = StepProgress(progress_bar, status_slot,
                            "④ Embedding Layer", W["embed"], W["clean"])
    cb_label = StepProgress(progress_bar, status_slot,
                            "⑦ LLM Labelling", W["label"],
                            W["clean"] + W["embed"] + W["umap_hdb"])

    # UMAP + HDBSCAN are fast — update bar manually mid-way
    def _mark_umap_hdb():
        frac = W["clean"] + W["embed"] + W["umap_hdb"]
        progress_bar.progress(min(frac, 1.0))
        status_slot.markdown(
            '<div class="pipeline-step">⑤⑥ UMAP + HDBSCAN Clustering…</div>',
            unsafe_allow_html=True,
        )

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        status_slot.markdown(
            '<div class="pipeline-step">⏳ Starting pipeline…</div>',
            unsafe_allow_html=True,
        )

        # We monkey-patch a mid-step callback for UMAP/HDBSCAN
        import pipeline as _pl
        _orig_umap_2d = _pl.step_umap_2d

        def _patched_umap_2d(embeddings, cfg_):
            _mark_umap_hdb()
            return _orig_umap_2d(embeddings, cfg_)

        _pl.step_umap_2d = _patched_umap_2d

        result: PipelineResult = run_pipeline(
            df, cfg=cfg,
            progress_callbacks={"clean": cb_clean, "embed": cb_embed, "label": cb_label},
        )

        _pl.step_umap_2d = _orig_umap_2d   # restore

        progress_bar.progress(1.0)
        status_slot.markdown(
            '<div class="pipeline-step">✅ Pipeline complete!</div>',
            unsafe_allow_html=True,
        )

    except ValueError as e:
        st.error(f"❌ Validation error: {e}")
        return
    except Exception:
        st.error("❌ Unexpected pipeline error:")
        st.code(traceback.format_exc())
        return

    # ── Warnings & LLM errors ─────────────────────────────────────────────────
    for w in result.warnings:
        st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)
    for e in result.llm_errors:
        st.markdown(f'<div class="err-box">🔴 {e}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Results")

    n_clusters    = len([s for s in result.cluster_summary if s["cluster_id"] != -1])
    n_noise       = int((result.cluster_labels == -1).sum()) if result.cluster_labels is not None else 0
    n_classified  = len(result.df) - n_noise
    total_time    = result.timing.get("total", 0)

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{len(result.df):,}",  "Total Tickets"),
        (m2, str(n_clusters),         "Clusters Found"),
        (m3, f"{n_classified:,}",     "Tickets Labelled"),
        (m4, f"{total_time:.1f}s",    "Total Time"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Cluster visualization ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗺️ Cluster Map")
    render_cluster_plot(result, df, cfg.text_column)

    # ── Cluster summary table ─────────────────────────────────────────────────
    st.markdown("### 🗂️ Cluster Summary")
    summary_df = pd.DataFrame(result.cluster_summary)[[
        "cluster_id", "size", "category", "subcategory",
        "top_terms", "label_source", "sample_ticket",
    ]]
    summary_df.columns = [
        "Cluster", "Size", "CorrectCategory", "CorrectSubcategory",
        "Top Terms", "Label Source", "Sample Ticket",
    ]
    st.dataframe(summary_df, use_container_width=True, height=320)

    # ── Augmented dataset ─────────────────────────────────────────────────────
    st.markdown("### 📋 Augmented Dataset (preview — first 20 rows)")
    preview_cols = [
        c for c in [
            cfg.text_column, cfg.category_column, cfg.subcategory_column,
            "CorrectCategory", "CorrectSubcategory",
        ] if c in result.df.columns
    ]
    st.dataframe(result.df[preview_cols].head(20), use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⬇️ Download")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "📄 Download CSV",
            data=utils.save_results(result.df, fmt="csv"),
            file_name="classified_tickets.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📊 Download Excel",
            data=utils.save_results(result.df, fmt="excel"),
            file_name="classified_tickets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl3:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        st.download_button(
            "🗜️ Download Source ZIP",
            data=utils.build_code_zip(project_dir=project_dir),
            file_name="ticket_classifier_source.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # ── Timing ────────────────────────────────────────────────────────────────
    with st.expander("⏱️ Step Timing"):
        timing_df = pd.DataFrame(
            [(k, f"{v:.3f}s") for k, v in result.timing.items()],
            columns=["Pipeline Step", "Time"],
        )
        st.table(timing_df)


if __name__ == "__main__":
    main()
