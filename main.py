"""
main.py  —  AI Ticket Classification Model
Streamlit front-end.

Run:  streamlit run main.py
"""

from __future__ import annotations

import os
import sys
import traceback
from threading import Lock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import (
    LLM_BACKENDS, PAID_BACKENDS, FREE_BACKENDS,
    load_file, export_file,
    clean_column, generate_embeddings, run_umap, run_hdbscan, label_clusters,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Ticket Classification",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]        { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, code, pre        { font-family: 'JetBrains Mono', monospace !important; }

/* ── Title ── */
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #00e5c0, #0ea5e9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.app-sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0f172a; border: 1px solid #1e293b;
    border-left: 3px solid #00e5c0;
    border-radius: 6px; padding: 0.45rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; color: #00e5c0;
    margin: 0.25rem 0; width: 100%;
}
.step-badge.active { border-left-color: #0ea5e9; color: #38bdf8; background: #0c1a2e; }
.step-badge.done   { border-left-color: #10b981; color: #34d399; background: #052e16; }

/* ── Metric card ── */
.metric-card {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #00e5c0;
}
.metric-label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }

/* ── Badges ── */
.badge-free {
    background:#064e3b; color:#6ee7b7; border-radius:9999px;
    padding:2px 9px; font-size:0.72rem; font-family:'JetBrains Mono',monospace;
}
.badge-paid {
    background:#1e3a5f; color:#93c5fd; border-radius:9999px;
    padding:2px 9px; font-size:0.72rem; font-family:'JetBrains Mono',monospace;
}

/* ── Warn / Error boxes ── */
.warn-box {
    background:#1c1200; border-left:3px solid #f59e0b;
    padding:0.55rem 1rem; border-radius:4px; color:#fbbf24; font-size:0.85rem; margin:0.3rem 0;
}
.err-box  {
    background:#1c0000; border-left:3px solid #ef4444;
    padding:0.55rem 1rem; border-radius:4px; color:#f87171; font-size:0.85rem; margin:0.3rem 0;
}

/* ── Progress bar colour ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00e5c0, #0ea5e9) !important;
}

/* ── Sidebar headers ── */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important; font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Progress helper
# ─────────────────────────────────────────────────────────────────────────────

class ProgressReporter:
    """Thread-safe progress bar + label updater."""
    def __init__(self, bar, label_slot, step_name: str, weight: float, offset: float):
        self.bar   = bar
        self.slot  = label_slot
        self.name  = step_name
        self.w     = weight
        self.off   = offset
        self._lock = Lock()

    def __call__(self, done: int, total: int):
        if total == 0:
            return
        frac = done / total
        pct  = min(self.off + frac * self.w, 1.0)
        with self._lock:
            self.bar.progress(pct)
            self.slot.markdown(
                f'<div class="step-badge active">⚡ {self.name} — {int(frac*100)}%</div>',
                unsafe_allow_html=True,
            )

    def done(self):
        with self._lock:
            self.slot.markdown(
                f'<div class="step-badge done">✅ {self.name} — done</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar and return a flat config dict."""
    cfg: dict = {}

    st.sidebar.markdown("## ⚙️ Configuration")

    # ── Text column ──────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📋 Data")
    cfg["text_col"] = st.sidebar.text_input(
        "Ticket text column *",
        value="ticket_context",
        help="Column that contains the ticket description / chat / query.",
    )

    # ── Embedding model ──────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔢 Embedding Model")
    cfg["embed_model"] = st.sidebar.selectbox(
        "Sentence-transformer",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        help="all-MiniLM-L6-v2 is fastest (~80 MB). all-mpnet-base-v2 is higher quality.",
    )

    # ── UMAP / HDBSCAN ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 📐 UMAP")
    cfg["umap_neighbors"] = st.sidebar.slider("n_neighbors", 5, 50, 15)
    cfg["umap_min_dist"]  = st.sidebar.slider("min_dist", 0.0, 0.5, 0.1, step=0.05)

    st.sidebar.markdown("### 🔵 HDBSCAN")
    cfg["hdb_min_cluster"] = st.sidebar.slider(
        "min_cluster_size", 2, 20, 3,
        help="Minimum tickets to form a cluster. Lower = more (smaller) clusters.",
    )
    cfg["hdb_min_samples"] = st.sidebar.slider(
        "min_samples", 1, 10, 2,
        help="Higher = more conservative core points (more noise points labelled -1).",
    )

    # ── LLM backend ──────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 LLM — Label Assignment")
    st.sidebar.caption(
        "The LLM reads ticket conversations and assigns "
        "**CorrectCategory** (e.g. Python, Azure, SQL…) and "
        "**CorrectSubcategory** (e.g. Runtime Error, Build Issue…)."
    )

    backend_label = st.sidebar.selectbox(
        "Choose LLM backend",
        list(LLM_BACKENDS.values()),
        index=0,   # Azure OpenAI first (user's primary)
    )
    # Reverse-lookup key from label
    cfg["backend"] = next(k for k, v in LLM_BACKENDS.items() if v == backend_label)

    is_paid = cfg["backend"] in PAID_BACKENDS
    is_free = cfg["backend"] in FREE_BACKENDS

    if is_free:
        st.sidebar.markdown('<span class="badge-free">✅ FREE — no key needed</span>', unsafe_allow_html=True)
        cfg["hf_token"] = st.sidebar.text_input(
            "HuggingFace token (optional)",
            type="password",
            help="Increases rate limits. Leave blank to use the free public endpoint.",
        )
        # Defaults for unused paid params
        cfg.update({"azure_api_key": "", "azure_endpoint": "", "azure_deployment": "gpt-4o",
                     "openai_api_key": "", "openai_model": "gpt-4o-mini",
                     "claude_api_key": "", "claude_model": "claude-haiku-4-5-20251001",
                     "gemini_api_key": "", "gemini_model": "gemini-2.5-flash"})

    else:
        st.sidebar.markdown('<span class="badge-paid">💳 PAID — API key required</span>', unsafe_allow_html=True)
        cfg["hf_token"] = ""

        if cfg["backend"] == "azure_openai":
            cfg["azure_api_key"]    = st.sidebar.text_input("Azure OpenAI API Key *", type="password")
            cfg["azure_endpoint"]   = st.sidebar.text_input(
                "Azure Endpoint *",
                placeholder="https://YOUR-RESOURCE.openai.azure.com/",
            )
            cfg["azure_deployment"] = st.sidebar.text_input(
                "Deployment name *",
                value="gpt-4o",
                help="The deployment name you used when deploying gpt-4o in Azure.",
            )
            cfg.update({"openai_api_key": "", "openai_model": "gpt-4o-mini",
                         "claude_api_key": "", "claude_model": "claude-haiku-4-5-20251001",
                         "gemini_api_key": "", "gemini_model": "gemini-2.5-flash"})

        elif cfg["backend"] == "openai":
            cfg["openai_api_key"] = st.sidebar.text_input("OpenAI API Key *", type="password")
            cfg["openai_model"]   = st.sidebar.selectbox(
                "Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            )
            cfg.update({"azure_api_key": "", "azure_endpoint": "", "azure_deployment": "gpt-4o",
                         "claude_api_key": "", "claude_model": "claude-haiku-4-5-20251001",
                         "gemini_api_key": "", "gemini_model": "gemini-2.5-flash"})

        elif cfg["backend"] == "claude":
            cfg["claude_api_key"] = st.sidebar.text_input("Anthropic API Key *", type="password")
            cfg["claude_model"]   = st.sidebar.selectbox(
                "Model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]
            )
            cfg.update({"azure_api_key": "", "azure_endpoint": "", "azure_deployment": "gpt-4o",
                         "openai_api_key": "", "openai_model": "gpt-4o-mini",
                         "gemini_api_key": "", "gemini_model": "gemini-2.5-flash"})

        elif cfg["backend"] == "gemini":
            cfg["gemini_api_key"] = st.sidebar.text_input("Google API Key *", type="password")
            cfg["gemini_model"]   = st.sidebar.selectbox(
                "Model", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            )
            cfg.update({"azure_api_key": "", "azure_endpoint": "", "azure_deployment": "gpt-4o",
                         "openai_api_key": "", "openai_model": "gpt-4o-mini",
                         "claude_api_key": "", "claude_model": "claude-haiku-4-5-20251001"})

    cfg["llm_samples"] = st.sidebar.slider(
        "Tickets sent to LLM per cluster", 3, 10, 5,
        help="More = better LLM context; slightly slower / more API tokens used.",
    )

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Cluster scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def render_scatter(coords_2d: np.ndarray, cluster_labels: np.ndarray,
                   cluster_info: list[dict], df_raw: pd.DataFrame,
                   text_col: str) -> None:
    """Interactive UMAP scatter plot coloured by cluster / category."""

    id_to_cat = {r["Cluster"]: r["CorrectCategory"]    for r in cluster_info}
    id_to_sub = {r["Cluster"]: r["CorrectSubcategory"]  for r in cluster_info}

    unique = sorted(set(cluster_labels))
    palette = [
        "#00e5c0","#0ea5e9","#f59e0b","#ef4444","#a855f7","#ec4899",
        "#22d3ee","#84cc16","#f97316","#6366f1","#14b8a6","#eab308",
        "#8b5cf6","#06b6d4","#d946ef","#fb923c","#4ade80","#f43f5e",
    ]
    color_map = {
        lbl: ("#4b5563" if lbl == -1 else palette[i % len(palette)])
        for i, lbl in enumerate(unique)
    }

    col_vals = df_raw[text_col].tolist() if text_col in df_raw.columns else [""] * len(df_raw)

    traces = []
    for lbl in unique:
        mask = cluster_labels == lbl
        cat  = id_to_cat.get(int(lbl), "?")
        sub  = id_to_sub.get(int(lbl), "?")
        name = "Noise" if lbl == -1 else f"C{lbl}: {cat}"

        hover_txts = [
            f"<b>Cluster {lbl}</b> | {cat}<br>"
            f"<i>{sub}</i><br><br>"
            f"{str(col_vals[i])[:120]}…"
            for i, m in enumerate(mask) if m
        ]

        traces.append(go.Scatter(
            x=coords_2d[mask, 0], y=coords_2d[mask, 1],
            mode="markers",
            name=name,
            marker=dict(
                color=color_map[lbl],
                size=9 if lbl != -1 else 5,
                opacity=0.85 if lbl != -1 else 0.35,
                line=dict(width=0.5, color="#000"),
            ),
            text=hover_txts,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(
            text="UMAP — tickets coloured by cluster & CorrectCategory",
            font=dict(family="JetBrains Mono", size=13, color="#e2e8f0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#080f1a", plot_bgcolor="#080f1a",
        font=dict(family="JetBrains Mono", color="#94a3b8"),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b", font_size=11),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Title ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="app-title">🎫 AI Ticket Classification Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">'
        'Upload → Clean → Embed → UMAP → HDBSCAN → LLM labels → Download'
        '</div>',
        unsafe_allow_html=True,
    )

    cfg = render_sidebar()

    # ── Pipeline steps legend ─────────────────────────────────────────────────
    with st.expander("📊 Pipeline Overview", expanded=False):
        for s in [
            "① Upload CSV / Excel file",
            "② Clean text  (remove emojis, URLs, punctuation, stopwords, tech-term normalization)",
            "③ Generate embeddings  (sentence-transformers)",
            "④ UMAP — 2-D reduction for visualisation",
            "⑤ HDBSCAN — unsupervised clustering on full embeddings",
            "⑥ LLM — reads ticket conversations → CorrectCategory + CorrectSubcategory",
            "⑦ Download augmented CSV / Excel",
        ]:
            st.markdown(f'<div class="step-badge">{s}</div>', unsafe_allow_html=True)

    st.divider()

    # ── STEP 1 — Upload ───────────────────────────────────────────────────────
    st.markdown("### 📂 Step 1 — Upload Dataset")

    col_up, col_dl = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload your ticket CSV or Excel file",
            type=["csv", "xlsx", "xls", "xlsb"],
            help=(
                f"Must contain a column named **{cfg['text_col']}** "
                "(or change the column name in the sidebar)."
            ),
        )
    with col_dl:
        sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_tickets.csv")
        if os.path.exists(sample):
            st.markdown("&nbsp;")
            with open(sample, "rb") as f:
                st.download_button(
                    "⬇ sample_tickets.csv",
                    data=f.read(),
                    file_name="sample_tickets.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    if uploaded is None:
        st.info("👆 Upload a file to get started, or download the sample CSV above.")
        return

    # Load
    try:
        df_raw = load_file(uploaded, uploaded.name)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return

    if cfg["text_col"] not in df_raw.columns:
        st.error(
            f"❌ Column **'{cfg['text_col']}'** not found.\n\n"
            f"Available columns: `{list(df_raw.columns)}`\n\n"
            "→ Update **Ticket text column** in the sidebar."
        )
        return

    st.success(f"✅ Loaded **{uploaded.name}** — {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    with st.expander("👁 Preview — first 5 rows"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    # ── Key validation: check api key present for paid backends ───────────────
    backend = cfg["backend"]
    key_missing = False
    if backend == "azure_openai" and (not cfg["azure_api_key"] or not cfg["azure_endpoint"]):
        st.warning("⚠️ Enter your **Azure OpenAI API Key** and **Endpoint** in the sidebar before running.")
        key_missing = True
    elif backend == "openai" and not cfg["openai_api_key"]:
        st.warning("⚠️ Enter your **OpenAI API Key** in the sidebar before running.")
        key_missing = True
    elif backend == "claude" and not cfg["claude_api_key"]:
        st.warning("⚠️ Enter your **Anthropic API Key** in the sidebar before running.")
        key_missing = True
    elif backend == "gemini" and not cfg["gemini_api_key"]:
        st.warning("⚠️ Enter your **Google API Key** in the sidebar before running.")
        key_missing = True

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown("### 🚀 Step 2 — Run Pipeline")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button(
            "▶  Run Classification",
            type="primary",
            use_container_width=True,
            disabled=key_missing,
        )
    with col_info:
        free_tag = "✅ Free" if backend in FREE_BACKENDS else "💳 Paid"
        st.caption(
            f"Model: `{cfg['embed_model']}` · "
            f"LLM: `{backend}` ({free_tag}) · "
            f"HDBSCAN min_cluster={cfg['hdb_min_cluster']}"
        )

    if not run_btn:
        return

    # ── Progress setup ────────────────────────────────────────────────────────
    prog_bar = st.progress(0.0)
    prog_slot = st.empty()

    WEIGHTS = {"clean": 0.08, "embed": 0.48, "umap_hdb": 0.14, "llm": 0.30}

    rp_clean = ProgressReporter(prog_bar, prog_slot, "② Text Cleaning",
                                 WEIGHTS["clean"], 0.0)
    rp_embed = ProgressReporter(prog_bar, prog_slot, "③ Generating Embeddings",
                                 WEIGHTS["embed"], WEIGHTS["clean"])
    rp_llm   = ProgressReporter(prog_bar, prog_slot, "⑥ LLM Labelling",
                                 WEIGHTS["llm"], WEIGHTS["clean"] + WEIGHTS["embed"] + WEIGHTS["umap_hdb"])

    def _mark(msg: str, pct: float):
        prog_bar.progress(min(pct, 1.0))
        prog_slot.markdown(f'<div class="step-badge active">⚡ {msg}</div>', unsafe_allow_html=True)

    # ── Execute pipeline ──────────────────────────────────────────────────────
    try:
        # STEP 2 — Clean
        df_clean = clean_column(df_raw, cfg["text_col"], progress_cb=rp_clean)
        rp_clean.done()

        # STEP 3 — Embed
        embeddings = generate_embeddings(
            df_clean["__cleaned__"].tolist(),
            model_name=cfg["embed_model"],
            progress_cb=rp_embed,
        )
        rp_embed.done()

        # STEP 4 — UMAP (2-D visualisation)
        _mark("④ UMAP — reducing dimensions…", WEIGHTS["clean"] + WEIGHTS["embed"])
        coords_2d = run_umap(
            embeddings,
            n_neighbors=cfg["umap_neighbors"],
            min_dist=cfg["umap_min_dist"],
        )

        # STEP 5 — HDBSCAN (on full embeddings for better clustering quality)
        _mark("⑤ HDBSCAN — clustering…", WEIGHTS["clean"] + WEIGHTS["embed"] + 0.07)
        cluster_labels = run_hdbscan(
            embeddings,
            min_cluster_size=cfg["hdb_min_cluster"],
            min_samples=cfg["hdb_min_samples"],
        )

        # STEP 6 — LLM labelling
        llm_kwargs = {
            "azure_api_key":    cfg.get("azure_api_key", ""),
            "azure_endpoint":   cfg.get("azure_endpoint", ""),
            "azure_deployment": cfg.get("azure_deployment", "gpt-4o"),
            "openai_api_key":   cfg.get("openai_api_key", ""),
            "openai_model":     cfg.get("openai_model", "gpt-4o-mini"),
            "claude_api_key":   cfg.get("claude_api_key", ""),
            "claude_model":     cfg.get("claude_model", "claude-haiku-4-5-20251001"),
            "gemini_api_key":   cfg.get("gemini_api_key", ""),
            "gemini_model":     cfg.get("gemini_model", "gemini-2.5-flash"),
            "hf_token":         cfg.get("hf_token", ""),
        }

        df_result, cluster_info, llm_errors = label_clusters(
            df=df_raw,                    # pass RAW text to LLM, not cleaned
            text_col=cfg["text_col"],
            cluster_labels=cluster_labels,
            backend=backend,
            llm_kwargs=llm_kwargs,
            samples_per_cluster=cfg["llm_samples"],
            progress_cb=rp_llm,
        )
        rp_llm.done()

        # Remove internal column if accidentally included
        df_result = df_result.drop(columns=["__cleaned__"], errors="ignore")

        prog_bar.progress(1.0)
        prog_slot.markdown(
            '<div class="step-badge done">✅ All steps complete!</div>',
            unsafe_allow_html=True,
        )

    except Exception:
        st.error("❌ Pipeline failed:")
        st.code(traceback.format_exc())
        return

    # ── LLM warnings ─────────────────────────────────────────────────────────
    for e in llm_errors:
        st.markdown(f'<div class="err-box">🔴 {e}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Results")

    n_clusters   = len([c for c in cluster_info if c["Cluster"] != -1])
    n_noise      = int((cluster_labels == -1).sum())
    n_classified = len(df_result) - n_noise

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{len(df_result):,}",  "Total Tickets"),
        (m2, str(n_clusters),         "Clusters Found"),
        (m3, f"{n_classified:,}",     "Tickets Labelled"),
        (m4, str(n_noise),            "Noise / Unclustered"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── UMAP scatter plot ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗺️ UMAP Cluster Visualisation")
    render_scatter(coords_2d, cluster_labels, cluster_info, df_raw, cfg["text_col"])

    # ── Cluster summary table ─────────────────────────────────────────────────
    st.markdown("### 🗂️ Cluster Summary")
    st.dataframe(
        pd.DataFrame(cluster_info),
        use_container_width=True,
        height=280,
    )

    # ── Output dataset preview ────────────────────────────────────────────────
    st.markdown("### 📋 Output Dataset (first 20 rows)")
    preview_cols = [c for c in [cfg["text_col"], "CorrectCategory", "CorrectSubcategory"]
                    if c in df_result.columns]
    st.dataframe(df_result[preview_cols].head(20), use_container_width=True)

    st.divider()

    # ── STEP 7 — Download ─────────────────────────────────────────────────────
    st.markdown("### ⬇️ Step 7 — Download Results")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📄 Download as CSV",
            data=export_file(df_result, "csv"),
            file_name="classified_tickets.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📊 Download as Excel",
            data=export_file(df_result, "excel"),
            file_name="classified_tickets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
