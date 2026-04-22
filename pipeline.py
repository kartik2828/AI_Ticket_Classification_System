"""
pipeline.py — Ticket classification pipeline.

Exact steps (matching the attached architecture diagram):
  1. User Upload (handled in main.py)
  2. Data Validation
  3. Text Cleaning
  4. Embedding Layer  (sentence-transformers)
  5. Dimensionality Reduction  (UMAP)
  6. HDBSCAN Clustering
  7. LLM Model for Labelling  → CorrectCategory + CorrectSubcategory
  8. Output

Each step is an isolated function for easy testing and extension.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import utils
from models.llm_labeler import LLMLabeler


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all tunable knobs in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Column names ──────────────────────────────────────────────────────────
    text_column: str        = "ticket_context"
    category_column: str    = "category"        # optional, used for context enrichment
    subcategory_column: str = "Subcategory"     # optional

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str    = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # ── UMAP — 2D (visualization) ─────────────────────────────────────────────
    umap_2d_n_neighbors: int  = 15
    umap_2d_min_dist: float   = 0.1

    # ── UMAP — cluster-space (higher-dim preserves more structure) ────────────
    umap_cluster_n_components: int = 5
    umap_cluster_n_neighbors: int  = 15
    umap_cluster_min_dist: float   = 0.0

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    hdbscan_min_cluster_size: int = 3
    hdbscan_min_samples: int      = 2

    # ── LLM ───────────────────────────────────────────────────────────────────
    # backend: chatgpt | claude | gemini | zephyr | mistral | flan_t5
    llm_backend: str         = "flan_t5"
    llm_api_key: str         = ""     # for paid backends
    llm_hf_token: str        = ""     # optional HF token for free backends
    llm_samples_per_cluster: int = 5  # how many sample tickets to send per cluster

    # ── Misc ──────────────────────────────────────────────────────────────────
    random_state: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    df: pd.DataFrame
    embeddings: Optional[np.ndarray]        = None
    reduced_2d: Optional[np.ndarray]        = None   # for scatter plot
    cluster_labels: Optional[np.ndarray]    = None
    cluster_summary: list[dict]             = field(default_factory=list)
    timing: dict[str, float]                = field(default_factory=dict)
    warnings: list[str]                     = field(default_factory=list)
    llm_errors: list[str]                   = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def step_validate(df: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    """
    Validate required columns exist and data is non-empty.
    Returns a list of warning messages (empty list = all OK).
    Raises ValueError for hard failures (missing required column).
    """
    warnings: list[str] = []

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    missing = utils.validate_columns(df, [cfg.text_column])
    if missing:
        raise ValueError(
            f"Required column '{cfg.text_column}' not found. "
            f"Available columns: {list(df.columns)}. "
            "Please update the Text Column name in the sidebar."
        )

    for opt_col in [cfg.category_column, cfg.subcategory_column]:
        if opt_col and opt_col not in df.columns:
            warnings.append(
                f"Optional column '{opt_col}' not found — "
                "skipping category-hint enrichment for that field."
            )

    null_count = df[cfg.text_column].isnull().sum()
    if null_count > 0:
        warnings.append(
            f"{null_count} row(s) have empty '{cfg.text_column}' — "
            "they will be treated as empty strings."
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def step_clean(df: pd.DataFrame, cfg: PipelineConfig,
               progress_cb: Optional[Callable] = None) -> pd.DataFrame:
    """
    Clean and normalize ticket text.
    Adds internal columns: __combined_text__, __cleaned_text__
    """
    stop_words = utils._get_nltk_resources()
    df = df.copy()
    df[cfg.text_column] = df[cfg.text_column].fillna("")

    # Build combined text = ticket body + optional category hints
    df["__combined_text__"] = df.apply(
        lambda row: utils.build_combined_text(
            row,
            cfg.text_column,
            cfg.category_column if cfg.category_column in df.columns else None,
            cfg.subcategory_column if cfg.subcategory_column in df.columns else None,
        ),
        axis=1,
    )

    total = len(df)
    cleaned: list[str] = []
    for i, raw in enumerate(df["__combined_text__"]):
        cleaned.append(utils.clean_text(raw, stop_words))
        if progress_cb and i % 20 == 0:
            progress_cb(i + 1, total)

    df["__cleaned_text__"] = cleaned
    if progress_cb:
        progress_cb(total, total)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Embedding Layer
# ─────────────────────────────────────────────────────────────────────────────

def step_embed(df: pd.DataFrame, cfg: PipelineConfig,
               progress_cb: Optional[Callable] = None) -> np.ndarray:
    """Convert cleaned text to dense vector embeddings."""
    texts = df["__cleaned_text__"].tolist()
    return utils.generate_embeddings(
        texts,
        model_name=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
        progress_callback=progress_cb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Dimensionality Reduction (UMAP)
# ─────────────────────────────────────────────────────────────────────────────

def step_umap_2d(embeddings: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """2D UMAP — used for interactive visualization only."""
    return utils.reduce_dimensions(
        embeddings,
        n_components=2,
        n_neighbors=cfg.umap_2d_n_neighbors,
        min_dist=cfg.umap_2d_min_dist,
        random_state=cfg.random_state,
    )


def step_umap_cluster(embeddings: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Higher-dimensional UMAP — used as input to HDBSCAN.
    More components = better cluster separation than raw embeddings.
    """
    n = min(cfg.umap_cluster_n_components, embeddings.shape[1])
    return utils.reduce_dimensions(
        embeddings,
        n_components=n,
        n_neighbors=cfg.umap_cluster_n_neighbors,
        min_dist=cfg.umap_cluster_min_dist,
        random_state=cfg.random_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — HDBSCAN Clustering
# ─────────────────────────────────────────────────────────────────────────────

def step_cluster(reduced: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Cluster the UMAP-reduced vectors. Returns integer labels (-1 = noise)."""
    return utils.cluster_embeddings(
        reduced,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — LLM Model for Labelling
# ─────────────────────────────────────────────────────────────────────────────

def step_label(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cfg: PipelineConfig,
    progress_cb: Optional[Callable] = None,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """
    For each cluster:
      1. Gather representative raw ticket texts.
      2. Send them to the LLM (or heuristic fallback) to infer:
            CorrectCategory   — the TECHNOLOGY / SKILL area (Python, React, Azure…)
            CorrectSubcategory — the issue type (Runtime Error, Build/Dep, Performance…)
      3. Assign those labels back to every row in the cluster.

    Returns: (augmented_df, cluster_summary_list, error_messages_list)
    """
    df = df.copy()
    df["CorrectCategory"]    = "Uncategorized"
    df["CorrectSubcategory"] = "Uncategorized"

    labeler = LLMLabeler(
        backend=cfg.llm_backend,
        api_key=cfg.llm_api_key,
        hf_token=cfg.llm_hf_token,
    )

    unique_clusters = sorted(set(cluster_labels))
    cluster_summary: list[dict] = []
    llm_errors: list[str] = []
    total = len(unique_clusters)

    for idx, cluster_id in enumerate(unique_clusters):
        mask          = cluster_labels == cluster_id
        cleaned_texts = df.loc[mask, "__cleaned_text__"].tolist()
        raw_texts     = (
            df.loc[mask, cfg.text_column].tolist()
            if cfg.text_column in df.columns else cleaned_texts
        )

        # ── Label inference ────────────────────────────────────────────────
        if cluster_id == -1:
            # Noise cluster — heuristic only, mark with [Review] prefix
            category, subcategory = utils.infer_labels_heuristic(-1, cleaned_texts)
            label_source = "heuristic (noise)"
        else:
            # Send RAW (not cleaned) tickets to LLM so it reads the real language
            category, subcategory = labeler.label(raw_texts[:cfg.llm_samples_per_cluster])
            label_source = cfg.llm_backend

            if category == "LLM-Error":
                # LLM failed → fallback to heuristic silently
                category, subcategory = utils.infer_labels_heuristic(cluster_id, cleaned_texts)
                llm_errors.append(f"Cluster {cluster_id}: LLM call failed, used heuristic fallback.")
                label_source = "heuristic (LLM-fallback)"

        # ── Assign to every row in this cluster ───────────────────────────
        df.loc[mask, "CorrectCategory"]    = category
        df.loc[mask, "CorrectSubcategory"] = subcategory

        # ── Build summary entry ───────────────────────────────────────────
        top_terms = utils.extract_top_terms(cleaned_texts, top_n=8)
        cluster_summary.append({
            "cluster_id":    int(cluster_id),
            "size":          int(mask.sum()),
            "category":      category,
            "subcategory":   subcategory,
            "top_terms":     ", ".join(top_terms),
            "sample_ticket": (raw_texts[0][:150] + "…") if raw_texts else "",
            "label_source":  label_source,
        })

        if progress_cb:
            progress_cb(idx + 1, total)

    return df, cluster_summary, llm_errors


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    cfg: Optional[PipelineConfig] = None,
    progress_callbacks: Optional[dict[str, Callable]] = None,
) -> PipelineResult:
    """
    Execute all pipeline steps in order and return a PipelineResult.

    progress_callbacks dict keys: 'clean' | 'embed' | 'label'
    Each callback signature: callback(current: int, total: int)
    """
    if cfg is None:
        cfg = PipelineConfig()
    if progress_callbacks is None:
        progress_callbacks = {}

    result  = PipelineResult(df=df)
    t_start = time.time()

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    t0 = time.time()
    result.warnings = step_validate(df, cfg)   # raises on hard error
    result.timing["1_validate"] = round(time.time() - t0, 3)

    # ── Step 3: Clean ─────────────────────────────────────────────────────────
    t0 = time.time()
    df_clean = step_clean(df, cfg, progress_cb=progress_callbacks.get("clean"))
    result.timing["2_clean"] = round(time.time() - t0, 3)

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    t0 = time.time()
    embeddings = step_embed(df_clean, cfg, progress_cb=progress_callbacks.get("embed"))
    result.embeddings = embeddings
    result.timing["3_embed"] = round(time.time() - t0, 3)

    # ── Step 5a: UMAP 2D (visualization) ─────────────────────────────────────
    t0 = time.time()
    result.reduced_2d = step_umap_2d(embeddings, cfg)
    result.timing["4a_umap_2d"] = round(time.time() - t0, 3)

    # ── Step 5b: UMAP cluster-space ───────────────────────────────────────────
    t0 = time.time()
    reduced_cluster = step_umap_cluster(embeddings, cfg)
    result.timing["4b_umap_cluster"] = round(time.time() - t0, 3)

    # ── Step 6: HDBSCAN ───────────────────────────────────────────────────────
    t0 = time.time()
    cluster_labels = step_cluster(reduced_cluster, cfg)
    result.cluster_labels = cluster_labels
    result.timing["5_hdbscan"] = round(time.time() - t0, 3)

    # ── Step 7: LLM labelling ─────────────────────────────────────────────────
    t0 = time.time()
    df_labeled, cluster_summary, llm_errors = step_label(
        df_clean, cluster_labels, cfg,
        progress_cb=progress_callbacks.get("label"),
    )
    result.cluster_summary = cluster_summary
    result.llm_errors      = llm_errors
    result.timing["6_label"] = round(time.time() - t0, 3)

    # ── Cleanup internal columns ──────────────────────────────────────────────
    df_labeled = df_labeled.drop(
        columns=["__combined_text__", "__cleaned_text__"], errors="ignore"
    )
    result.df = df_labeled
    result.timing["total"] = round(time.time() - t_start, 3)

    return result
