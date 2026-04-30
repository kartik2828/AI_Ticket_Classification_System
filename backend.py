"""
backend.py  —  Full pipeline logic.

Steps (in order):
  1. load_file()          — read CSV / Excel into DataFrame
  2. clean_text()         — strip noise from ticket text column
  3. generate_embeddings()— sentence-transformers → dense vectors
  4. run_umap()           — 2-D reduction for visualisation
  5. run_hdbscan()        — unsupervised clustering on embeddings
  6. label_with_llm()     — LLM reads tickets, assigns Category + Subcategory
  7. export_file()        — serialize augmented DataFrame to bytes

LLM backends supported:
  Paid  : Azure OpenAI, OpenAI, Anthropic Claude, Google Gemini
  Free  : HuggingFace Zephyr-7B, HuggingFace Mistral-7B
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from prompts import SYSTEM_PROMPT, build_user_prompt, VALID_CATEGORIES, VALID_SUBCATEGORIES


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — File loading
# ═══════════════════════════════════════════════════════════════════════════

def load_file(file_obj, filename: str) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a DataFrame.
    Column names are stripped of surrounding whitespace.
    Raises ValueError for unsupported formats.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_obj, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif ext == ".xlsb":
        df = pd.read_excel(file_obj, engine="pyxlsb")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Upload CSV or Excel.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def export_file(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """Serialize DataFrame to bytes. fmt = 'csv' or 'excel'."""
    buf = io.BytesIO()
    if fmt == "excel":
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Classified Tickets")
    else:
        df.to_csv(buf, index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Text cleaning
# ═══════════════════════════════════════════════════════════════════════════

# Emoji regex pattern (broad Unicode ranges)
_EMOJI_RE = re.compile(
    "[\U00010000-\U0010FFFF"   # supplementary multilingual plane
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# Tech-term normalizations: protect key terms before punctuation removal
_TECH_NORM: list[tuple[str, str]] = [
    (r"\bc\+\+\b",               "cplusplus"),
    (r"\bc#\b",                  "csharp"),
    (r"\b\.net\b",               "dotnet"),
    (r"\bnode\.?js\b",           "nodejs"),
    (r"\bvue\.?js\b",            "vuejs"),
    (r"\btype\s*script\b",       "typescript"),
    (r"\bjava\s*script\b",       "javascript"),
    (r"\bpow(?:er)?\s*bi\b",     "powerbi"),
    (r"\bpow(?:er)?\s*shell\b",  "powershell"),
    (r"\bk8s\b",                 "kubernetes"),
    (r"\bscikit[\s\-]learn\b",   "scikitlearn"),
    (r"\btensor\s*flow\b",       "tensorflow"),
    (r"\bpy\s*torch\b",          "pytorch"),
    (r"\bspring\s*boot\b",       "springboot"),
    (r"\brest\s*api\b",          "restapi"),
    (r"\bgraph\s*ql\b",          "graphql"),
    (r"\bci[/\s]?cd\b",          "cicd"),
    (r"\baws\b",                 "aws"),
    (r"\bgcp\b",                 "gcp"),
    (r"\bnull\s*pointer\b",      "nullpointerexception"),
    (r"\bnpe\b",                 "nullpointerexception"),
    (r"\bhelm\s*chart\b",        "helmchart"),
    (r"\bpow(?:er)?\s*bi\b",     "powerbi"),
]

def _get_stopwords() -> set[str]:
    """Return NLTK English stopwords, downloading if needed."""
    import nltk
    for res, kind in [("stopwords", "corpora"), ("punkt", "tokenizers")]:
        try:
            nltk.data.find(f"{kind}/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


_STOPWORDS: Optional[set[str]] = None  # loaded once on first call


def clean_text(text: str) -> str:
    """
    Clean a single ticket string:
      - Remove emojis
      - Remove URLs and email addresses
      - Remove ticket IDs (INC-001, TKT#123, …)
      - Normalize tech terms (C++ → cplusplus, Node.js → nodejs, …)
      - Lowercase
      - Remove punctuation / special characters
      - Collapse extra whitespace
      - Remove English stopwords (keep tech-relevant short tokens)
    Returns a clean string suitable for embedding.
    """
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = _get_stopwords()

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # 1. Remove emojis
    text = _EMOJI_RE.sub(" ", text)

    # 2. Remove URLs  (http/https/www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 4. Remove common ticket ID patterns (INC-001, TKT#123, REQ 456, #789)
    text = re.sub(
        r"\b(?:inc|req|tkt|ticket|issue|bug|task|story)[\s\-#]*\d+\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"#\d+", " ", text)

    # 5. Lowercase before tech normalizations
    text = text.lower()

    # 6. Tech-term normalization (must happen before punctuation removal)
    for pattern, replacement in _TECH_NORM:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 7. Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Stopword removal — keep tokens > 2 chars not in stopword list
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(tokens)


def clean_column(df: pd.DataFrame, col: str,
                 progress_cb: Optional[Callable] = None) -> pd.DataFrame:
    """
    Apply clean_text() to every row in `col`.
    Adds a '__cleaned__' column; keeps the original column untouched.
    """
    df = df.copy()
    df[col] = df[col].fillna("")
    total = len(df)
    cleaned: list[str] = []

    for i, val in enumerate(df[col]):
        cleaned.append(clean_text(val))
        if progress_cb and i % 20 == 0:
            progress_cb(i + 1, total)

    if progress_cb:
        progress_cb(total, total)

    df["__cleaned__"] = cleaned
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Embeddings
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings(texts: list[str],
                         model_name: str = "all-MiniLM-L6-v2",
                         batch_size: int = 64,
                         progress_cb: Optional[Callable] = None) -> np.ndarray:
    """
    Encode cleaned texts with a sentence-transformer model.
    Returns ndarray of shape (n, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    total = len(texts)
    parts: list[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        parts.append(vecs)
        if progress_cb:
            progress_cb(min(i + batch_size, total), total)

    return np.vstack(parts)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — UMAP  (2-D for visualisation)
# ═══════════════════════════════════════════════════════════════════════════

def run_umap(embeddings: np.ndarray,
             n_neighbors: int = 15,
             min_dist: float = 0.1,
             random_state: int = 42) -> np.ndarray:
    """
    Reduce embeddings to 2-D using UMAP for scatter-plot visualisation.
    Returns ndarray of shape (n, 2).
    """
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — HDBSCAN  (clustering on embeddings, not UMAP coords)
# ═══════════════════════════════════════════════════════════════════════════

def run_hdbscan(embeddings: np.ndarray,
                min_cluster_size: int = 3,
                min_samples: int = 2) -> np.ndarray:
    """
    Cluster using HDBSCAN on the raw embeddings (not 2-D UMAP coords).
    Using embeddings directly gives better cluster quality.
    Returns integer label array; -1 = noise / unclustered.
    """
    import hdbscan  # type: ignore

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return clusterer.fit_predict(embeddings)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — LLM labelling
# ═══════════════════════════════════════════════════════════════════════════

# ── JSON response parser ────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> tuple[str, str]:
    """
    Robustly extract (CorrectCategory, CorrectSubcategory) from LLM output.
    Handles markdown fences, stray text, and malformed JSON.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        return (
            str(obj.get("CorrectCategory",    "Other")).strip(),
            str(obj.get("CorrectSubcategory", "Other")).strip(),
        )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Regex fallback
    cat = re.search(r'"CorrectCategory"\s*:\s*"([^"]+)"', text)
    sub = re.search(r'"CorrectSubcategory"\s*:\s*"([^"]+)"', text)
    return (
        cat.group(1).strip() if cat else "Other",
        sub.group(1).strip() if sub else "Other",
    )


# ── Individual LLM callers ──────────────────────────────────────────────────

def _call_azure_openai(tickets: list[str], api_key: str,
                        endpoint: str, deployment: str) -> tuple[str, str]:
    """Azure OpenAI (e.g. gpt-4o deployment)."""
    from openai import AzureOpenAI  # type: ignore

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-01",
    )
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(tickets)},
        ],
        temperature=0,
        max_tokens=80,
    )
    return _parse_llm_json(resp.choices[0].message.content or "")


def _call_openai(tickets: list[str], api_key: str,
                  model: str = "gpt-4o-mini") -> tuple[str, str]:
    """Standard OpenAI API."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(tickets)},
        ],
        temperature=0,
        max_tokens=80,
        response_format={"type": "json_object"},
    )
    return _parse_llm_json(resp.choices[0].message.content or "")


def _call_claude(tickets: list[str], api_key: str,
                  model: str = "claude-haiku-4-5-20251001") -> tuple[str, str]:
    """Anthropic Claude API."""
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=80,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(tickets)}],
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse_llm_json(raw)


def _call_gemini(tickets: list[str], api_key: str,
                  model: str = "gemini-2.5-flash") -> tuple[str, str]:
    """Google Gemini API."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0, "max_output_tokens": 80},
    )
    resp = gm.generate_content(build_user_prompt(tickets))
    return _parse_llm_json(resp.text if hasattr(resp, "text") else "")


def _call_hf_free(tickets: list[str], model_id: str,
                   hf_token: str = "") -> tuple[str, str]:
    """
    HuggingFace Inference API — free tier (no local GPU needed).
    model_id examples:
      'HuggingFaceH4/zephyr-7b-beta'
      'mistralai/Mistral-7B-Instruct-v0.3'
    """
    import requests  # type: ignore

    user_msg = build_user_prompt(tickets)
    # Universal chat-template format (works for Zephyr and Mistral)
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_msg}\n"
        f"<|assistant|>\n"
    )

    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.1,
            "return_full_text": False,
            "do_sample": False,
        },
    }

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    for attempt in range(2):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 503:          # model loading
            time.sleep(20)
            continue
        r.raise_for_status()
        data = r.json()
        break

    raw = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
    return _parse_llm_json(raw)


# ── Public dispatcher ───────────────────────────────────────────────────────

# Registry: backend_key → human label
LLM_BACKENDS: dict[str, str] = {
    "azure_openai": "Azure OpenAI  (gpt-4o deployment) — Recommended",
    "openai":       "OpenAI  (gpt-4o-mini / gpt-4o)",
    "claude":       "Anthropic Claude  (claude-3-haiku / sonnet)",
    "gemini":       "Google Gemini  (gemini-2.5-flash)",
    "hf_zephyr":    "Zephyr-7B  [FREE — HuggingFace]",
    "hf_mistral":   "Mistral-7B  [FREE — HuggingFace]",
}

PAID_BACKENDS = {"azure_openai", "openai", "claude", "gemini"}
FREE_BACKENDS  = {"hf_zephyr", "hf_mistral"}


def call_llm(
    tickets: list[str],
    backend: str,
    # Azure OpenAI
    azure_api_key: str = "",
    azure_endpoint: str = "",
    azure_deployment: str = "gpt-4o",
    # OpenAI
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    # Claude
    claude_api_key: str = "",
    claude_model: str = "claude-haiku-4-5-20251001",
    # Gemini
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    # HuggingFace
    hf_token: str = "",
) -> tuple[str, str]:
    """
    Route to the selected LLM backend and return (CorrectCategory, CorrectSubcategory).
    All exceptions are caught and re-raised with a user-friendly message.
    """
    try:
        if backend == "azure_openai":
            return _call_azure_openai(tickets, azure_api_key, azure_endpoint, azure_deployment)
        elif backend == "openai":
            return _call_openai(tickets, openai_api_key, openai_model)
        elif backend == "claude":
            return _call_claude(tickets, claude_api_key, claude_model)
        elif backend == "gemini":
            return _call_gemini(tickets, gemini_api_key, gemini_model)
        elif backend == "hf_zephyr":
            return _call_hf_free(tickets, "HuggingFaceH4/zephyr-7b-beta", hf_token)
        elif backend == "hf_mistral":
            return _call_hf_free(tickets, "mistralai/Mistral-7B-Instruct-v0.3", hf_token)
        else:
            return "Other", "Other"
    except Exception as exc:
        raise RuntimeError(f"[LLM:{backend}] {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator — label all clusters
# ═══════════════════════════════════════════════════════════════════════════

def label_clusters(
    df: pd.DataFrame,
    text_col: str,
    cluster_labels: np.ndarray,
    backend: str,
    llm_kwargs: dict,
    samples_per_cluster: int = 5,
    progress_cb: Optional[Callable] = None,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """
    For every unique cluster discovered by HDBSCAN:
      1. Pick up to `samples_per_cluster` raw ticket texts.
      2. Send them to the LLM → (CorrectCategory, CorrectSubcategory).
      3. Write those labels to every row in that cluster.

    Returns:
      df             — augmented DataFrame with CorrectCategory + CorrectSubcategory
      cluster_info   — list of dicts (one per cluster) for the summary table
      errors         — list of error strings for clusters that failed
    """
    df = df.copy()
    df["CorrectCategory"]    = "Pending"
    df["CorrectSubcategory"] = "Pending"

    unique_ids = sorted(set(cluster_labels))
    cluster_info: list[dict] = []
    errors: list[str] = []
    total = len(unique_ids)

    for idx, cid in enumerate(unique_ids):
        mask     = cluster_labels == cid
        raw_txts = df.loc[mask, text_col].tolist()

        if cid == -1:
            # Noise — skip LLM, mark for manual review
            cat, sub = "Needs Review", "Unclustered"
        else:
            try:
                cat, sub = call_llm(
                    raw_txts[:samples_per_cluster],
                    backend=backend,
                    **llm_kwargs,
                )
            except RuntimeError as e:
                errors.append(str(e))
                cat, sub = "LLM Error", "LLM Error"

        df.loc[mask, "CorrectCategory"]    = cat
        df.loc[mask, "CorrectSubcategory"] = sub

        cluster_info.append({
            "Cluster":             int(cid),
            "Size":                int(mask.sum()),
            "CorrectCategory":     cat,
            "CorrectSubcategory":  sub,
            "Sample Ticket":       (raw_txts[0][:120] + "…") if raw_txts else "",
        })

        if progress_cb:
            progress_cb(idx + 1, total)

    return df, cluster_info, errors
