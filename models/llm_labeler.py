"""
models/llm_labeler.py — LLM-assisted ticket classification.

Supported backends (6 total):
  PAID (need API token):
    1. chatgpt    — OpenAI GPT-4o / GPT-3.5-turbo
    2. claude     — Anthropic Claude 3 Haiku / Sonnet
    3. gemini     — Google Gemini 1.5 Flash / Pro

  FREE (no token needed):
    4. hf_zephyr  — Zephyr-7B via HuggingFace free Inference API
    5. hf_mistral — Mistral-7B-Instruct via HuggingFace free Inference API
    6. hf_falcon  — Falcon-7B-Instruct via HuggingFace free Inference API

All backends receive the SAME carefully engineered prompt and return a
(CorrectCategory, CorrectSubcategory) tuple focused on TECHNOLOGY / SKILL AREA.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Master prompt — used by ALL backends
#
# Key design decisions:
#   • Tells the model the domain is TECHNOLOGY support tickets
#   • Lists all valid categories (tech skills) so the model picks from them
#   • Asks it to READ the conversation/context, not just pattern-match keywords
#   • Forces strict JSON output so we can parse it reliably
#   • Gives a worked example to anchor the format
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert IT and software engineering support analyst.
Your ONLY job is to read technology support ticket descriptions and classify them
into the correct technology category and subcategory.

Valid CorrectCategory values (pick the SINGLE best fit):
  Python | Java | JavaScript | React | Angular / Vue | TypeScript |
  Azure | AWS | GCP | Power BI | SQL / Database | C++ / C |
  DevOps / CI-CD | Machine Learning | API / Backend | Other

Valid CorrectSubcategory values (pick the SINGLE best fit):
  Runtime Error | Build / Dependency | Performance | Authentication / Auth |
  Data / Query Issue | Deployment / Config | UI / Rendering |
  Networking / CORS | Visualisation | Model / Algorithm | Other

Rules:
1. Base your decision on the FULL conversation and context of the ticket.
2. Focus on WHAT TECHNOLOGY the person is working with, not generic buckets.
3. If a ticket mentions Spring Boot or Maven → Java. React hooks → React. DAX measures → Power BI.
4. You MUST return ONLY a valid JSON object. No explanation. No markdown. No extra text.
5. The JSON must have exactly two keys: "CorrectCategory" and "CorrectSubcategory".

Example:
  Ticket: "My pandas DataFrame merge is producing duplicate rows with a left join."
  Output: {"CorrectCategory": "Python", "CorrectSubcategory": "Data / Query Issue"}"""

USER_PROMPT_TEMPLATE = """Classify the following support ticket cluster.
These {n} tickets were grouped together because they are semantically similar.
Read each one carefully and decide the single best (CorrectCategory, CorrectSubcategory) pair.

--- TICKETS ---
{tickets}
--- END ---

Return ONLY the JSON object:"""


def _build_user_prompt(sample_texts: list[str], max_samples: int = 5) -> str:
    samples = sample_texts[:max_samples]
    numbered = "\n".join(f"{i+1}. {t[:300]}" for i, t in enumerate(samples))
    return USER_PROMPT_TEMPLATE.format(n=len(samples), tickets=numbered)


def _parse_llm_response(raw: str) -> tuple[str, str]:
    """
    Robustly extract (CorrectCategory, CorrectSubcategory) from LLM output.
    Handles markdown fences, stray text, and partial responses.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Try direct JSON parse first
    try:
        data = json.loads(cleaned)
        return (
            str(data.get("CorrectCategory", "Uncategorized")).strip(),
            str(data.get("CorrectSubcategory", "Uncategorized")).strip(),
        )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex extraction
    cat_m = re.search(r'"CorrectCategory"\s*:\s*"([^"]+)"', cleaned)
    sub_m = re.search(r'"CorrectSubcategory"\s*:\s*"([^"]+)"', cleaned)
    category = cat_m.group(1).strip() if cat_m else "Uncategorized"
    subcategory = sub_m.group(1).strip() if sub_m else "Uncategorized"
    return category, subcategory


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 — OpenAI ChatGPT  (PAID)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_chatgpt(sample_texts: list[str],
                        api_key: str,
                        model: str = "gpt-4o-mini") -> tuple[str, str]:
    """
    Classify via OpenAI Chat Completions API.

    Models: gpt-4o, gpt-4o-mini (cheapest), gpt-3.5-turbo
    Requires: OPENAI_API_KEY
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(sample_texts)},
        ],
        temperature=0,
        max_tokens=80,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 — Anthropic Claude  (PAID)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_claude(sample_texts: list[str],
                       api_key: str,
                       model: str = "claude-haiku-4-5-20251001") -> tuple[str, str]:
    """
    Classify via Anthropic Messages API.

    Models: claude-haiku-4-5-20251001 (fast/cheap), claude-sonnet-4-6
    Requires: ANTHROPIC_API_KEY
    """
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=80,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(sample_texts)}],
    )
    raw = msg.content[0].text if msg.content else ""
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3 — Google Gemini  (PAID)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_gemini(sample_texts: list[str],
                       api_key: str,
                       model: str = "gemini-1.5-flash") -> tuple[str, str]:
    """
    Classify via Google Generative AI SDK.

    Models: gemini-1.5-flash (fast/cheap), gemini-1.5-pro
    Requires: GOOGLE_API_KEY
    """
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0, "max_output_tokens": 80},
    )
    prompt = _build_user_prompt(sample_texts)
    resp = gemini_model.generate_content(prompt)
    raw = resp.text if hasattr(resp, "text") else ""
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper — HuggingFace Inference API (free tier, no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def _call_hf_inference_api(model_id: str, prompt: str,
                             hf_token: Optional[str] = None,
                             max_new_tokens: int = 120) -> str:
    """
    Call HuggingFace Inference API (free public endpoint, rate-limited).
    Optionally accepts an HF token for higher limits.
    """
    import requests  # type: ignore

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "return_full_text": False,
            "do_sample": False,
        },
    }

    # Retry once if model is loading (503)
    for attempt in range(2):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 503:
            time.sleep(15)
            continue
        resp.raise_for_status()
        data = resp.json()
        break

    if isinstance(data, list) and data:
        return data[0].get("generated_text", "")
    return str(data)


def _build_hf_prompt(sample_texts: list[str]) -> str:
    """
    Build a prompt format suitable for instruction-tuned open-source models
    that don't use a separate system message.
    """
    user_part = _build_user_prompt(sample_texts)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_part}\n"
        f"<|assistant|>\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backend 4 — Zephyr-7B-beta  (FREE)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_zephyr(sample_texts: list[str],
                       hf_token: Optional[str] = None) -> tuple[str, str]:
    """
    Classify using HuggingFaceH4/zephyr-7b-beta — FREE via HF Inference API.
    No API key required (optional HF token for higher rate limits).
    """
    prompt = _build_hf_prompt(sample_texts)
    raw = _call_hf_inference_api("HuggingFaceH4/zephyr-7b-beta", prompt, hf_token)
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 5 — Mistral-7B-Instruct  (FREE)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_mistral(sample_texts: list[str],
                        hf_token: Optional[str] = None) -> tuple[str, str]:
    """
    Classify using mistralai/Mistral-7B-Instruct-v0.2 — FREE via HF Inference API.
    """
    # Mistral uses [INST] format
    user_part = _build_user_prompt(sample_texts)
    prompt = f"[INST] {SYSTEM_PROMPT}\n\n{user_part} [/INST]"
    raw = _call_hf_inference_api("mistralai/Mistral-7B-Instruct-v0.2", prompt, hf_token)
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 6 — Flan-T5-Large  (FREE, very fast)
# ─────────────────────────────────────────────────────────────────────────────

def label_with_flan_t5(sample_texts: list[str],
                        hf_token: Optional[str] = None) -> tuple[str, str]:
    """
    Classify using google/flan-t5-large — FREE via HF Inference API.
    Smallest and fastest free model; good for quick tests.
    """
    user_part = _build_user_prompt(sample_texts)
    # flan-t5 is a seq2seq model — no chat template needed
    prompt = f"{SYSTEM_PROMPT}\n\n{user_part}"
    raw = _call_hf_inference_api("google/flan-t5-large", prompt, hf_token, max_new_tokens=60)
    return _parse_llm_response(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Unified LLMLabeler class
# ─────────────────────────────────────────────────────────────────────────────

class LLMLabeler:
    """
    Single interface to all 6 supported LLM backends.

    Instantiate once per pipeline run; call .label(sample_texts) per cluster.
    Falls back to ('Uncategorized', 'Uncategorized') on any error so the
    pipeline never crashes due to API issues.
    """

    # Maps UI-facing name → internal key
    BACKEND_OPTIONS: dict[str, str] = {
        # Paid
        "ChatGPT (OpenAI) — needs token":   "chatgpt",
        "Claude (Anthropic) — needs token": "claude",
        "Gemini (Google) — needs token":    "gemini",
        # Free
        "Zephyr-7B (Free, no token)":       "zephyr",
        "Mistral-7B (Free, no token)":      "mistral",
        "Flan-T5-Large (Free, fastest)":    "flan_t5",
    }

    PAID_BACKENDS = {"chatgpt", "claude", "gemini"}
    FREE_BACKENDS = {"zephyr", "mistral", "flan_t5"}

    def __init__(self, backend: str = "flan_t5",
                 api_key: str = "",
                 hf_token: str = ""):
        """
        Args:
            backend:   Internal backend key (see BACKEND_OPTIONS values).
            api_key:   API token for paid backends (OpenAI / Anthropic / Google).
            hf_token:  Optional HuggingFace token (increases HF rate limits).
        """
        if backend not in list(self.BACKEND_OPTIONS.values()):
            raise ValueError(f"Unknown backend '{backend}'. "
                             f"Choose from: {list(self.BACKEND_OPTIONS.values())}")
        self.backend = backend
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")

    def label(self, sample_texts: list[str]) -> tuple[str, str]:
        """
        Classify one cluster. Returns (CorrectCategory, CorrectSubcategory).
        Catches all exceptions and returns ('Uncategorized', 'Uncategorized').
        """
        try:
            if self.backend == "chatgpt":
                return label_with_chatgpt(sample_texts, api_key=self.api_key)
            elif self.backend == "claude":
                return label_with_claude(sample_texts, api_key=self.api_key)
            elif self.backend == "gemini":
                return label_with_gemini(sample_texts, api_key=self.api_key)
            elif self.backend == "zephyr":
                return label_with_zephyr(sample_texts, hf_token=self.hf_token or None)
            elif self.backend == "mistral":
                return label_with_mistral(sample_texts, hf_token=self.hf_token or None)
            elif self.backend == "flan_t5":
                return label_with_flan_t5(sample_texts, hf_token=self.hf_token or None)
            else:
                return "Uncategorized", "Uncategorized"
        except Exception as exc:
            # Surface the error as a warning but never crash the pipeline
            print(f"[LLMLabeler] Backend '{self.backend}' error: {exc}")
            return "LLM-Error", "LLM-Error"

    @property
    def needs_token(self) -> bool:
        return self.backend in self.PAID_BACKENDS

    @property
    def is_free(self) -> bool:
        return self.backend in self.FREE_BACKENDS

    def validate(self) -> Optional[str]:
        """
        Return an error message string if the configuration is invalid,
        or None if everything looks fine.
        """
        if self.needs_token and not self.api_key:
            return (f"Backend '{self.backend}' requires an API key. "
                    "Enter it in the sidebar or set LLM_API_KEY env variable.")
        return None
