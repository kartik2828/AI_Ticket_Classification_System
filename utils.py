"""
utils.py — Helper utilities for text cleaning, embedding generation,
           tech-domain label inference, and file I/O.

DOMAIN: Technology support tickets — the goal is to classify tickets by the
        TECHNOLOGY / SKILL involved (Python, Java, React, Azure, GCP, etc.),
        NOT by generic hardware/software buckets.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# NLTK bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _get_nltk_resources() -> set:
    """Download required NLTK corpora once, return English stopwords."""
    import nltk
    for resource, kind in [("stopwords", "corpora"), ("punkt", "tokenizers"), ("wordnet", "corpora")]:
        try:
            nltk.data.find(f"{kind}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# Tech-domain normalizations  (preserve key terms before stopword stripping)
# ─────────────────────────────────────────────────────────────────────────────

# These patterns protect critical tech keywords from being mangled or removed.
TECH_NORMALIZATIONS: dict[str, str] = {
    r"\bc\+\+\b":               "cplusplus",
    r"\bc#\b":                  "csharp",
    r"\b\.net\b":               "dotnet",
    r"\bnode\.?js\b":           "nodejs",
    r"\bvue\.?js\b":            "vuejs",
    r"\btype\s*script\b":       "typescript",
    r"\bjava\s*script\b":       "javascript",
    r"\bpow(er)?\s*bi\b":       "powerbi",
    r"\bpow(er)?\s*shell\b":    "powershell",
    r"\bk8s\b":                 "kubernetes",
    r"\bml\b":                  "machinelearning",
    r"\bai\b":                  "artificialintelligence",
    r"\bllm\b":                 "largelanguagemodel",
    r"\bgit\s*hub\b":           "github",
    r"\bgit\s*lab\b":           "gitlab",
    r"\bci[/\s]?cd\b":          "cicd",
    r"\bapi\b":                 "api",
    r"\brest\s*api\b":          "restapi",
    r"\bgql\b":                 "graphql",
    r"\bgraph\s*ql\b":          "graphql",
    r"\baws\b":                 "aws",
    r"\bgcp\b":                 "gcp",
    r"\baz(ure)?\b":            "azure",
    r"\btf\b":                  "terraform",
    r"\bpandas\b":              "pandas",
    r"\bnp\b":                  "numpy",
    r"\bsk\s*learn\b":          "scikitlearn",
    r"\bscikit[\s-]learn\b":    "scikitlearn",
    r"\btensor\s*flow\b":       "tensorflow",
    r"\btorch\b":               "pytorch",
    r"\bpy\s*torch\b":          "pytorch",
    r"\bspring\s*boot\b":       "springboot",
    r"\bmaven\b":               "maven",
    r"\bgradle\b":              "gradle",
    r"\bnull\s*pointer\b":      "nullpointerexception",
    r"\bnpe\b":                 "nullpointerexception",
    r"\borm\b":                 "orm",
    r"\bjpa\b":                 "jpa",
    r"\bhibernate\b":           "hibernate",
    r"\bdocker\s*file\b":       "dockerfile",
    r"\bhelm\s*chart\b":        "helmchart",
    r"\bdag\b":                 "dag",
    r"\bairflow\b":             "apacheairflow",
    r"\bkafka\b":               "apachekafka",
    r"\bspark\b":               "apachespark",
    r"\bdbt\b":                 "dbt",
    r"\bpbi\b":                 "powerbi",
}


def clean_text(text: str, stop_words: Optional[set] = None) -> str:
    """
    Normalize and clean a raw ticket text string for embedding.

    Critically: tech keywords (Python, React, Azure, etc.) are preserved and
    normalized BEFORE stopword removal so they survive the cleaning pipeline.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower().strip()

    # 1. Protect tech terms via normalization (do BEFORE removing punctuation)
    for pattern, replacement in TECH_NORMALIZATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Remove URLs and email addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 3. Remove ticket IDs (INC0001, TKT-123, #456)
    text = re.sub(r"\b(inc|req|tkt|ticket|issue|#)\s*[\-]?\d+\b", " ", text, flags=re.IGNORECASE)

    # 4. Remove leftover punctuation (keep alphanumeric + spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Stopword removal (keep all words > 2 chars that are not stop words)
    if stop_words:
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        text = " ".join(tokens)

    return text


def build_combined_text(row: pd.Series, text_col: str,
                         cat_col: Optional[str], sub_col: Optional[str]) -> str:
    """Combine the ticket body with any existing category hints (weighted 1x extra)."""
    parts = [str(row.get(text_col, ""))]
    if cat_col and cat_col in row.index and pd.notna(row[cat_col]):
        parts.append(str(row[cat_col]))
    if sub_col and sub_col in row.index and pd.notna(row[sub_col]):
        parts.append(str(row[sub_col]))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2",
                         batch_size: int = 64,
                         progress_callback=None) -> np.ndarray:
    """
    Generate sentence embeddings using sentence-transformers.

    Returns: ndarray of shape (n_samples, embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    total = len(texts)
    all_embeddings: list[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(vecs)
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)

    return np.vstack(all_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2,
                       n_neighbors: int = 15, min_dist: float = 0.1,
                       random_state: int = 42) -> np.ndarray:
    """UMAP dimensionality reduction."""
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# HDBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def cluster_embeddings(reduced: np.ndarray, min_cluster_size: int = 3,
                        min_samples: int = 2) -> np.ndarray:
    """HDBSCAN clustering. Returns integer labels (-1 = noise)."""
    import hdbscan  # type: ignore

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return clusterer.fit_predict(reduced)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback — TECH domain keyword maps
# ─────────────────────────────────────────────────────────────────────────────

# CorrectCategory → technology / skill area
TECH_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Python":            ["python", "pandas", "numpy", "scikitlearn", "tensorflow",
                          "pytorch", "fastapi", "flask", "django", "virtualenv",
                          "pip", "pypi", "pep", "asyncio", "dataframe", "matplotlib",
                          "seaborn", "jupyter", "conda", "scipy"],
    "Java":              ["java", "springboot", "spring", "maven", "gradle", "jvm",
                          "hibernate", "jpa", "nullpointerexception", "jar", "classpath",
                          "tomcat", "junit", "intellij", "eclipse", "kotlin"],
    "JavaScript":        ["javascript", "nodejs", "npm", "yarn", "webpack", "babel",
                          "express", "eslint", "jest", "mocha", "typescript",
                          "async", "await", "promise", "callback", "closure"],
    "React":             ["react", "usestate", "useeffect", "usecallback", "usememo",
                          "component", "jsx", "tsx", "redux", "context", "hooks",
                          "reactrouter", "nextjs", "vite", "statemanagement", "rerender"],
    "Angular / Vue":     ["angular", "vuejs", "ng", "directive", "observable",
                          "rxjs", "ngmodule", "component", "template", "binding",
                          "compositionapi", "pinia", "vuex", "nuxt"],
    "TypeScript":        ["typescript", "tsc", "tsconfig", "interface", "generic",
                          "typeguard", "enum", "decorator", "strictmode"],
    "Azure":             ["azure", "devops", "armtemplate", "appservice", "blobstorage",
                          "activedirectory", "jwt", "keyvault", "aksazure", "functionapp",
                          "subscriptionazure", "resourcegroup", "cosmosdb", "azuread"],
    "AWS":               ["aws", "ec2", "s3", "lambda", "iam", "cloudwatch",
                          "rds", "dynamodb", "cloudformation", "beanstalk", "sqs",
                          "sns", "vpc", "route53", "apigateway"],
    "GCP":               ["gcp", "bigquery", "cloudrun", "pubsub", "cloudfunction",
                          "gke", "computeengine", "cloudsql", "dataflow", "firestore",
                          "cloudstorageGCP", "serviceaccount", "terraform"],
    "Power BI":          ["powerbi", "dax", "measure", "calculated", "dataset",
                          "report", "dashboard", "rowlevelsecurity", "refresh",
                          "gateway", "pbix", "embedded", "visualisation", "slicer"],
    "SQL / Database":    ["sql", "query", "deadlock", "index", "join", "stored",
                          "procedure", "postgres", "postgresql", "mysql", "mssql",
                          "sqlserver", "oracle", "transaction", "schema", "migration",
                          "dbt", "etl", "orm", "jpa"],
    "C++ / C":           ["cplusplus", "cpp", "csharp", "dotnet", "clang", "gcc",
                          "segfault", "segmentation", "pointer", "memory", "vector",
                          "iterator", "template", "valgrind", "cmake", "makefile"],
    "DevOps / CI-CD":    ["cicd", "jenkins", "github", "gitlab", "actions", "pipeline",
                          "docker", "kubernetes", "helm", "helmchart", "terraform",
                          "ansible", "deployment", "container", "pod", "manifest"],
    "Machine Learning":  ["machinelearning", "artificialintelligence", "model", "accuracy",
                          "training", "inference", "feature", "dataset", "tensorflow",
                          "pytorch", "scikitlearn", "regression", "classification",
                          "neural", "network", "overfitting", "underfitting", "epoch"],
    "API / Backend":     ["restapi", "graphql", "api", "endpoint", "cors", "http",
                          "request", "response", "authentication", "oauth", "token",
                          "middleware", "serialization", "payload", "webhook"],
}

# CorrectSubcategory → specific issue type
TECH_SUBCATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Runtime Error":         ["error", "exception", "crash", "traceback", "segfault",
                               "runtime", "nullpointerexception", "keyerror", "typeerror"],
    "Build / Dependency":    ["build", "maven", "gradle", "npm", "pip", "package",
                               "dependency", "module", "import", "classpath", "pom",
                               "requirements", "version", "conflict"],
    "Performance":           ["slow", "performance", "timeout", "latency", "memory",
                               "oomkilled", "spike", "optimization", "cache", "index"],
    "Authentication / Auth": ["authentication", "token", "jwt", "oauth", "login",
                               "permission", "403", "401", "unauthorized", "credential",
                               "sso", "iam", "rbac"],
    "Data / Query Issue":    ["query", "sql", "deadlock", "join", "data", "null",
                               "schema", "migration", "dataframe", "merge", "filter"],
    "Deployment / Config":   ["deploy", "deployment", "config", "environment", "pipeline",
                               "cicd", "yaml", "helm", "terraform", "docker", "pod",
                               "container", "kubernetes"],
    "UI / Rendering":        ["render", "component", "display", "ui", "rerender",
                               "useeffect", "template", "binding", "observable", "reactive"],
    "Networking / CORS":     ["cors", "network", "connection", "request", "api",
                               "endpoint", "proxy", "firewall", "port", "socket"],
    "Visualisation":         ["powerbi", "dashboard", "report", "chart", "dax",
                               "measure", "filter", "visual", "slice", "refresh"],
    "Model / Algorithm":     ["model", "accuracy", "training", "epoch", "loss",
                               "shape", "mismatch", "feature", "overfitting"],
}


def extract_top_terms(texts: list[str], top_n: int = 10) -> list[str]:
    """Frequency-rank the most meaningful tokens across a list of cleaned texts."""
    # Generic filler words to exclude even after stopword removal
    filler = {
        "error", "issue", "problem", "getting", "using", "work", "need", "please",
        "help", "also", "like", "seems", "even", "every", "time", "make", "call",
        "line", "file", "code", "just", "back", "right", "still", "shows",
        "trying", "run", "running", "return", "returns", "calling", "called",
        "trying", "tried", "different", "same", "function", "value",
    }
    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(t for t in text.split() if len(t) > 3 and t not in filler)

    counter = Counter(all_tokens)
    return [word for word, _ in counter.most_common(top_n)]


def _score_keyword_map(terms: list[str], keyword_map: dict[str, list[str]]) -> str:
    """Score a list of terms against every entry in a keyword map; return best match."""
    scores: dict[str, int] = {label: 0 for label in keyword_map}
    for label, keywords in keyword_map.items():
        for term in terms:
            for kw in keywords:
                if kw in term or term in kw:
                    scores[label] += 1
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "Uncategorized"
    return max(scores, key=lambda k: scores[k])


def infer_labels_heuristic(cluster_id: int, cluster_texts: list[str]) -> tuple[str, str]:
    """
    Fallback heuristic: infer (CorrectCategory, CorrectSubcategory) from
    dominant keywords in the cluster's cleaned texts.
    """
    if not cluster_texts:
        return "Uncategorized", "Uncategorized"

    top_terms = extract_top_terms(cluster_texts, top_n=15)
    category = _score_keyword_map(top_terms, TECH_CATEGORY_KEYWORDS)
    subcategory = _score_keyword_map(top_terms, TECH_SUBCATEGORY_KEYWORDS)

    if cluster_id == -1:
        category = f"[Review] {category}"

    return category, subcategory


# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(file_obj, filename: str) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame, stripping column-name whitespace."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_obj, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif ext == ".xlsb":
        df = pd.read_excel(file_obj, engine="pyxlsb")
    else:
        raise ValueError(f"Unsupported format '{ext}'. Upload CSV or Excel.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return list of required columns that are missing from df."""
    return [c for c in required if c not in df.columns]


def save_results(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """Serialize augmented DataFrame to bytes for Streamlit download."""
    buf = io.BytesIO()
    if fmt == "excel":
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Classified Tickets")
    else:
        df.to_csv(buf, index=False)
    return buf.getvalue()


def build_code_zip(project_dir: str = ".") -> bytes:
    """Bundle all source files into a ZIP for download."""
    include = [
        "main.py", "pipeline.py", "utils.py",
        "models/llm_labeler.py", "models/__init__.py",
        "requirements.txt", "sample_tickets.csv", "README.md",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            full = os.path.join(project_dir, rel)
            if os.path.exists(full):
                zf.write(full, arcname=rel)
    return buf.getvalue()
