# AI Ticket Classification Model

Classifies technology support tickets by **skill / technology area**
(Python, Java, React, Azure, GCP, Power BI, SQL, C++, DevOps, ML…)
using unsupervised learning + optional LLM label assignment.

---

## What it does

Given a dataset of IT/engineering support tickets, the pipeline:

1. **Cleans** text (lowercasing, tech-term normalization, stopwords)
2. **Embeds** each ticket using `sentence-transformers`
3. **Reduces** dimensions with UMAP (2D for visualization + 5D for clustering)
4. **Clusters** with HDBSCAN (no fixed K — discovers natural groupings)
5. **Labels** each cluster via an LLM that reads the ticket conversations and assigns:
   - **CorrectCategory** — the technology/skill: `Python`, `React`, `Azure`, `SQL / Database`, etc.
   - **CorrectSubcategory** — the issue type: `Runtime Error`, `Build / Dependency`, `Performance`, etc.

---

## Project structure

```
ticket_classifier/
├── main.py               ← Streamlit UI
├── pipeline.py           ← Pipeline steps (validate → clean → embed → UMAP → HDBSCAN → LLM)
├── utils.py              ← Text cleaning, embeddings, UMAP, HDBSCAN, file I/O
├── models/
│   └── llm_labeler.py    ← 6 LLM backends (3 paid + 3 free)
├── requirements.txt
├── sample_tickets.csv    ← 40 sample tickets (Python, Java, React, Azure, GCP, etc.)
└── README.md
```

### Files you'll edit most often

| File | What to change |
|------|----------------|
| `utils.py` | Edit `TECH_CATEGORY_KEYWORDS` and `TECH_SUBCATEGORY_KEYWORDS` to add/rename categories |
| `models/llm_labeler.py` | Edit `SYSTEM_PROMPT` to refine how the LLM classifies; add new backends |
| `pipeline.py` | Tune pipeline steps or add new ones |
| `main.py` | UI changes |

---

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

App opens at `http://localhost:8501`

---

## LLM backends (6 options)

### Paid — need API key

| Backend | Model | Token needed |
|---------|-------|-------------|
| **ChatGPT (OpenAI)** | gpt-4o-mini / gpt-4o / gpt-3.5-turbo | `OPENAI_API_KEY` |
| **Claude (Anthropic)** | claude-haiku / claude-sonnet | `ANTHROPIC_API_KEY` |
| **Gemini (Google)** | gemini-1.5-flash / gemini-1.5-pro | `GOOGLE_API_KEY` |

Enter your key in the sidebar. Never stored — session only.

### Free — no token needed

| Backend | Model | Notes |
|---------|-------|-------|
| **Zephyr-7B** | HuggingFaceH4/zephyr-7b-beta | Best free quality |
| **Mistral-7B** | mistralai/Mistral-7B-Instruct-v0.2 | Good quality, fast |
| **Flan-T5-Large** | google/flan-t5-large | Fastest; good for testing |

All three use HuggingFace's public Inference API. No sign-up required.
Add an optional HF token in the sidebar for higher rate limits.

---

## Input format

Required column: `ticket_context` (your ticket description/conversation).
Optional columns: `category`, `Subcategory`, `ID`.

```csv
ID,ticket_context,category,Subcategory
1,"I keep getting a KeyError in Python when accessing a dict...",Unknown,Unknown
2,"React useEffect is causing infinite re-render loop...",Unknown,Unknown
```

---

## Output columns added

| Column | Example value |
|--------|---------------|
| `CorrectCategory` | `Python` / `React` / `Azure` / `SQL / Database` |
| `CorrectSubcategory` | `Runtime Error` / `Build / Dependency` / `Performance` |

---

## Valid category values

```
Python | Java | JavaScript | React | Angular / Vue | TypeScript |
Azure | AWS | GCP | Power BI | SQL / Database | C++ / C |
DevOps / CI-CD | Machine Learning | API / Backend | Other
```

## Valid subcategory values

```
Runtime Error | Build / Dependency | Performance | Authentication / Auth |
Data / Query Issue | Deployment / Config | UI / Rendering |
Networking / CORS | Visualisation | Model / Algorithm | Other
```

---

## Adding custom categories

Edit `TECH_CATEGORY_KEYWORDS` in `utils.py`:

```python
TECH_CATEGORY_KEYWORDS["Salesforce"] = [
    "salesforce", "apex", "lwc", "soql", "visualforce", "sfdc", "crm"
]
```

Edit the valid categories list in `SYSTEM_PROMPT` inside `models/llm_labeler.py`.

---

## Tuning HDBSCAN (if clusters look wrong)

| Problem | Fix |
|---------|-----|
| Too few clusters (everything merged) | Lower `min_cluster_size` in sidebar |
| Too many noise points (grey on map) | Lower `min_samples` |
| Too many tiny clusters | Raise `min_cluster_size` |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Required column 'ticket_context' not found` | Change **Text column** in sidebar to match your file |
| Paid LLM error: 401 Unauthorized | Check API key in sidebar |
| Free LLM returns "LLM-Error" | HF Inference API rate-limited; wait 30s or add HF token |
| Only 1 cluster found | Lower `min_cluster_size` to 2 |
