"""
prompts.py  —  Few-shot prompt templates for ticket classification.

The LLM receives representative ticket texts from ONE cluster and must
decide CorrectCategory (the technology/skill) and CorrectSubcategory
(the type of problem) purely by reading and reasoning about the conversation.

Few-shot examples anchor the format AND teach the model the expected
category taxonomy so it does not invent its own labels.
"""

# ── Category & Subcategory taxonomy (source of truth) ──────────────────────
VALID_CATEGORIES = [
    "Python", "Java", "JavaScript", "React", "Angular", "Vue",
    "TypeScript", "Node.js", "C++", "C#", ".NET", "Go", "Rust",
    "Azure", "AWS", "GCP", "DevOps / CI-CD", "Docker / Kubernetes",
    "SQL / Database", "Power BI", "Machine Learning / AI",
    "API / Backend", "Mobile (iOS/Android)", "Security", "Other",
]

VALID_SUBCATEGORIES = [
    "Runtime Error / Exception",
    "Build / Dependency Issue",
    "Performance / Timeout",
    "Authentication / Authorization",
    "Data / Query Issue",
    "Deployment / Configuration",
    "UI / Rendering Issue",
    "Networking / CORS",
    "Visualisation / Reporting",
    "Model Training / Accuracy",
    "Integration Issue",
    "Installation / Setup",
    "Documentation / How-to",
    "Other",
]

# ── Few-shot examples ───────────────────────────────────────────────────────
# Each example shows: ticket(s) → correct JSON output.
# Chosen to cover diverse categories and subcategories.

FEW_SHOT_EXAMPLES = """
### Example 1
Tickets:
1. "Getting KeyError when I access dict after .get() call in my Python script."
2. "pandas DataFrame merge produces duplicate rows on left join with common ID."
3. "Python virtualenv won't activate on Windows — 'source' command not found."
Answer:
{"CorrectCategory": "Python", "CorrectSubcategory": "Runtime Error / Exception"}

### Example 2
Tickets:
1. "React useEffect runs infinitely — I update state inside with no dep array."
2. "CORS error in React when calling backend API — blocked by browser policy."
3. "useState setter not updating value immediately after calling it."
Answer:
{"CorrectCategory": "React", "CorrectSubcategory": "UI / Rendering Issue"}

### Example 3
Tickets:
1. "Azure DevOps pipeline fails at deploy stage — ARM template 409 Conflict."
2. "Azure Blob Storage SAS token expires before large file upload completes."
3. "Azure AD app registration not returning correct JWT claims."
Answer:
{"CorrectCategory": "Azure", "CorrectSubcategory": "Deployment / Configuration"}

### Example 4
Tickets:
1. "SQL query runs 45 minutes on 10M row table — no indexes on join columns."
2. "Deadlock in SQL Server when two transactions update same rows."
3. "PostgreSQL stored procedure returns different results with same parameters."
Answer:
{"CorrectCategory": "SQL / Database", "CorrectSubcategory": "Performance / Timeout"}

### Example 5
Tickets:
1. "NullPointerException in Spring Boot REST endpoint after deserialization."
2. "Maven build fails — 'package does not exist' even after adding to pom.xml."
3. "Spring Boot @Autowired bean not injecting — 'No qualifying bean of type'."
Answer:
{"CorrectCategory": "Java", "CorrectSubcategory": "Runtime Error / Exception"}

### Example 6
Tickets:
1. "Power BI report not refreshing on schedule — history shows failed."
2. "DAX CALCULATE formula returns blank with date filter."
3. "Power BI embedded report shows no data for RLS-configured user."
Answer:
{"CorrectCategory": "Power BI", "CorrectSubcategory": "Visualisation / Reporting"}

### Example 7
Tickets:
1. "Kubernetes pod OOMKilled — memory spikes to 700Mi, limit is 512Mi."
2. "Docker container exits with code 137 immediately after start."
3. "Helm chart upgrade fails: cannot patch resource with kind Deployment."
Answer:
{"CorrectCategory": "Docker / Kubernetes", "CorrectSubcategory": "Deployment / Configuration"}

### Example 8
Tickets:
1. "TensorFlow shape mismatch when passing variable-size batches to model."
2. "Scikit-learn pipeline ValueError — transformer expects DataFrame, gets array."
3. "Model accuracy drops from 92% training to 61% on production data."
Answer:
{"CorrectCategory": "Machine Learning / AI", "CorrectSubcategory": "Model Training / Accuracy"}
"""

# ── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a senior software engineering support analyst.

Your task: read a small set of support tickets (all from the same cluster of
similar issues) and assign exactly ONE CorrectCategory and ONE CorrectSubcategory.

Rules:
1. Base your decision on the TECHNOLOGY the user is working with and the TYPE of problem.
2. Choose from the lists below — do not invent new values.
3. Return ONLY a JSON object with two keys. No explanation, no markdown, no extra text.

Valid CorrectCategory values:
{", ".join(VALID_CATEGORIES)}

Valid CorrectSubcategory values:
{", ".join(VALID_SUBCATEGORIES)}

Study these examples carefully before answering:
{FEW_SHOT_EXAMPLES}
"""

# ── User message template ───────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """Now classify this cluster of {n} tickets:

Tickets:
{ticket_list}

Return ONLY the JSON:
{{"CorrectCategory": "...", "CorrectSubcategory": "..."}}"""


def build_user_prompt(ticket_texts: list[str], max_tickets: int = 5) -> str:
    """Format up to max_tickets raw ticket texts into the user message."""
    selected = ticket_texts[:max_tickets]
    lines = "\n".join(f'{i + 1}. "{t[:300].strip()}"' for i, t in enumerate(selected))
    return USER_PROMPT_TEMPLATE.format(n=len(selected), ticket_list=lines)
