# PostHog Engineering Impact Dashboard

An interactive dashboard that identifies the top 5 most impactful engineers at PostHog using 90 days of GitHub data — built for the Workweave take-home assignment.

**Live demo:** https://posthogimpactdashboard-2cbdgrlapnbkrxljfcrbs7.streamlit.app/

---

## How Impact is Defined

Impact is scored across 4 dimensions:

| Dimension | Weight | Signals |
|---|---|---|
| **Shipping Quality** | 35% | PR cycle time (open → merge), review revision rounds (CHANGES_REQUESTED count) |
| **Team Unblocking** | 25% | Reviews given on others' PRs, review depth (APPROVED/CHANGES_REQUESTED ratio), unique authors helped |
| **Collaboration Breadth** | 20% | Codebase areas touched (conventional commit scopes + file paths), issue & PR comment engagement |
| **Sustained Output** | 20% | Active weeks out of 13, week-over-week consistency (inverse of variance) |

Each dimension is scored 0–100 via min-max normalization across all engineers, then combined into a weighted impact score. The score breakdown is shown as a transparent stacked bar on each card — no mystery numbers.

### LLM-as-Judge (Semantic Complexity)
After basic scoring, the top 20 engineers' most substantive PRs are evaluated by **Groq llama-3.3-70b** for semantic complexity (1–10). The model returns a score, category, and 2-sentence reasoning per PR. Results are shown as a qualitative signal in each engineer's detail section — not folded into the overall score.

---

## Data Sources

All data from the [PostHog/posthog](https://github.com/PostHog/posthog) GitHub repository, last 90 days:

- **3,000 merged PRs** (via GitHub Search API, 3 × monthly windows)
- **10,373 PR reviews**
- **5,000 inline PR review comments**
- **5,000 issue comments**
- **5,000 commits**

---

## Tech Stack

- **Dashboard:** Streamlit + Altair
- **Analysis:** Pandas + NumPy
- **LLM scoring:** Groq API (llama-3.3-70b-versatile)
- **Data fetching:** GitHub REST API

---

## Running Locally

```bash
# 1. Create and activate virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard (uses pre-computed data)
streamlit run app.py
```

### Re-fetching Data (optional)
```bash
# Requires a GitHub personal access token
python fetch_data.py
```

### Re-running LLM Analysis (optional)
```bash
# Requires a Groq API key in groq_key.json: {"api_key": "gsk_..."}
python llm_analysis.py
```

---

## Project Structure

```
├── app.py               # Streamlit dashboard
├── fetch_data.py        # GitHub data fetcher (run locally)
├── llm_analysis.py      # LLM complexity scorer (run locally)
├── requirements.txt     # Deployment dependencies
├── data/
│   ├── prs.json
│   ├── reviews.json
│   ├── pr_comments.json
│   ├── issue_comments.json
│   ├── commits.json
│   └── llm_scores.json  # Pre-computed LLM scores
└── .streamlit/
    └── config.toml      # Dark theme config
```
