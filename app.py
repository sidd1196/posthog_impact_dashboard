import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import altair as alt
import re

DATA_DIR = Path(__file__).parent / "data"
SINCE = datetime.now(timezone.utc) - timedelta(days=90)
WEIGHTS = {"shipping_quality": 0.35, "team_unblocking": 0.25, "collab_breadth": 0.20, "sustained_output": 0.20}
BOTS = {"renovate", "renovate[bot]", "dependabot", "dependabot[bot]", "github-actions", "github-actions[bot]",
        "posthog-bot", "PostHog-bot", "semantic-release-bot"}

st.set_page_config(page_title="PostHog Engineering Impact", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.card { background:#1e1e2e; border-radius:12px; padding:16px; text-align:center; border:1px solid #2d2d3f; }
.rank { font-size:11px; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.engineer-name { font-size:16px; font-weight:700; margin:6px 0 2px 0; }
.score-big { font-size:36px; font-weight:800; color:#f0a500; line-height:1; }
.score-label { font-size:11px; color:#888; margin-bottom:10px; }
.dim-label { font-size:11px; color:#aaa; display:flex; justify-content:space-between; margin-bottom:2px; }
.avatar { border-radius:50%; width:56px; height:56px; }
.top-bar { display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    def read(name):
        path = DATA_DIR / f"{name}.json"
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)
    return read("prs"), read("reviews"), read("pr_comments"), read("issue_comments"), read("commits")


@st.cache_data(show_spinner=False)
def load_llm_scores():
    path = DATA_DIR / "llm_scores.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Scoring ─────────────────────────────────────────────────────────────────

def extract_scope(title):
    m = re.match(r"^\w+\(([^)]+)\)", title or "")
    return m.group(1).lower() if m else None


def percentile_rank(series):
    """Min-max normalize to 0-100."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100)


def weekly_stats(dates, since, weeks=13):
    if not dates:
        return 0, 1.0
    df = pd.DataFrame({"date": pd.to_datetime(dates, utc=True, format="ISO8601")})
    df = df[df["date"] >= since]
    df["week"] = df["date"].dt.to_period("W")
    weekly = df.groupby("week").size().reindex(
        pd.period_range(since, periods=weeks, freq="W"), fill_value=0
    )
    active = int((weekly > 0).sum())
    cv = float(weekly.std() / weekly.mean()) if weekly.mean() > 0 else 1.0
    return active, min(cv, 3.0)


@st.cache_data(show_spinner=False)
def compute_scores(prs_raw, reviews_raw, pr_comments_raw, issue_comments_raw, commits_raw):
    # ── normalize PR merged_at (Search API nests it under pull_request) ──
    prs = []
    for p in prs_raw:
        merged_at = p.get("merged_at") or (p.get("pull_request") or {}).get("merged_at")
        if not merged_at:
            continue
        prs.append({
            "number": p["number"],
            "author": p["user"]["login"],
            "title": p.get("title", ""),
            "created_at": p["created_at"],
            "merged_at": merged_at,
            "html_url": p.get("html_url", ""),
            "labels": [l["name"] if isinstance(l, dict) else l for l in p.get("labels", [])],
        })
    pr_df = pd.DataFrame(prs) if prs else pd.DataFrame(columns=["number","author","title","created_at","merged_at","html_url","labels"])

    # Filter bots
    all_authors = set(pr_df["author"].unique()) if not pr_df.empty else set()
    rev_df = pd.DataFrame(reviews_raw) if reviews_raw else pd.DataFrame(columns=["user","state","pr_number","pr_author","submitted_at"])
    if not rev_df.empty:
        rev_df["reviewer"] = rev_df["user"].apply(lambda u: u["login"] if isinstance(u, dict) else u)
        all_authors |= set(rev_df["reviewer"].unique())

    engineers = all_authors - BOTS

    if pr_df.empty and rev_df.empty:
        return pd.DataFrame()

    # ── PR-level metrics ──
    pr_df["created_dt"] = pd.to_datetime(pr_df["created_at"], utc=True)
    pr_df["merged_dt"] = pd.to_datetime(pr_df["merged_at"], utc=True)
    pr_df["cycle_hours"] = (pr_df["merged_dt"] - pr_df["created_dt"]).dt.total_seconds() / 3600
    pr_df["scope"] = pr_df["title"].apply(extract_scope)

    # Review rounds per PR: count of CHANGES_REQUESTED per PR
    if not rev_df.empty:
        changes_req = rev_df[rev_df["state"] == "CHANGES_REQUESTED"].groupby("pr_number").size().rename("changes_requested")
        pr_df = pr_df.merge(changes_req, left_on="number", right_index=True, how="left")
        pr_df["changes_requested"] = pr_df["changes_requested"].fillna(0)
    else:
        pr_df["changes_requested"] = 0

    rows = []
    for eng in engineers:
        # ── Shipping Quality ──
        my_prs = pr_df[pr_df["author"] == eng]
        n_prs = len(my_prs)
        avg_cycle = my_prs["cycle_hours"].median() if n_prs > 0 else np.nan
        avg_rounds = my_prs["changes_requested"].mean() if n_prs > 0 else np.nan
        scopes = set(my_prs["scope"].dropna().tolist())

        # ── Team Unblocking ──
        if not rev_df.empty:
            others_reviews = rev_df[(rev_df["reviewer"] == eng) & (rev_df.get("pr_author", pd.Series()).ne(eng) if "pr_author" in rev_df.columns else pd.Series(True, index=rev_df.index))]
            reviews_given = len(others_reviews)
            substantive = len(others_reviews[others_reviews["state"].isin(["APPROVED", "CHANGES_REQUESTED"])])
            unique_helped = others_reviews["pr_author"].nunique() if "pr_author" in others_reviews.columns else 0
        else:
            reviews_given = substantive = unique_helped = 0

        # ── Collaboration Breadth ──
        ic_count = sum(1 for c in issue_comments_raw if (c.get("user") or {}).get("login") == eng)
        prc_count = sum(1 for c in pr_comments_raw if (c.get("user") or {}).get("login") == eng)
        # Add scopes from PR comments file paths
        commented_paths = [c.get("path", "") for c in pr_comments_raw if (c.get("user") or {}).get("login") == eng]
        commented_dirs = {p.split("/")[0] for p in commented_paths if p}
        total_areas = len(scopes | commented_dirs)

        # ── Sustained Output ──
        pr_dates = my_prs["merged_dt"].dropna().tolist()
        commit_dates = [
            c["commit"]["author"]["date"]
            for c in commits_raw
            if (c.get("author") or {}).get("login") == eng
        ]
        all_dates = [str(d) for d in pr_dates] + commit_dates
        active_weeks, weekly_cv = weekly_stats(all_dates, SINCE)

        rows.append({
            "engineer": eng,
            "n_prs": n_prs,
            "avg_cycle_hours": avg_cycle,
            "avg_review_rounds": avg_rounds,
            "reviews_given": reviews_given,
            "substantive_ratio": substantive / reviews_given if reviews_given > 0 else 0,
            "unique_authors_helped": unique_helped,
            "issue_comments": ic_count,
            "pr_comment_count": prc_count,
            "codebase_areas": total_areas,
            "active_weeks": active_weeks,
            "weekly_cv": weekly_cv,
            "scopes": list(scopes),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Only score engineers with meaningful activity
    df = df[df["n_prs"] + df["reviews_given"] > 2].copy()
    if df.empty:
        return df

    # ── Dimension scores (0-100) ──
    df["cycle_score"] = percentile_rank(1 / (df["avg_cycle_hours"].fillna(df["avg_cycle_hours"].median()) + 1))
    df["rounds_score"] = percentile_rank(1 / (df["avg_review_rounds"].fillna(1) + 0.5))
    df["shipping_quality"] = 0.5 * df["cycle_score"] + 0.5 * df["rounds_score"]

    df["rev_volume_score"] = percentile_rank(df["reviews_given"])
    df["rev_depth_score"] = percentile_rank(df["substantive_ratio"])
    df["authors_helped_score"] = percentile_rank(df["unique_authors_helped"])
    df["team_unblocking"] = 0.4 * df["rev_volume_score"] + 0.3 * df["rev_depth_score"] + 0.3 * df["authors_helped_score"]

    df["areas_score"] = percentile_rank(df["codebase_areas"])
    df["issue_score"] = percentile_rank(df["issue_comments"] + df["pr_comment_count"])
    df["collab_breadth"] = 0.6 * df["areas_score"] + 0.4 * df["issue_score"]

    df["weeks_score"] = (df["active_weeks"] / 13 * 100).clip(0, 100)
    df["consistency_score"] = percentile_rank(1 / (df["weekly_cv"] + 0.1))
    df["sustained_output"] = 0.6 * df["weeks_score"] + 0.4 * df["consistency_score"]

    df["impact_score"] = (
        WEIGHTS["shipping_quality"] * df["shipping_quality"] +
        WEIGHTS["team_unblocking"] * df["team_unblocking"] +
        WEIGHTS["collab_breadth"] * df["collab_breadth"] +
        WEIGHTS["sustained_output"] * df["sustained_output"]
    ).round(1)

    return df.sort_values("impact_score", ascending=False).reset_index(drop=True)


# ── UI Components ────────────────────────────────────────────────────────────

DIM_COLORS = {
    "shipping_quality": "#4ade80",
    "team_unblocking": "#60a5fa",
    "collab_breadth": "#f472b6",
    "sustained_output": "#fb923c",
}
DIM_LABELS = {
    "shipping_quality": "Shipping Quality",
    "team_unblocking": "Team Unblocking",
    "collab_breadth": "Collaboration Breadth",
    "sustained_output": "Sustained Output",
}


def dim_bar(label, value, color):
    pct = min(max(float(value), 0), 100)
    return f"""
    <div style="margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#aaa;margin-bottom:2px">
        <span>{label}</span><span>{pct:.0f}</span>
      </div>
      <div style="background:#2d2d3f;border-radius:4px;height:6px">
        <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
      </div>
    </div>"""


def stacked_score_bar(row):
    """Horizontal stacked bar showing each dimension's weighted contribution to final score."""
    segments = ""
    for dim, weight in WEIGHTS.items():
        contribution = float(row[dim]) * weight  # 0-100 * weight → contribution out of 100
        width_pct = contribution  # already in 0-100 space
        color = DIM_COLORS[dim]
        label = DIM_LABELS[dim]
        segments += f'<div title="{label}: {contribution:.1f} pts ({int(weight*100)}% × {row[dim]:.0f})" style="width:{width_pct}%;background:{color};height:10px;display:inline-block"></div>'
    return f"""
    <div style="margin:8px 0 4px 0">
      <div style="font-size:10px;color:#888;margin-bottom:3px;text-align:left">Score breakdown (hover segments)</div>
      <div style="background:#2d2d3f;border-radius:4px;height:10px;overflow:hidden;display:flex">
        {segments}
      </div>
      <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;justify-content:center">
        {"".join(f'<span style="font-size:9px;color:{DIM_COLORS[d]}">■ {DIM_LABELS[d].split()[0]} {float(row[d])*WEIGHTS[d]:.1f}</span>' for d in WEIGHTS)}
      </div>
    </div>"""


def complexity_badge(llm_scores, engineer):
    data = llm_scores.get(engineer)
    if not data:
        return ""
    score = data.get("avg_complexity", 0)
    color = "#4ade80" if score >= 7 else "#f0a500" if score >= 4 else "#888"
    return f'<div style="margin:6px 0;font-size:11px;color:{color};background:#2d2d3f;border-radius:6px;padding:3px 8px;display:inline-block">🤖 Semantic Complexity {score:.1f}/10</div>'


def render_card(rank, row, llm_scores):
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal = medals.get(rank, f"#{rank}")
    avatar = f"https://github.com/{row['engineer']}.png?size=56"
    bars = "".join(dim_bar(DIM_LABELS[d], row[d], DIM_COLORS[d]) for d in DIM_LABELS)
    stacked = stacked_score_bar(row)
    badge = complexity_badge(llm_scores, row["engineer"])
    st.markdown(f"""
    <div class="card">
      <div class="rank">{medal} Rank {rank}</div>
      <img src="{avatar}" class="avatar" onerror="this.style.display='none'">
      <div class="engineer-name">{row['engineer']}</div>
      <div class="score-big">{row['impact_score']:.0f}</div>
      <div class="score-label">Impact Score</div>
      {badge}
      {stacked}
      <div style="border-top:1px solid #2d2d3f;margin:8px 0"></div>
      {bars}
    </div>
    """, unsafe_allow_html=True)


def render_detail(row, pr_df_full, rev_df_full, commits_raw, llm_scores):
    eng = row["engineer"]
    llm_data = llm_scores.get(eng)
    label_suffix = f" · 🤖 complexity {llm_data['avg_complexity']:.1f}/10" if llm_data else ""
    with st.expander(f"  {eng} — full breakdown{label_suffix}"):
        c1, c2, c3 = st.columns(3)

        # Recent PRs
        with c1:
            st.markdown("**Recent PRs merged**")
            eng_prs = pr_df_full[pr_df_full["author"] == row["engineer"]].sort_values("merged_dt", ascending=False).head(5)
            if eng_prs.empty:
                st.caption("No PRs in dataset")
            for _, pr in eng_prs.iterrows():
                cycle = f"{pr['cycle_hours']:.0f}h" if not pd.isna(pr['cycle_hours']) else "—"
                rounds = int(pr.get('changes_requested', 0))
                round_str = f" · {rounds} revision{'s' if rounds != 1 else ''}" if rounds > 0 else ""
                st.markdown(f"- [{pr['title'][:55]}{'…' if len(pr['title'])>55 else ''}]({pr['html_url']})  \n  `{cycle} to merge`{round_str}")

        # Reviews given
        with c2:
            st.markdown("**Reviews given (others' PRs)**")
            if not rev_df_full.empty and "pr_author" in rev_df_full.columns:
                eng_revs = rev_df_full[(rev_df_full["reviewer"] == row["engineer"]) & (rev_df_full["pr_author"] != row["engineer"])]
                by_state = eng_revs["state"].value_counts()
                for state, cnt in by_state.items():
                    icon = {"APPROVED": "✅", "CHANGES_REQUESTED": "🔁", "COMMENTED": "💬", "DISMISSED": "🚫"}.get(state, "•")
                    st.markdown(f"- {icon} **{state}**: {cnt}")
                top_helped = eng_revs["pr_author"].value_counts().head(4)
                if not top_helped.empty:
                    st.caption("Most reviewed:")
                    st.caption(", ".join(f"@{a} ({n})" for a, n in top_helped.items()))
            else:
                st.caption("No review data")

        # Codebase areas + activity chart
        with c3:
            st.markdown("**Codebase areas touched**")
            if row["scopes"]:
                st.markdown(" ".join(f"`{s}`" for s in sorted(row["scopes"])[:12]))
            else:
                st.caption("No scope tags found")

            st.markdown("**Weekly activity (PRs + commits)**")
            eng_prs = pr_df_full[pr_df_full["author"] == row["engineer"]]
            pr_dates = eng_prs["merged_dt"].dropna().tolist()
            commit_dates = [
                pd.Timestamp(c["commit"]["author"]["date"], tz="UTC")
                for c in commits_raw
                if (c.get("author") or {}).get("login") == row["engineer"]
            ]
            all_dates = [pd.Timestamp(d) for d in pr_dates] + commit_dates
            if all_dates:
                act_df = pd.DataFrame({"date": pd.to_datetime(all_dates, utc=True)})
                act_df = act_df[act_df["date"] >= SINCE]
                act_df["week"] = act_df["date"].dt.to_period("W").dt.start_time
                weekly = act_df.groupby("week").size().reset_index(name="count")
                chart = alt.Chart(weekly).mark_bar(color="#f0a500").encode(
                    x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=-30)),
                    y=alt.Y("count:Q", title="contributions"),
                    tooltip=["week:T", "count:Q"]
                ).properties(height=120)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No activity data")

        # ── LLM Semantic Complexity Analysis ──
        if llm_data and llm_data.get("pr_analyses"):
            st.markdown("---")
            st.markdown("**🤖 LLM Semantic Complexity Analysis** — Gemini Flash assessed the 3 most substantive PRs")
            avg = llm_data["avg_complexity"]
            color = "#4ade80" if avg >= 7 else "#f0a500" if avg >= 4 else "#aaa"
            st.markdown(f"Average complexity: <span style='color:{color};font-weight:700;font-size:18px'>{avg:.1f} / 10</span>", unsafe_allow_html=True)
            cols_llm = st.columns(len(llm_data["pr_analyses"]))
            cat_colors = {
                "feature": "#60a5fa", "bugfix": "#f87171", "refactor": "#c084fc",
                "infrastructure": "#fb923c", "docs": "#94a3b8", "dependency": "#6b7280", "test": "#34d399"
            }
            for col, pr_analysis in zip(cols_llm, llm_data["pr_analyses"]):
                with col:
                    score = pr_analysis["complexity_score"]
                    cat = pr_analysis.get("category", "unknown")
                    cat_color = cat_colors.get(cat, "#888")
                    bar_color = "#4ade80" if score >= 7 else "#f0a500" if score >= 4 else "#888"
                    title = pr_analysis["title"]
                    url = pr_analysis.get("html_url", "#")
                    reasoning = pr_analysis.get("reasoning", "")
                    st.markdown(f"""
<div style="background:#1a1a2e;border-radius:8px;padding:12px;border:1px solid #2d2d3f;height:100%">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="background:{cat_color}22;color:{cat_color};font-size:10px;padding:2px 7px;border-radius:4px">{cat}</span>
    <span style="font-size:20px;font-weight:800;color:{bar_color}">{score}/10</span>
  </div>
  <div style="font-size:12px;font-weight:600;margin-bottom:6px"><a href="{url}" target="_blank" style="color:#e0e0e0;text-decoration:none">{title[:65]}{'…' if len(title)>65 else ''}</a></div>
  <div style="background:#2d2d3f;border-radius:3px;height:4px;margin-bottom:8px">
    <div style="background:{bar_color};width:{score*10}%;height:4px;border-radius:3px"></div>
  </div>
  <div style="font-size:11px;color:#aaa;line-height:1.5;font-style:italic">"{reasoning}"</div>
</div>""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Check data exists
    missing = [n for n in ["prs", "reviews", "pr_comments", "issue_comments", "commits"] if not (DATA_DIR / f"{n}.json").exists()]
    if missing:
        st.warning(f"Data files not ready yet: {', '.join(missing)}. Run `fetch_data.py` first.")
        st.stop()

    with st.spinner("Loading and scoring data..."):
        prs_raw, reviews_raw, pr_comments_raw, issue_comments_raw, commits_raw = load_data()
        scores = compute_scores(prs_raw, reviews_raw, pr_comments_raw, issue_comments_raw, commits_raw)
        llm_scores = load_llm_scores()

    if scores.empty:
        st.error("Not enough data to compute scores.")
        st.stop()

    top5 = scores.head(5)

    # Rebuild PR df for detail view (cached inside compute_scores but we need it here too)
    prs_norm = []
    for p in prs_raw:
        merged_at = p.get("merged_at") or (p.get("pull_request") or {}).get("merged_at")
        if not merged_at:
            continue
        prs_norm.append({"number": p["number"], "author": p["user"]["login"], "title": p.get("title", ""),
                          "created_at": p["created_at"], "merged_at": merged_at, "html_url": p.get("html_url", "")})
    pr_df = pd.DataFrame(prs_norm)
    pr_df["created_dt"] = pd.to_datetime(pr_df["created_at"], utc=True)
    pr_df["merged_dt"] = pd.to_datetime(pr_df["merged_at"], utc=True)
    pr_df["cycle_hours"] = (pr_df["merged_dt"] - pr_df["created_dt"]).dt.total_seconds() / 3600
    pr_df["scope"] = pr_df["title"].apply(extract_scope)
    if reviews_raw:
        rev_df = pd.DataFrame(reviews_raw)
        rev_df["reviewer"] = rev_df["user"].apply(lambda u: u["login"] if isinstance(u, dict) else u)
        changes_req = rev_df[rev_df["state"] == "CHANGES_REQUESTED"].groupby("pr_number").size().rename("changes_requested")
        pr_df = pr_df.merge(changes_req, left_on="number", right_index=True, how="left")
        pr_df["changes_requested"] = pr_df["changes_requested"].fillna(0)
    else:
        rev_df = pd.DataFrame()
        pr_df["changes_requested"] = 0

    # ── Header ──
    date_range = f"{SINCE.strftime('%b %d')} – {datetime.now().strftime('%b %d, %Y')}"
    st.markdown(f"## PostHog · Engineering Impact Dashboard")
    st.caption(f"Top 5 most impactful engineers · Last 90 days · {date_range} · {len(prs_raw):,} PRs · {len(commits_raw):,} commits analysed")
    st.divider()

    # ── Top 5 Cards ──
    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            render_card(i + 1, row, llm_scores)

    st.divider()

    # ── Detail Expanders ──
    st.markdown("### Engineer Breakdowns")
    for _, row in top5.iterrows():
        render_detail(row, pr_df, rev_df, commits_raw, llm_scores)

    # ── Sidebar Methodology ──
    with st.sidebar:
        st.markdown("## Methodology")
        st.markdown(f"**Scoring period:** {date_range}")
        st.markdown(f"**Data sources:** GitHub PRs, reviews, review comments, issue comments, commits")
        st.markdown("---")
        for dim, weight in WEIGHTS.items():
            label = DIM_LABELS[dim]
            color = DIM_COLORS[dim]
            st.markdown(f"**{label}** — {int(weight*100)}%")

        st.markdown("---")
        with st.expander("Shipping Quality (35%)"):
            st.markdown("""
- **Cycle time score** (50%): median hours from PR open → merge. Faster = higher score.
- **Review rounds score** (50%): avg `CHANGES_REQUESTED` reviews per PR. Fewer = cleaner work.
            """)
        with st.expander("Team Unblocking (25%)"):
            st.markdown("""
- **Reviews given** (40%): count of reviews left on *other people's* PRs.
- **Review depth** (30%): ratio of `APPROVED` + `CHANGES_REQUESTED` vs total (substantive vs passive).
- **Authors helped** (30%): distinct engineers whose PRs they reviewed.
            """)
        with st.expander("Collaboration Breadth (20%)"):
            st.markdown("""
- **Codebase areas** (60%): distinct scopes from `feat(scope):` titles + commented file directories.
- **Discussion engagement** (40%): issue comments + inline PR review comments.
            """)
        with st.expander("Sustained Output (20%)"):
            st.markdown("""
- **Active weeks** (60%): weeks with ≥1 merged PR or commit out of 13 total.
- **Consistency** (40%): inverse of week-over-week variance (low variance = reliable cadence).
            """)
        with st.expander("🤖 LLM Semantic Complexity"):
            st.markdown("""
- **Model**: Gemini 1.5 Flash
- **What it scores**: top 3 most substantive PRs per engineer (routine `chore`/`bump` PRs excluded)
- **Scale**: 1–10 (1–3 routine, 4–6 moderate, 7–9 high, 10 exceptional)
- **Output**: score + category + 2-sentence reasoning per PR
- **Shown as**: badge on card + per-PR cards with reasoning in the breakdown
- **Note**: shown as a qualitative signal, not folded into the overall impact score
            """)
        st.markdown("---")
        st.caption("All dimensions normalised 0–100 via min-max scaling across all engineers with >2 contributions.")


main()
