import os
import re
import textwrap
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # Optional dependency handled at runtime
    SentimentIntensityAnalyzer = None

DATA_DEFAULT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "EM Survey Results Cleaned.xlsx"))

STOPWORDS = {
    "the", "and", "for", "that", "with", "have", "this", "from", "your", "about", "their", "they",
    "would", "should", "could", "into", "those", "there", "what", "when", "where", "while", "being",
    "been", "also", "over", "very", "more", "than", "will", "just", "like", "make", "needs",
    "need", "able", "across", "such", "much", "take", "help", "helping", "helps", "get", "got",
    "really", "still", "even", "back", "through", "per", "etc", "some", "many", "most", "each", "every",
    "ensure", "ensuring", "within", "among", "onto", "toward", "towards", "however", "amongst", "others",
    "making", "made", "often", "maybe", "might", "must", "cannot", "cant", "dont", "doesnt", "shouldnt",
    "isnt", "arent", "wont", "ive", "im", "weve", "youre", "theres", "its", "ours", "ourselves",
}

AI_PROVIDERS: Dict[str, Dict[str, object]] = {
    "OpenAI": {
        "label": "OpenAI (GPT-4o / mini)",
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        "package": "openai",
        "default_model": "gpt-4o-mini",
        "max_output_tokens": 900,
    },
    "Anthropic": {
        "label": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "package": "anthropic",
        "default_model": "claude-3-5-sonnet-20240620",
        "max_output_tokens": 900,
    },
}

st.set_page_config(page_title="EM Skills Dashboard", layout="wide")

# Sidebar: data input
st.sidebar.header("Data")
use_uploaded = st.sidebar.checkbox("Use uploaded file instead of default", value=False)
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"]) if use_uploaded else None
if st.sidebar.button("Clear cache and rerun"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer():
    if SentimentIntensityAnalyzer is None:
        return None
    return SentimentIntensityAnalyzer()


@st.cache_data(show_spinner=False)
def load_data(file) -> Tuple[pd.DataFrame, str, Dict[str, pd.DataFrame]]:
    def read_with_preferred_sheet(src):
        xls = pd.ExcelFile(src, engine="openpyxl")
        sheets = [s.strip() for s in xls.sheet_names]
        preferred = ["Quantitative", "Quantiative", "Quant"]
        chosen = None
        for name in preferred:
            for s in sheets:
                if s.lower() == name.lower():
                    chosen = s
                    break
            if chosen:
                break
        if chosen is None:
            chosen = sheets[0]
        df_quant = pd.read_excel(xls, sheet_name=chosen)
        extras: Dict[str, pd.DataFrame] = {}
        for target in ["Qualitative"]:
            for s in sheets:
                if s.lower() == target.lower():
                    extras[target] = pd.read_excel(xls, sheet_name=s)
                    break
        return df_quant, chosen, extras

    try:
        if file is None:
            path = DATA_DEFAULT_PATH
            if not os.path.exists(path):
                st.error("Default data file not found. Upload an .xlsx file from the sidebar.")
                return pd.DataFrame(), "", {}
            df, chosen_sheet, extras = read_with_preferred_sheet(path)
            source_label = path
        else:
            df, chosen_sheet, extras = read_with_preferred_sheet(file)
            source_label = "Uploaded file"
    except PermissionError:
        st.error("Permission denied when reading the workbook. Close Excel/Teams or copy the file locally and retry.")
        return pd.DataFrame(), "", {}

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for key, frame in extras.items():
        extras[key] = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    st.sidebar.caption(f"Loaded sheet: {chosen_sheet}")
    return df, source_label, extras


def detect_question_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c.strip() for c in df.columns.astype(str)]
    imp_pattern = re.compile(r"^\s*IMP\s*:\s*", re.IGNORECASE)
    perf_pattern = re.compile(r"^\s*PERF\s*:\s*", re.IGNORECASE)
    imp_cols = [c for c in cols if imp_pattern.search(c)]
    perf_cols = [c for c in cols if perf_pattern.search(c)]

    if not imp_cols or not perf_cols:
        upper_cols = [c.upper() for c in cols]

        def pick(preds):
            selected = []
            for col, upper in zip(cols, upper_cols):
                if any(p in upper for p in preds):
                    selected.append(col)
            return selected

        if not imp_cols:
            imp_cols = pick(["IMP:", "IMP ", "IMP_", "IMPORTANCE", " MOST IMPORTANT", "WHAT DO YOU VIEW"])
        if not perf_cols:
            perf_cols = pick(["PERF:", "PERF ", "PERF_", "PERFORMANCE", "HOW DO YOU FEEL", "PERFORM AGAINST"])

    if not imp_cols:
        imp_cols = [c for c in cols if "IMPORT" in c.upper() or c.strip().upper().endswith("(IMP)") or c.strip().upper().endswith("- IMPORTANCE")]
    if not perf_cols:
        perf_cols = [c for c in cols if "PERF" in c.upper() or "PERFORM" in c.upper() or c.strip().upper().endswith("(PERF)") or c.strip().upper().endswith("- PERFORMANCE")]

    return imp_cols, perf_cols


def normalize_skill_name(col: str) -> str:
    cleaned = col
    cleaned = cleaned.replace("[IMP]", "").replace("[PERF]", "").strip()
    cleaned = cleaned.replace("Importance", "").replace("Performance", "").replace(":", "").replace("-", " ")
    for prefix in ["IMP_", "PERF_", "IMP ", "PERF ", "IMP:", "PERF:"]:
        if cleaned.upper().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return " ".join(cleaned.split()).title()


def tidy_data(df: pd.DataFrame, imp_override: List[str] | None = None, perf_override: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    if imp_override is None or perf_override is None:
        auto_imp, auto_perf = detect_question_columns(df)
        imp_cols = imp_override if imp_override is not None else auto_imp
        perf_cols = perf_override if perf_override is not None else auto_perf
    else:
        imp_cols, perf_cols = imp_override, perf_override
    if not imp_cols and not perf_cols:
        return pd.DataFrame(), []

    imp_perf_set = set(imp_cols + perf_cols)
    id_cols = [c for c in df.columns if c not in imp_perf_set]

    imp_long = pd.melt(df, id_vars=id_cols, value_vars=imp_cols, var_name="question", value_name="importance") if imp_cols else pd.DataFrame()
    perf_long = pd.melt(df, id_vars=id_cols, value_vars=perf_cols, var_name="question", value_name="performance") if perf_cols else pd.DataFrame()

    if imp_long.empty and perf_long.empty:
        return pd.DataFrame(), []

    def add_skill(frame, value_col):
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["skill"] = frame["question"].astype(str).apply(normalize_skill_name)
        return frame.drop(columns=["question"]).rename(columns={value_col: value_col})

    imp_long = add_skill(imp_long, "importance")
    perf_long = add_skill(perf_long, "performance")

    if not imp_long.empty and not perf_long.empty:
        on_cols = id_cols + ["skill"]
        tidy = pd.merge(imp_long, perf_long, on=on_cols, how="outer")
    else:
        tidy = imp_long if not imp_long.empty else perf_long

    for col in ["importance", "performance"]:
        if col in tidy.columns:
            tidy[col] = pd.to_numeric(tidy[col], errors="coerce")

    if {"importance", "performance"}.issubset(tidy.columns):
        tidy["gap"] = tidy["importance"] - tidy["performance"]

    filter_candidates = []
    for col in id_cols:
        series = tidy[col]
        if series.dtype == object:
            nunique = series.nunique(dropna=True)
            if 1 < nunique <= 20:
                filter_candidates.append(col)
    return tidy, filter_candidates


def make_filters(df: pd.DataFrame, filter_cols: List[str]):
    filters = {}
    if filter_cols:
        st.sidebar.header("Filters")
    for col in filter_cols:
        values = [v for v in df[col].dropna().unique()]
        values = sorted(values, key=lambda x: str(x))
        selected = st.sidebar.multiselect(col, options=values, default=[])
        if selected:
            filters[col] = selected
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    if not filters or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for col, vals in filters.items():
        if col in df.columns:
            mask &= df[col].isin(vals)
    return df[mask].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("skill", dropna=False)
    agg = grp.agg(
        n=("skill", "count"),
        importance_mean=("importance", "mean"),
        performance_mean=("performance", "mean"),
        gap_mean=("gap", "mean") if "gap" in df.columns else ("importance", "mean"),
        importance_median=("importance", "median"),
        performance_median=("performance", "median"),
    ).reset_index()
    if "gap_mean" in agg.columns:
        agg = agg.sort_values("gap_mean", ascending=False)
    return agg


def categorize_sentiment(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "unknown"
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z']+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def tidy_qualitative(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.dropna(how="all").copy()
    if frame.empty:
        return frame

    n_rows = len(frame)
    candidate_meta = {"id", "name", "role", "team", "office"}
    id_cols: List[str] = []
    for col in frame.columns:
        col_key = str(col).strip().lower()
        if col_key in candidate_meta:
            id_cols.append(col)
            continue
        nunique = frame[col].nunique(dropna=True)
        if frame[col].dtype == object and nunique <= max(5, int(0.2 * n_rows)):
            id_cols.append(col)
    id_cols = list(dict.fromkeys(id_cols))
    question_cols = [c for c in frame.columns if c not in id_cols]
    if not question_cols:
        return pd.DataFrame()

    long = pd.melt(frame, id_vars=id_cols, value_vars=question_cols, var_name="question", value_name="response")
    long = long.dropna(subset=["response"])
    if long.empty:
        return long
    long["response"] = long["response"].astype(str).str.strip()
    long = long[long["response"].ne("")]
    if long.empty:
        return long

    analyzer = get_sentiment_analyzer()
    if analyzer is not None:
        long["sentiment"] = long["response"].apply(lambda text: analyzer.polarity_scores(text)["compound"])
    else:
        long["sentiment"] = np.nan
    long["sentiment_label"] = long["sentiment"].apply(categorize_sentiment)
    long["tokens"] = long["response"].apply(tokenize)
    return long


def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame()
    summary = df.groupby("question").agg(
        responses=("response", "count"),
        sentiment_avg=("sentiment", "mean"),
        positive_pct=("sentiment_label", lambda s: (s == "positive").mean() * 100),
        neutral_pct=("sentiment_label", lambda s: (s == "neutral").mean() * 100),
        negative_pct=("sentiment_label", lambda s: (s == "negative").mean() * 100),
    ).reset_index()
    summary[["positive_pct", "neutral_pct", "negative_pct"]] = summary[["positive_pct", "neutral_pct", "negative_pct"]].round(1)
    summary["sentiment_avg"] = summary["sentiment_avg"].round(3)
    return summary.sort_values("responses", ascending=False)


def top_terms(df: pd.DataFrame, question: str, limit: int = 15) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    subset = df[df["question"] == question]
    if subset.empty:
        return pd.DataFrame()
    counts = Counter()
    for tokens in subset["tokens"]:
        counts.update(tokens)
    most_common = counts.most_common(limit)
    return pd.DataFrame(most_common, columns=["term", "count"])


def select_example(series: pd.Series) -> str:
    if series.empty:
        return ""
    ordered = sorted(series, key=len)
    for text in ordered:
        if len(text) >= 80:
            return text
    return ordered[-1]


def summarize_skill_mentions(df: pd.DataFrame, skills: List[str]) -> pd.DataFrame:
    if df.empty or not skills:
        return pd.DataFrame()
    patterns = {skill: re.compile(r"\\b" + re.escape(skill.lower()) + r"\\b") for skill in skills}
    rows = []
    for _, row in df.iterrows():
        text = row["response"]
        lower = text.lower()
        matched = [skill for skill, pattern in patterns.items() if pattern.search(lower)]
        for skill in matched:
            rows.append({
                "skill": skill,
                "question": row["question"],
                "response": row["response"],
                "sentiment": row.get("sentiment", np.nan),
            })
    if not rows:
        return pd.DataFrame()
    mentions = pd.DataFrame(rows)
    summary = mentions.groupby("skill").agg(
        mentions=("response", "count"),
        sentiment_avg=("sentiment", "mean"),
        example=("response", select_example),
        questions=("question", lambda s: ", ".join(sorted(set(s))[:3])),
    ).reset_index()
    summary["sentiment_avg"] = summary["sentiment_avg"].round(3)
    summary["example"] = summary["example"].apply(lambda text: textwrap.shorten(text, width=160, placeholder="..."))
    return summary.sort_values("mentions", ascending=False)


def render_ai_sidebar() -> Dict[str, object]:
    with st.sidebar.expander("AI qualitative analysis (optional)", expanded=False):
        st.caption("Provide an API key only if you want a model-generated summary. Keys stay in session memory only.")
        provider_options = list(AI_PROVIDERS.keys())
        provider_labels = {key: AI_PROVIDERS[key]["label"] for key in provider_options}
        saved_provider = st.session_state.get("ai_provider", provider_options[0])
        provider_index = provider_options.index(saved_provider) if saved_provider in provider_options else 0
        provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=provider_index,
            format_func=lambda opt: provider_labels[opt],
            key="ai_provider",
        )

        models = AI_PROVIDERS[provider]["models"]
        saved_model = st.session_state.get("ai_model", AI_PROVIDERS[provider]["default_model"])
        model_index = models.index(saved_model) if saved_model in models else 0
        model = st.selectbox("Model", options=models, index=model_index, key="ai_model")

        api_key = st.text_input("API key", type="password", help="Used only at runtime; never written to disk.", key="ai_api_key")
        max_responses = st.slider(
            "Responses to include in prompt",
            min_value=5,
            max_value=50,
            value=st.session_state.get("ai_max_responses", 20),
            key="ai_max_responses",
        )
        temperature = st.slider(
            "Creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("ai_temperature", 0.3),
            step=0.05,
            key="ai_temperature",
        )
        st.caption("Install `openai` or `anthropic` if you plan to use those providers. Network access is required when requesting an AI summary.")

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key.strip(),
        "max_responses": int(max_responses),
        "temperature": float(temperature),
    }


def prepare_ai_responses(question_subset: pd.DataFrame, max_responses: int) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if question_subset.empty:
        return results
    limited = question_subset.head(max_responses)
    for idx, (_, row) in enumerate(limited.iterrows(), start=1):
        meta_parts = []
        for col in ["Role", "Name", "Id"]:
            if col in row and pd.notna(row[col]):
                meta_parts.append(f"{col}:{row[col]}")
        sentiment = row.get("sentiment_label")
        sentiment_note = f" | sentiment:{sentiment}" if isinstance(sentiment, str) and sentiment not in {None, "unknown"} else ""
        text = textwrap.shorten(str(row["response"]), width=600, placeholder="...")
        results.append(
            {
                "header": ", ".join(meta_parts) if meta_parts else f"Response {idx}",
                "text": text,
                "sentiment_note": sentiment_note,
            }
        )
    return results


def build_ai_prompt(
    question: str,
    responses: List[Dict[str, str]],
    top_terms: pd.DataFrame | None,
    quant_summary: pd.DataFrame | None,
    skill_mentions: pd.DataFrame | None,
) -> str:
    lines: List[str] = []
    lines.append("You are an insights analyst reviewing engagement manager survey feedback.")
    lines.append("Summarize dominant themes, celebrate strengths, note skill gaps, and recommend next steps.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")

    if quant_summary is not None and not quant_summary.empty:
        lines.append("Quantitative skill gaps (top 5 by average gap):")
        for _, row in quant_summary.head(5).iterrows():
            gap = row.get("gap_mean")
            imp = row.get("importance_mean")
            perf = row.get("performance_mean")
            n = row.get("n")
            lines.append(
                f"- {row['skill']}: gap={gap:+.2f}, importance={imp:.2f}, performance={perf:.2f} (responses={int(n)})"
            )
        lines.append("")

    if skill_mentions is not None and not skill_mentions.empty:
        lines.append("Skills referenced in qualitative feedback:")
        for _, row in skill_mentions.head(5).iterrows():
            sentiment = row.get("sentiment_avg")
            sentiment_txt = f"{sentiment:+.3f}" if pd.notna(sentiment) else "n/a"
            lines.append(
                f"- {row['skill']}: mentions={int(row['mentions'])}, sentiment_avg={sentiment_txt}, example={row['example']}"
            )
        lines.append("")

    if top_terms is not None and not top_terms.empty:
        terms_line = ", ".join(f"{term}({count})" for term, count in top_terms.head(15).itertuples(index=False))
        lines.append(f"Top keywords: {terms_line}")
        lines.append("")

    lines.append("Sample responses (truncated):")
    for record in responses:
        lines.append(f"- {record['header']}{record['sentiment_note']}: {record['text']}")
    lines.append("")
    lines.append("Return a concise narrative (<200 words) covering key themes, perceived capability gaps, sentiment balance, and recommended actions.")
    return "\n".join(lines)


def summarize_with_ai(settings: Dict[str, object], prompt: str) -> str:
    provider = settings["provider"]
    model = settings["model"]
    api_key = settings["api_key"]
    temperature = settings.get("temperature", 0.3)

    if not api_key:
        raise ValueError("API key is required for AI summarization.")

    if provider == "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("Install the `openai` package to use OpenAI models.") from exc

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=int(AI_PROVIDERS[provider]["max_output_tokens"]),
            temperature=temperature,
        )
        return response.output_text.strip()

    if provider == "Anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("Install the `anthropic` package to use Claude models.") from exc

        client = Anthropic(api_key=api_key)
        completion = client.messages.create(
            model=model,
            max_tokens=int(AI_PROVIDERS[provider]["max_output_tokens"]),
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in getattr(completion, "content", []) if getattr(block, "type", "") == "text"]
        if not parts:
            return ""
        return "\n".join(parts).strip()

    raise ValueError(f"Unsupported provider: {provider}")


def main():
    df_raw, data_label, extras = load_data(uploaded if use_uploaded else None)
    if df_raw.empty:
        st.stop()

    df_qual_raw = extras.get("Qualitative") if extras else None

    auto_imp, auto_perf = detect_question_columns(df_raw)
    with st.sidebar.expander("Detection & Mapping", expanded=not (auto_imp or auto_perf)):
        st.caption("If auto-detection misses your headers, pick them below.")
        st.write("Detected importance:", auto_imp if auto_imp else "None")
        st.write("Detected performance:", auto_perf if auto_perf else "None")
        all_cols = [str(c) for c in df_raw.columns]
        imp_override = st.multiselect("Importance columns (override)", options=all_cols, default=auto_imp)
        perf_override = st.multiselect("Performance columns (override)", options=all_cols, default=auto_perf)

    tidy, filter_cols = tidy_data(df_raw, imp_override=imp_override, perf_override=perf_override)
    if tidy.empty:
        st.error("Could not detect IMP/PERF columns. Please verify the column names in the Excel file or use the overrides in the sidebar.")
        with st.expander("Column headers preview"):
            st.write(list(df_raw.columns))
        st.stop()

    filters = make_filters(tidy, filter_cols)
    df = apply_filters(tidy, filters)
    ai_settings = render_ai_sidebar()

    st.title("EM Survey: Importance vs Performance")
    st.caption("Positive gap = importance exceeds performance (potential development opportunity). Negative gap = performance exceeds importance.")
    with st.expander("Detection summary", expanded=False):
        st.write({
            "rows": int(len(df)),
            "skills": int(df["skill"].nunique()),
            "importance_nonnull": int(df["importance"].notna().sum()) if "importance" in df.columns else 0,
            "performance_nonnull": int(df["performance"].notna().sum()) if "performance" in df.columns else 0,
        })

    agg = aggregate(df)
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Skill Gap Ranking")
        st.dataframe(
            agg[["skill", "n", "importance_mean", "performance_mean", "gap_mean", "importance_median", "performance_median"]]
            .rename(columns={
                "n": "Responses",
                "importance_mean": "Importance (avg)",
                "performance_mean": "Performance (avg)",
                "gap_mean": "Gap (avg)",
            })
            .style.format({
                "Importance (avg)": "{:.2f}",
                "Performance (avg)": "{:.2f}",
                "Gap (avg)": "{:+.2f}",
            }),
            use_container_width=True,
        )

    with cols[1]:
        st.subheader("Download Aggregates")
        csv = agg.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="em_skill_gaps.csv", mime="text/csv")

    if {"importance", "performance"}.issubset(df.columns):
        scatter_df = agg
        fig = px.scatter(
            scatter_df,
            x="performance_mean",
            y="importance_mean",
            color="gap_mean",
            color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
            hover_data={"skill": True, "gap_mean": ":+.2f", "performance_mean": ":.2f", "importance_mean": ":.2f", "n": True},
            text="skill",
        )
        fig.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
        fig.update_layout(title="Importance vs Performance (avg)", xaxis_title="Performance (avg)", yaxis_title="Importance (avg)")
        min_axis = float(np.nanmin([scatter_df["performance_mean"].min(), scatter_df["importance_mean"].min()]))
        max_axis = float(np.nanmax([scatter_df["performance_mean"].max(), scatter_df["importance_mean"].max()]))
        fig.add_shape(type="line", x0=min_axis, y0=min_axis, x1=max_axis, y1=max_axis, line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Radar: Compare Skills")
    top_skills = agg["skill"].tolist()
    selected_skills = st.multiselect("Select skills (optional)", options=top_skills, default=top_skills[:5])
    radar_basis = agg
    if selected_skills:
        radar_basis = radar_basis[radar_basis["skill"].isin(selected_skills)]
    if not radar_basis.empty:
        radar_long = pd.melt(radar_basis, id_vars=["skill"], value_vars=["importance_mean", "performance_mean"], var_name="metric", value_name="value")
        fig_radar = px.line_polar(radar_long, r="value", theta="skill", color="metric", line_close=True, range_r=[0, 7])
        fig_radar.update_layout(legend_title_text="Metric")
        st.plotly_chart(fig_radar, use_container_width=True)

    if "gap" in df.columns:
        st.subheader("Gap Distribution by Skill")
        fig_box = px.box(df, x="skill", y="gap", points="all")
        fig_box.update_layout(xaxis_title="Skill", yaxis_title="Gap (importance - performance)")
        st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Data Preview"):
        st.write("Tidy data sample:")
        st.dataframe(df.head(50), use_container_width=True)

    if df_qual_raw is not None and not df_qual_raw.empty:
        st.header("Qualitative Insights")
        qual_tidy = tidy_qualitative(df_qual_raw)
        if qual_tidy.empty:
            st.info("No qualitative responses detected in the workbook.")
        else:
            if SentimentIntensityAnalyzer is None:
                st.warning("Install vaderSentiment to enable sentiment scoring (already listed in requirements).")
            qual_filtered = apply_filters(qual_tidy, filters)
            if qual_filtered.empty:
                st.info("No qualitative responses match the selected filters.")
            else:
                sent_summary = sentiment_summary(qual_filtered)
                if not sent_summary.empty:
                    st.subheader("Sentiment by Question")
                    st.dataframe(
                        sent_summary.rename(columns={
                            "sentiment_avg": "Sentiment (avg)",
                            "positive_pct": "% Positive",
                            "neutral_pct": "% Neutral",
                            "negative_pct": "% Negative",
                        }).style.format({
                            "Sentiment (avg)": "{:+.3f}",
                            "% Positive": "{:.1f}",
                            "% Neutral": "{:.1f}",
                            "% Negative": "{:.1f}",
                        }),
                        use_container_width=True,
                    )

                qual_skill_summary = summarize_skill_mentions(qual_filtered, agg["skill"].tolist())
                if not qual_skill_summary.empty:
                    st.subheader("Skill Mentions in Written Feedback")
                    merged = qual_skill_summary.merge(agg[["skill", "gap_mean"]], on="skill", how="left")
                    st.dataframe(
                        merged.rename(columns={
                            "mentions": "Mentions",
                            "sentiment_avg": "Sentiment (avg)",
                            "gap_mean": "Gap (avg)",
                        }).style.format({
                            "Sentiment (avg)": "{:+.3f}",
                            "Gap (avg)": "{:+.2f}",
                        }),
                        use_container_width=True,
                    )

                st.subheader("Question Deep Dive")
                question_options = qual_filtered["question"].unique().tolist()
                selected_question = st.selectbox("Select a question", options=question_options)
                question_subset = qual_filtered[qual_filtered["question"] == selected_question]
                st.caption(f"Responses: {len(question_subset)}")
                if "sentiment" in question_subset.columns and question_subset["sentiment"].notna().any():
                    st.metric("Average sentiment", f"{question_subset['sentiment'].mean():+.3f}")

                terms = top_terms(qual_filtered, selected_question)
                if not terms.empty:
                    st.markdown("**Top terms mentioned**")
                    st.table(terms)

                responses_for_ai = prepare_ai_responses(question_subset, ai_settings["max_responses"])
                if ai_settings.get("api_key"):
                    if not responses_for_ai:
                        st.info("Add or select responses to enable AI summarization.")
                    else:
                        if st.button("Generate AI summary", key=f"ai_summary_{abs(hash(selected_question))}"):
                            with st.spinner("Requesting AI summary..."):
                                try:
                                    prompt_text = build_ai_prompt(
                                        selected_question,
                                        responses_for_ai,
                                        terms,
                                        agg,
                                        qual_skill_summary,
                                    )
                                    ai_output = summarize_with_ai(ai_settings, prompt_text)
                                except ModuleNotFoundError as exc:
                                    st.error(str(exc))
                                except Exception as exc:
                                    st.error(f"AI request failed: {exc}")
                                else:
                                    st.markdown("**AI-generated summary**")
                                    st.write(ai_output if ai_output else "No summary returned.")
                else:
                    st.caption("Add an API key in the sidebar to enable AI-generated summaries.")

                display_cols = [c for c in ["Name", "Role", "Id"] if c in question_subset.columns]
                display_cols += ["response", "sentiment_label"]
                st.markdown("**Sample responses**")
                st.dataframe(
                    question_subset[display_cols]
                    .rename(columns={"response": "Response", "sentiment_label": "Sentiment"}),
                    use_container_width=True,
                )
    else:
        st.info("No Qualitative sheet detected. Add a sheet named 'Qualitative' to enable written feedback analysis.")

    st.sidebar.divider()
    st.sidebar.caption("Data path: " + ("Uploaded file" if use_uploaded and uploaded else data_label or DATA_DEFAULT_PATH))


if __name__ == "__main__":
    main()

