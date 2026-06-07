from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Model Comparisons",
    page_icon=":bar_chart:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)

MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
AI_LIFT_PATH = METRICS_DIR / "ai_lift_comparison.csv"
RECENT_MODEL_COMPARISON_PATH = METRICS_DIR / "recent_period_model_comparison.csv"
RECENT_AI_LIFT_PATH = METRICS_DIR / "recent_period_ai_lift_comparison.csv"

PREFERRED_FEATURE_ORDER = [
    "baseline",
    "baseline_with_persistence",
    "ai_compact",
    "ai_compact_with_persistence",
]
PREFERRED_MODEL_ORDER = ["logistic_regression", "xgboost"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fetch_json(endpoint: str, timeout: int = 60) -> dict[str, Any]:
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_tables_from_api() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = fetch_json("/model-comparisons")
    tables = payload.get("tables", {})

    model_df = pd.DataFrame(tables.get("model_comparison", []))
    ai_lift_df = pd.DataFrame(tables.get("ai_lift_comparison", []))
    recent_model_df = pd.DataFrame(tables.get("recent_period_model_comparison", []))
    recent_ai_lift_df = pd.DataFrame(tables.get("recent_period_ai_lift_comparison", []))
    return model_df, ai_lift_df, recent_model_df, recent_ai_lift_df


def render_missing(path: Path) -> None:
    st.warning(f"Missing artifact: {path.as_posix()}")


def add_delta_color(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: "#2E8B57" if value >= 0 else "#B22222")


def humanize_token(value: str) -> str:
    text = value.replace("_", " ").strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def humanize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: humanize_token(col)
        for col in df.columns
    }
    return df.rename(columns=renamed)


def order_grouped_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["feature_order"] = ordered["feature_set"].apply(
        lambda feature: PREFERRED_FEATURE_ORDER.index(feature)
        if feature in PREFERRED_FEATURE_ORDER
        else len(PREFERRED_FEATURE_ORDER)
    )
    ordered["model_order"] = ordered["model_name"].apply(
        lambda model: PREFERRED_MODEL_ORDER.index(model)
        if model in PREFERRED_MODEL_ORDER
        else len(PREFERRED_MODEL_ORDER)
    )
    ordered = ordered.sort_values(["feature_order", "model_order", metric_column], ascending=[True, True, False])
    return ordered.drop(columns=["feature_order", "model_order"])


st.title("Model Comparison Explorer")
st.write("Compare baseline and AI-enhanced model runs from metrics artifacts.")

with st.sidebar:
    st.header("Controls")
    data_source = st.radio(
        "Data source",
        options=["API", "Local CSV files"],
        index=0,
    )
    st.caption(f"API base URL: {API_BASE_URL}")

    metric_options = {
        "Test Macro F1": "test_macro_f1",
        "Test Accuracy": "test_accuracy",
        "Validation Macro F1": "validation_macro_f1",
        "Validation Accuracy": "validation_accuracy",
    }
    metric_label = st.selectbox(
        "Ranking metric",
        options=list(metric_options.keys()),
        index=0,
    )
    metric_column = metric_options[metric_label]
    leaderboard_mode = st.radio(
        "Leaderboard mode",
        options=["Highest score", "Grouped by feature + model"],
        index=0,
    )
    if leaderboard_mode == "Grouped by feature + model":
        st.caption("Grouped mode keeps one best row per feature/model pair.")

if data_source == "API":
    try:
        model_df, ai_lift_df, recent_model_df, recent_ai_lift_df = load_tables_from_api()
    except Exception as exc:
        st.warning(f"API load failed, falling back to local CSV files: {exc}")
        model_df = load_csv(MODEL_COMPARISON_PATH)
        ai_lift_df = load_csv(AI_LIFT_PATH)
        recent_model_df = load_csv(RECENT_MODEL_COMPARISON_PATH)
        recent_ai_lift_df = load_csv(RECENT_AI_LIFT_PATH)
else:
    model_df = load_csv(MODEL_COMPARISON_PATH)
    ai_lift_df = load_csv(AI_LIFT_PATH)
    recent_model_df = load_csv(RECENT_MODEL_COMPARISON_PATH)
    recent_ai_lift_df = load_csv(RECENT_AI_LIFT_PATH)

if model_df.empty:
    render_missing(MODEL_COMPARISON_PATH)
    st.stop()

filtered_model_df = model_df.copy()
filtered_model_df = filtered_model_df[filtered_model_df["status"] == "ok"].copy()
filtered_model_df = filtered_model_df.sort_values(metric_column, ascending=False)

if leaderboard_mode == "Grouped by feature + model":
    grouped_idx = filtered_model_df.groupby(["feature_set", "model_name"])[metric_column].idxmax()
    grouped_df = filtered_model_df.loc[grouped_idx].copy()
    grouped_df = grouped_df[grouped_df["feature_set"].isin(PREFERRED_FEATURE_ORDER)].copy()

    if grouped_df.empty:
        st.warning("No grouped rows available for the selected metric.")
        st.stop()

    grouped_df = order_grouped_rows(grouped_df)
    best_row = grouped_df.iloc[0]
    best_model_label = f"{humanize_token(best_row['feature_set'])} + {humanize_token(best_row['model_name'])}"
    compared_count = int(len(grouped_df))
else:
    best_row = filtered_model_df.iloc[0]
    best_model_label = f"{humanize_token(best_row['feature_set'])} + {humanize_token(best_row['model_name'])}"
    compared_count = int(len(filtered_model_df))

col1, col2, col3 = st.columns(3)
col1.metric("Best configuration", best_model_label)
col2.metric("Best metric value", f"{best_row[metric_column]:.4f}")
col3.metric("Configs compared", compared_count)

st.subheader("Overall Model Leaderboard")
leaderboard_columns = [
    "feature_set",
    "model_name",
    "feature_count",
    "validation_accuracy",
    "validation_macro_f1",
    "test_accuracy",
    "test_macro_f1",
]
if leaderboard_mode == "Grouped by feature + model":
    grouped_display_df = grouped_df[
        [
            "feature_set",
            "model_name",
            "feature_count",
            "validation_accuracy",
            "validation_macro_f1",
            "test_accuracy",
            "test_macro_f1",
        ]
    ]
    grouped_display_df["feature_set"] = grouped_display_df["feature_set"].map(humanize_token)
    grouped_display_df["model_name"] = grouped_display_df["model_name"].map(humanize_token)
    st.dataframe(humanize_columns(grouped_display_df), use_container_width=True, hide_index=True)
else:
    ranking_df = filtered_model_df[leaderboard_columns].copy()
    ranking_df["feature_set"] = ranking_df["feature_set"].map(humanize_token)
    ranking_df["model_name"] = ranking_df["model_name"].map(humanize_token)
    st.dataframe(humanize_columns(ranking_df), use_container_width=True, hide_index=True)

st.subheader("Top Performers")
top_n = st.slider("Top rows", min_value=3, max_value=20, value=8, step=1)
if leaderboard_mode == "Grouped by feature + model":
    top_grouped_df = grouped_df.copy()
    top_grouped_df["series"] = top_grouped_df["feature_set"].map(humanize_token) + " + " + top_grouped_df["model_name"].map(humanize_token)
    st.bar_chart(
        top_grouped_df.head(top_n).set_index("series")[[metric_column]],
        use_container_width=True,
    )
else:
    top_ranked_df = filtered_model_df.copy()
    top_ranked_df["series"] = top_ranked_df["feature_set"].map(humanize_token) + " + " + top_ranked_df["model_name"].map(humanize_token)
    st.bar_chart(
        top_ranked_df.head(top_n).set_index("series")[[metric_column]],
        use_container_width=True,
    )

if not ai_lift_df.empty:
    st.subheader("AI Lift Summary (Overall)")

    positive_lift_count = int((ai_lift_df["test_macro_f1_lift"] > 0).sum())
    neutral_lift_count = int((ai_lift_df["test_macro_f1_lift"] == 0).sum())
    negative_lift_count = int((ai_lift_df["test_macro_f1_lift"] < 0).sum())

    l1, l2, l3 = st.columns(3)
    l1.metric("Positive AI test macro F1 lift", positive_lift_count)
    l2.metric("Neutral lift", neutral_lift_count)
    l3.metric("Negative lift", negative_lift_count)

    lift_display_df = ai_lift_df.copy()
    lift_display_df["model_name"] = lift_display_df["model_name"].map(humanize_token)
    lift_display_df["comparison_type"] = lift_display_df["comparison_type"].map(humanize_token)
    lift_display_df["baseline_feature_set"] = lift_display_df["baseline_feature_set"].map(humanize_token)
    lift_display_df["ai_feature_set"] = lift_display_df["ai_feature_set"].map(humanize_token)

    styled_lift = humanize_columns(lift_display_df).style.apply(
        lambda col: [f"color: {c}" for c in add_delta_color(col)] if "lift" in col.name.lower() else ["" for _ in col],
        axis=0,
    )
    st.dataframe(styled_lift, use_container_width=True, hide_index=True)
else:
    render_missing(AI_LIFT_PATH)
