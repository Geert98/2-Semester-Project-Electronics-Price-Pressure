from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.storage import load_dataframe_from_mongo
from src.text_cleaning import clean_news_text
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)


def _build_eda_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    frame = df.copy()

    if "published_at" not in frame.columns:
        frame["published_at"] = pd.NaT
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame = frame.dropna(subset=["published_at"])

    if "clean_text" not in frame.columns:
        frame["clean_text"] = ""
    if "content" not in frame.columns:
        frame["content"] = ""
    if "title" not in frame.columns:
        frame["title"] = ""

    frame["clean_text"] = frame["clean_text"].fillna("")
    frame["content"] = frame["content"].fillna("")
    frame["title"] = frame["title"].fillna("")

    fallback_text = frame["content"].where(frame["content"].str.strip() != "", frame["title"])
    frame["analysis_text"] = frame["clean_text"].where(frame["clean_text"].str.strip() != "", fallback_text)
    frame["analysis_text"] = frame["analysis_text"].apply(clean_news_text)

    analyzer = SentimentIntensityAnalyzer()
    frame["sentiment"] = frame["analysis_text"].apply(
        lambda value: analyzer.polarity_scores(value)["compound"] if isinstance(value, str) and value.strip() else 0.0
    )
    frame["article_char_count"] = frame["analysis_text"].str.len()
    frame["article_word_count"] = frame["analysis_text"].str.split().str.len().fillna(0).astype(int)
    frame["month"] = frame["published_at"].dt.to_period("M").dt.to_timestamp()

    return frame.sort_values("published_at").reset_index(drop=True)


def _safe_hist(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(series.dropna(), bins=30, color="#1f4e79", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _safe_timeseries_plot(df: pd.DataFrame, value_col: str, title: str, ylabel: str, output_path: Path) -> None:
    if df.empty or value_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["month"], df[value_col], marker="o", linewidth=2, color="#8a4b08")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_eda(config_path: str = "configs/config.yaml", output_dir: str | Path = "artifacts/eda") -> pd.DataFrame:
    config = load_config(config_path)
    collection_name = config["storage"]["mongo"]["test_clean_news_collection"]

    output_path = Path(output_dir)
    plots_path = output_path / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    df = load_dataframe_from_mongo(config, collection_name, sort_by="published_at")
    if df.empty:
        logger.warning("No rows found in MongoDB collection %s", collection_name)
        return df

    eda_df = _build_eda_frame(df)
    if eda_df.empty:
        logger.warning("No usable cleaned news rows after timestamp parsing in %s", collection_name)
        return eda_df

    monthly = (
        eda_df.groupby("month", as_index=False)
        .agg(
            article_count=("title", "count"),
            avg_sentiment=("sentiment", "mean"),
            median_sentiment=("sentiment", "median"),
            avg_char_count=("article_char_count", "mean"),
            avg_word_count=("article_word_count", "mean"),
            median_word_count=("article_word_count", "median"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    distribution_summary = pd.DataFrame(
        [
            {
                "metric": "sentiment",
                "mean": eda_df["sentiment"].mean(),
                "median": eda_df["sentiment"].median(),
                "std": eda_df["sentiment"].std(),
                "min": eda_df["sentiment"].min(),
                "max": eda_df["sentiment"].max(),
            },
            {
                "metric": "article_char_count",
                "mean": eda_df["article_char_count"].mean(),
                "median": eda_df["article_char_count"].median(),
                "std": eda_df["article_char_count"].std(),
                "min": eda_df["article_char_count"].min(),
                "max": eda_df["article_char_count"].max(),
            },
            {
                "metric": "article_word_count",
                "mean": eda_df["article_word_count"].mean(),
                "median": eda_df["article_word_count"].median(),
                "std": eda_df["article_word_count"].std(),
                "min": eda_df["article_word_count"].min(),
                "max": eda_df["article_word_count"].max(),
            },
        ]
    )

    eda_df.to_csv(output_path / "news_eda_rows.csv", index=False)
    monthly.to_csv(output_path / "news_eda_monthly.csv", index=False)
    distribution_summary.to_csv(output_path / "news_eda_distribution_summary.csv", index=False)

    summary = {
        "collection": collection_name,
        "row_count": int(len(eda_df)),
        "month_count": int(monthly["month"].nunique()),
        "sentiment_mean": float(eda_df["sentiment"].mean()),
        "sentiment_median": float(eda_df["sentiment"].median()),
        "avg_char_count": float(eda_df["article_char_count"].mean()),
        "avg_word_count": float(eda_df["article_word_count"].mean()),
        "monthly_summary_path": str(output_path / "news_eda_monthly.csv"),
        "distribution_summary_path": str(output_path / "news_eda_distribution_summary.csv"),
    }
    with open(output_path / "news_eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _safe_hist(eda_df["sentiment"], "Sentiment Distribution", "VADER compound sentiment", plots_path / "sentiment_distribution.png")
    _safe_hist(eda_df["article_word_count"], "Article Length Distribution", "Word count", plots_path / "article_length_distribution.png")
    _safe_timeseries_plot(monthly, "avg_sentiment", "Monthly Average Sentiment", "Average sentiment", plots_path / "monthly_avg_sentiment.png")
    _safe_timeseries_plot(monthly, "avg_word_count", "Monthly Average Article Length", "Average word count", plots_path / "monthly_avg_article_length.png")

    logger.info("Saved EDA outputs to %s", output_path)
    return eda_df


if __name__ == "__main__":
    setup_env()
    logging.basicConfig(level=logging.INFO)
    run_eda()