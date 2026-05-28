from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.storage import get_sqlite_path, load_dataframe_from_sqlite
from src.utils import load_config


def build_comparison_df(config: dict, window: int = 3) -> pd.DataFrame:
    """Return a DataFrame with monthly avg_sentiment, ppi_value and rolling averages."""
    news_monthly_path = Path("artifacts/eda/news_eda_monthly.csv")
    if not news_monthly_path.exists():
        raise FileNotFoundError(f"Missing news monthly CSV: {news_monthly_path}")

    news_df = pd.read_csv(news_monthly_path)
    if news_df.empty or "avg_sentiment" not in news_df.columns:
        raise ValueError("news_eda_monthly.csv does not contain avg_sentiment")

    news_df = news_df[["month", "avg_sentiment"]].copy()
    news_df["month"] = pd.to_datetime(news_df["month"], errors="coerce")
    news_df["avg_sentiment"] = pd.to_numeric(news_df["avg_sentiment"], errors="coerce")
    news_df = news_df.dropna(subset=["month", "avg_sentiment"]) 

    news_df = (
        news_df.groupby("month", as_index=False)["avg_sentiment"].mean().sort_values("month").reset_index(drop=True)
    )

    config_obj = config
    fred_path = get_sqlite_path(config_obj)
    fred_df = load_dataframe_from_sqlite(fred_path, "fred_series")
    if fred_df.empty or fred_df.shape[1] < 2:
        raise ValueError(f"No fred_series table found in sqlite DB: {fred_path}")

    fred_df = fred_df.iloc[:, :2].copy()
    fred_df.columns = ["date", "ppi_value"]
    fred_df["date"] = pd.to_datetime(fred_df["date"], errors="coerce")
    fred_df["ppi_value"] = pd.to_numeric(fred_df["ppi_value"], errors="coerce")
    fred_df = fred_df.dropna(subset=["date", "ppi_value"]) 
    fred_df["month"] = fred_df["date"].dt.to_period("M").dt.to_timestamp()
    fred_df = fred_df.groupby("month", as_index=False)["ppi_value"].mean().sort_values("month").reset_index(drop=True)

    comparison_df = news_df.merge(fred_df, on="month", how="inner").sort_values("month").reset_index(drop=True)
    if comparison_df.empty:
        raise ValueError("No overlapping months between news and FRED series")

    comparison_df["sentiment_roll"] = comparison_df["avg_sentiment"].rolling(window=window, min_periods=window).mean()
    comparison_df["ppi_roll"] = comparison_df["ppi_value"].rolling(window=window, min_periods=window).mean()
    comparison_df = comparison_df.dropna(subset=["sentiment_roll", "ppi_roll"]).reset_index(drop=True)
    return comparison_df


def save_dual_axis_chart(df: pd.DataFrame, out_path: Path, window: int = 3) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corr = df["sentiment_roll"].corr(df["ppi_roll"])

    fig, ax1 = plt.subplots(figsize=(12.5, 6))
    ax2 = ax1.twinx()

    ax1.plot(df["month"], df["sentiment_roll"], color="#60a5fa", linewidth=2.2, label=f"News sentiment ({window}M rolling avg)")
    ax2.plot(df["month"], df["ppi_roll"], color="#f59e0b", linewidth=2.2, label=f"FRED PPI ({window}M rolling avg)")

    corr_text = "n/a" if pd.isna(corr) else f"{corr:.2f}"
    ax1.set_title(f"{window}-Month Rolling Average: News Sentiment vs FRED PPI (corr={corr_text})")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Rolling avg sentiment")
    ax2.set_ylabel("Rolling avg FRED PPI")
    ax1.grid(alpha=0.22)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return corr


def save_normalized_overlay(df: pd.DataFrame, out_path: Path, window: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm_sent = (df["sentiment_roll"] - df["sentiment_roll"].mean()) / df["sentiment_roll"].std()
    norm_ppi = (df["ppi_roll"] - df["ppi_roll"].mean()) / df["ppi_roll"].std()

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.plot(df["month"], norm_sent, color="#60a5fa", linewidth=2.2, label="Sentiment (normalized)")
    ax.plot(df["month"], norm_ppi, color="#f59e0b", linewidth=2.2, label="FRED PPI (normalized)")
    ax.set_title(f"{window}-Month Rolling Average (normalized): Sentiment vs FRED PPI")
    ax.set_xlabel("Month")
    ax.set_ylabel("Z-score")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "configs/config.yaml", window: int = 3) -> None:
    config = load_config(config_path)
    df = build_comparison_df(config, window=window)

    plots_dir = Path("artifacts/eda/plots")
    dual_path = plots_dir / "news_fred_rolling_average.png"
    norm_path = plots_dir / "news_fred_rolling_normalized.png"

    corr = save_dual_axis_chart(df, dual_path, window=window)
    save_normalized_overlay(df, norm_path, window=window)

    print(f"Saved charts to: {dual_path} and {norm_path}; rolling correlation={corr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()
    main(config_path=args.config, window=args.window)
