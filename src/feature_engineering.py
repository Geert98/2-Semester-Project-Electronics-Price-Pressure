from __future__ import annotations

# This script is responsible for transforming processed news data and
# structured FRED data into a model-ready monthly feature table.
#
# In the overall pipeline, this file is where the raw/processed inputs become
# actual machine learning features and labels.
#
# The script:
# 1. loads the cleaned news dataset
# 2. loads the FRED PPI time series
# 3. computes article-level sentiment scores
# 4. creates keyword indicator columns
# 5. aggregates news data to the monthly level
# 6. creates lagged and rolling PPI features
# 7. creates the next-month prediction target
# 8. assigns target classes using quantile-based thresholds
# 9. saves the final model table
#
# The output of this script is the central training dataset used by the model.

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.storage import get_sqlite_path, load_dataframe_from_mongo, load_dataframe_from_sqlite
from src.utils import load_config

logger = logging.getLogger(__name__)


def _compute_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """
    Compute the VADER compound sentiment score for a piece of text.

    Why this function exists:
    - The project uses news titles as a lightweight textual signal.
    - Sentiment is one of the features used to describe market-related news pressure.
    - VADER is simple, fast, and sufficient for an MVP pipeline.

    Parameters
    ----------
    text : str
        Cleaned article title text.
    analyzer : SentimentIntensityAnalyzer
        Initialized VADER analyzer object.

    Returns
    -------
    float
        Compound sentiment score in the range [-1, 1].
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return analyzer.polarity_scores(text)["compound"]


def _contains_keyword(text: str, keyword: str) -> int:
    """
    Check whether a keyword appears in a text string.

    Why this function exists:
    - The project uses simple keyword frequency signals as interpretable text features.
    - This keeps the feature engineering lightweight and easy to explain.

    Parameters
    ----------
    text : str
        Text to search.
    keyword : str
        Keyword to look for.

    Returns
    -------
    int
        1 if the keyword is found, otherwise 0.
    """
    if not isinstance(text, str):
        return 0
    return int(keyword.lower() in text.lower())


def _assign_target_class(value: float, low_cut: float, high_cut: float) -> str | float:
    """
    Assign a pressure class based on next-month percentage change.

    Why this function exists:
    - The project models the problem as a 3-class classification task:
      low, medium, or high price pressure.
    - Quantile-based thresholds are used instead of fixed thresholds in order
      to create a more balanced class distribution.

    Parameters
    ----------
    value : float
        Next-month percentage change value.
    low_cut : float
        Lower quantile threshold.
    high_cut : float
        Upper quantile threshold.

    Returns
    -------
    str | float
        Target class label or NaN if the input is missing.
    """
    if pd.isna(value):
        return np.nan
    if value <= low_cut:
        return "low"
    if value <= high_cut:
        return "medium"
    return "high"


def build_feature_table(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Build the monthly model table by combining processed news data and FRED data.

    Why this function exists:
    - The machine learning model needs one row per time period.
    - This function converts article-level data into monthly aggregated features.
    - It also adds lagged and rolling market features and constructs the target variable.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Final model-ready feature table.
    """
    # Load configuration and initialize the sentiment analyzer.
    config = load_config(config_path)
    analyzer = SentimentIntensityAnalyzer()

    # Load the cleaned news dataset and the FRED table from persistent storage.
    news_collection = config["storage"]["mongo"]["clean_news_collection"]
    news_df = load_dataframe_from_mongo(config, news_collection, sort_by="published_at")

    fred_path = get_sqlite_path(config)
    fred_df = load_dataframe_from_sqlite(fred_path, "fred_series")

    if fred_df.empty:
        raise ValueError(f"No FRED data found in SQLite database: {fred_path}")

    # Standardize FRED columns and data types.
    fred_df.columns = ["date", "ppi_value"]
    fred_df["date"] = pd.to_datetime(fred_df["date"], errors="coerce")
    fred_df["ppi_value"] = pd.to_numeric(fred_df["ppi_value"], errors="coerce")

    # Remove invalid rows and sort the time series.
    fred_df = fred_df.dropna(subset=["date", "ppi_value"]).sort_values("date").reset_index(drop=True)

    # Create a monthly timestamp key so FRED can be joined with monthly news features.
    fred_df["month"] = fred_df["date"].dt.to_period("M").dt.to_timestamp()

    # Handle the case where no cleaned news data is available.
    # In that case, the model table will still be built from PPI data only.
    if news_df.empty:
        logger.warning("No cleaned news rows found. Building feature table from PPI only.")
        monthly_news = pd.DataFrame({"month": fred_df["month"].unique()})
    else:
        # Standardize date fields in the processed news data.
        news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce")
        news_df["month"] = pd.to_datetime(news_df["month"], errors="coerce")

        # Compute article-level sentiment and a binary negative indicator.
        news_df["sentiment"] = news_df["clean_text"].apply(lambda x: _compute_sentiment(x, analyzer))
        news_df["is_negative"] = (news_df["sentiment"] < 0).astype(int)

        # Create one binary keyword column per configured keyword.
        keywords = config["features"]["keywords"]
        for kw in keywords:
            col_name = f"kw_{kw.replace(' ', '_')}"
            news_df[col_name] = news_df["clean_text"].apply(
                lambda x, keyword=kw: _contains_keyword(x, keyword)
            )

        # Define the monthly aggregation logic.
        # The goal is to summarize all article-level signals into one row per month.
        agg_dict = {
            "title": "count",
            "sentiment": "mean",
            "is_negative": "mean",
            "title_len": "mean",
            "source": pd.Series.nunique,
        }

        # Add monthly sums for each keyword feature.
        for kw in keywords:
            agg_dict[f"kw_{kw.replace(' ', '_')}"] = "sum"

        # Aggregate article-level news data to monthly feature level.
        monthly_news = (
            news_df.groupby("month")
            .agg(agg_dict)
            .reset_index()
            .rename(
                columns={
                    "title": "article_count",
                    "sentiment": "avg_sentiment",
                    "is_negative": "negative_share",
                    "title_len": "avg_title_len",
                    "source": "unique_sources",
                }
            )
        )

    # Create the monthly PPI table.
    monthly_ppi = (
        fred_df[["month", "ppi_value"]]
        .drop_duplicates()
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Merge the structured PPI features with the aggregated monthly news features.
    df = monthly_ppi.merge(monthly_news, on="month", how="left")

    # Replace missing news-based features with 0.
    # This allows the model table to remain complete even in months with no articles.
    news_feature_cols = [col for col in df.columns if col not in ["month", "ppi_value"]]
    df[news_feature_cols] = df[news_feature_cols].fillna(0)

    # Sort by time to prepare lagged and rolling features.
    df = df.sort_values("month").reset_index(drop=True)

    # Create core PPI-based time-series features.
    df["ppi_pct_change"] = df["ppi_value"].pct_change() * 100
    df["ppi_lag_1"] = df["ppi_value"].shift(1)
    df["ppi_pct_change_lag_1"] = df["ppi_pct_change"].shift(1)
    df["ppi_ma_3"] = df["ppi_value"].rolling(window=3).mean().shift(1)
    df["ppi_std_3"] = df["ppi_value"].rolling(window=3).std().shift(1)

    # Define the supervised learning target as the next month's percentage change.
    # This means the model will use information available at month t
    # to predict the class of month t+1.
    df["target_pct_change_next_month"] = df["ppi_pct_change"].shift(-1)

    # Use quantile-based cuts so the target classes become more balanced.
    usable_target = df["target_pct_change_next_month"].dropna()
    low_q = float(config["target"]["low_quantile"])
    high_q = float(config["target"]["high_quantile"])

    low_cut = usable_target.quantile(low_q)
    high_cut = usable_target.quantile(high_q)

    logger.info(
        "Target quantile cuts calculated: low_cut=%.4f, high_cut=%.4f",
        low_cut,
        high_cut,
    )

    # Convert the numeric next-month change into a 3-class target label.
    df["target_class"] = df["target_pct_change_next_month"].apply(
        lambda x: _assign_target_class(x, low_cut, high_cut)
    )

    # Remove rows that cannot be used for training because lagged features
    # or target labels are missing.
    df = df.dropna(
        subset=[
            "ppi_lag_1",
            "ppi_pct_change_lag_1",
            "ppi_ma_3",
            "ppi_std_3",
            "target_class",
        ]
    )

    # Save the final feature table as a processed artifact.
    output_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Saved feature table to %s (%s rows)", output_path, len(df))
    logger.info("Target class distribution:\n%s", df["target_class"].value_counts(dropna=False).to_string())

    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual feature engineering tests.
    build_feature_table()