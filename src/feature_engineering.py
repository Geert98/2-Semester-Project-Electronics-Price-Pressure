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
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

TEXT_FEATURE_CHAR_LIMIT = 5000

RELEVANCE_TERMS = [
    "semiconductor",
    "semiconductors",
    "microchip",
    "microchips",
    "chip",
    "chips",
    "dram",
    "nand",
    "ssd",
    "memory",
    "electronics",
]

UPWARD_PRESSURE_TERMS = [
    "shortage",
    "shortages",
    "supply disruption",
    "supply chain pressure",
    "tariff",
    "tariffs",
    "export control",
    "export controls",
    "sanction",
    "sanctions",
    "chip ban",
    "production halt",
    "capacity constraint",
    "lead time",
    "price increase",
    "rising prices",
    "higher prices",
    "demand surge",
    "ai demand",
    "inventory drawdown",
]

DOWNWARD_PRESSURE_TERMS = [
    "oversupply",
    "glut",
    "inventory surplus",
    "weak demand",
    "price cut",
    "price cuts",
    "falling prices",
    "lower prices",
    "downturn",
    "slowdown",
    "excess inventory",
]

TOPIC_TERMS = {
    "shortage": ["shortage", "shortages", "capacity constraint", "lead time"],
    "tariff": ["tariff", "tariffs", "duties"],
    "demand": ["demand", "ai demand", "demand surge", "data center"],
    "supply_chain": ["supply chain", "logistics", "shipping", "disruption"],
    "export_controls": ["export control", "export controls", "sanction", "sanctions", "chip ban"],
    "oversupply": ["oversupply", "glut", "excess inventory", "inventory surplus"],
}


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


def _count_terms(text: str, terms: list[str]) -> int:
    """
    Count how many configured terms appear in a cleaned article text.
    """
    if not isinstance(text, str) or not text:
        return 0
    normalized = text.lower()
    return sum(1 for term in terms if term in normalized)


def _bounded_relevance_score(text: str) -> float:
    """
    Estimate whether an article is actually about electronics/semiconductors.
    """
    return min(_count_terms(text, RELEVANCE_TERMS) / 2, 1.0)


def _pressure_direction_score(text: str) -> int:
    """
    Positive values suggest upward price pressure; negative values suggest relief.
    """
    return _count_terms(text, UPWARD_PRESSURE_TERMS) - _count_terms(text, DOWNWARD_PRESSURE_TERMS)


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


def _load_fred_indicators(config: dict) -> pd.DataFrame:
    """
    Load optional monthly FRED indicator features from SQLite.
    """
    fred_path = get_sqlite_path(config)
    indicators_df = load_dataframe_from_sqlite(fred_path, "fred_indicators")
    if indicators_df.empty:
        return pd.DataFrame()

    indicators_df["month"] = pd.to_datetime(indicators_df["month"], errors="coerce")
    indicators_df = indicators_df.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    indicator_cols = [col for col in indicators_df.columns if col != "month"]
    for col in indicator_cols:
        indicators_df[col] = pd.to_numeric(indicators_df[col], errors="coerce")

    return indicators_df


def _add_indicator_features(df: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    """
    Add simple change and lag features for external monthly indicators.
    """
    for col in indicator_cols:
        df[f"{col}_pct_change"] = df[col].pct_change() * 100
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_pct_change_lag_1"] = df[f"{col}_pct_change"].shift(1)
    return df


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
    # Load local environment variables before resolving MongoDB settings.
    setup_env()

    # Load configuration and initialize the sentiment analyzer.
    config = load_config(config_path)
    analyzer = SentimentIntensityAnalyzer()

    # Load the cleaned news dataset and the FRED table from persistent storage.
    news_collection = config["storage"]["mongo"]["test_clean_news_collection"]
    enriched_collection = config["storage"]["mongo"].get("test_enriched_news_collection")
    news_df = load_dataframe_from_mongo(config, news_collection, sort_by="published_at")
    enriched_df = (
        load_dataframe_from_mongo(config, enriched_collection, sort_by="published_at")
        if enriched_collection
        else pd.DataFrame()
    )

    fred_path = get_sqlite_path(config)
    fred_df = load_dataframe_from_sqlite(fred_path, "fred_series")
    indicators_df = _load_fred_indicators(config)

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
        news_df["full_text_len"] = news_df["clean_text"].fillna("").str.len()
        news_df["signal_text"] = news_df["clean_text"].fillna("").str.slice(0, TEXT_FEATURE_CHAR_LIMIT)
        news_df["sentiment"] = news_df["signal_text"].apply(lambda x: _compute_sentiment(x, analyzer))
        news_df["is_negative"] = (news_df["sentiment"] < 0).astype(int)
        news_df["text_len"] = news_df["signal_text"].str.len()
        news_df["has_article_body"] = (
            news_df.get("content", pd.Series("", index=news_df.index)).fillna("").str.strip() != ""
        ).astype(int)
        news_df["relevance_score"] = news_df["signal_text"].apply(_bounded_relevance_score)
        news_df["pressure_direction_score"] = news_df["signal_text"].apply(_pressure_direction_score)
        news_df["pressure_strength"] = news_df["pressure_direction_score"].abs().clip(upper=5)
        news_df["upward_pressure_article"] = (news_df["pressure_direction_score"] > 0).astype(int)
        news_df["downward_pressure_article"] = (news_df["pressure_direction_score"] < 0).astype(int)

        for topic, terms in TOPIC_TERMS.items():
            news_df[f"topic_{topic}"] = news_df["signal_text"].apply(
                lambda x, topic_terms=terms: int(_count_terms(x, topic_terms) > 0)
            )

        # Create one binary keyword column per configured keyword.
        keywords = config["features"]["keywords"]
        for kw in keywords:
            col_name = f"kw_{kw.replace(' ', '_')}"
            news_df[col_name] = news_df["signal_text"].apply(
                lambda x, keyword=kw: _contains_keyword(x, keyword)
            )

        # Define the monthly aggregation logic.
        # The goal is to summarize all article-level signals into one row per month.
        agg_dict = {
            "title": "count",
            "sentiment": "mean",
            "is_negative": "mean",
            "title_len": "mean",
            "full_text_len": "mean",
            "text_len": "mean",
            "has_article_body": "mean",
            "relevance_score": "mean",
            "pressure_direction_score": "mean",
            "pressure_strength": "mean",
            "upward_pressure_article": "mean",
            "downward_pressure_article": "mean",
            "source": pd.Series.nunique,
        }

        for topic in TOPIC_TERMS:
            agg_dict[f"topic_{topic}"] = "sum"

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
                    "full_text_len": "avg_full_text_len",
                    "text_len": "avg_text_len",
                    "has_article_body": "article_body_share",
                    "relevance_score": "avg_relevance_score",
                    "pressure_direction_score": "avg_pressure_direction_score",
                    "pressure_strength": "avg_pressure_strength",
                    "upward_pressure_article": "upward_pressure_share",
                    "downward_pressure_article": "downward_pressure_share",
                    "source": "unique_sources",
                }
            )
        )

        if not enriched_df.empty:
            enriched_df["published_at"] = pd.to_datetime(enriched_df["published_at"], errors="coerce")
            enriched_df["month"] = pd.to_datetime(enriched_df["month"], errors="coerce")
            enriched_df = enriched_df.dropna(subset=["month"])

            enriched_df["is_relevant"] = enriched_df["is_relevant"].astype(int)
            enriched_df["is_negative"] = enriched_df["is_negative"].astype(int)
            enriched_df["ai_upward_pressure_article"] = (
                enriched_df["price_pressure_direction"] == "upward"
            ).astype(int)
            enriched_df["ai_downward_pressure_article"] = (
                enriched_df["price_pressure_direction"] == "downward"
            ).astype(int)
            enriched_df["ai_neutral_pressure_article"] = (
                enriched_df["price_pressure_direction"] == "neutral"
            ).astype(int)

            ai_numeric_cols = [
                "relevance_score",
                "sentiment",
                "is_negative",
                "is_relevant",
                "content_char_count",
                "scored_char_count",
                "price_pressure_direction_score",
                "price_pressure_strength",
                "ai_upward_pressure_article",
                "ai_downward_pressure_article",
                "ai_neutral_pressure_article",
            ]
            for col in ai_numeric_cols:
                enriched_df[col] = pd.to_numeric(enriched_df[col], errors="coerce").fillna(0)

            ai_agg_dict = {
                "url": "count",
                "relevance_score": "mean",
                "sentiment": "mean",
                "is_negative": "mean",
                "is_relevant": ["mean", "sum"],
                "content_char_count": "mean",
                "scored_char_count": "mean",
                "price_pressure_direction_score": "mean",
                "price_pressure_strength": "mean",
                "ai_upward_pressure_article": "mean",
                "ai_downward_pressure_article": "mean",
                "ai_neutral_pressure_article": "mean",
            }

            for topic in TOPIC_TERMS:
                flag_col = f"topic_{topic}_flag"
                count_col = f"topic_{topic}_count"
                if flag_col in enriched_df.columns:
                    enriched_df[flag_col] = pd.to_numeric(enriched_df[flag_col], errors="coerce").fillna(0)
                    ai_agg_dict[flag_col] = "sum"
                if count_col in enriched_df.columns:
                    enriched_df[count_col] = pd.to_numeric(enriched_df[count_col], errors="coerce").fillna(0)
                    ai_agg_dict[count_col] = "sum"

            monthly_ai = enriched_df.groupby("month").agg(ai_agg_dict)
            monthly_ai.columns = [
                "_".join(col).strip("_") if isinstance(col, tuple) else col
                for col in monthly_ai.columns.to_flat_index()
            ]
            ai_rename_map = {
                "url_count": "ai_enriched_article_count",
                "relevance_score_mean": "ai_avg_relevance_score",
                "sentiment_mean": "ai_avg_sentiment",
                "is_negative_mean": "ai_negative_share",
                "is_relevant_mean": "ai_relevant_share",
                "is_relevant_sum": "ai_relevant_article_count",
                "content_char_count_mean": "ai_avg_content_char_count",
                "scored_char_count_mean": "ai_avg_scored_char_count",
                "price_pressure_direction_score_mean": "ai_avg_pressure_direction_score",
                "price_pressure_strength_mean": "ai_avg_pressure_strength",
                "ai_upward_pressure_article_mean": "ai_upward_pressure_share",
                "ai_downward_pressure_article_mean": "ai_downward_pressure_share",
                "ai_neutral_pressure_article_mean": "ai_neutral_pressure_share",
            }
            for topic in TOPIC_TERMS:
                ai_rename_map[f"topic_{topic}_flag_sum"] = f"ai_topic_{topic}_article_count"
                ai_rename_map[f"topic_{topic}_count_sum"] = f"ai_topic_{topic}_term_count"

            monthly_ai = monthly_ai.reset_index().rename(columns=ai_rename_map)
            monthly_news = monthly_news.merge(monthly_ai, on="month", how="left")

    # Create the monthly PPI table.
    monthly_ppi = (
        fred_df[["month", "ppi_value"]]
        .drop_duplicates()
        .sort_values("month")
        .reset_index(drop=True)
    )

    # Merge the structured PPI features with the aggregated monthly news features.
    df = monthly_ppi.merge(monthly_news, on="month", how="left")

    indicator_cols: list[str] = []
    if not indicators_df.empty:
        indicator_cols = [col for col in indicators_df.columns if col != "month"]
        df = df.merge(indicators_df, on="month", how="left")
        df[indicator_cols] = df[indicator_cols].ffill()

    # Replace missing news-based features with 0.
    # This allows the model table to remain complete even in months with no articles.
    news_feature_cols = [
        col
        for col in df.columns
        if col not in ["month", "ppi_value", *indicator_cols]
    ]
    df[news_feature_cols] = df[news_feature_cols].fillna(0)

    # Sort by time to prepare lagged and rolling features.
    df = df.sort_values("month").reset_index(drop=True)

    # Create core PPI-based time-series features.
    df["ppi_pct_change"] = df["ppi_value"].pct_change() * 100
    df["ppi_lag_1"] = df["ppi_value"].shift(1)
    df["ppi_pct_change_lag_1"] = df["ppi_pct_change"].shift(1)
    df["ppi_ma_3"] = df["ppi_value"].rolling(window=3).mean().shift(1)
    df["ppi_std_3"] = df["ppi_value"].rolling(window=3).std().shift(1)
    df = _add_indicator_features(df, indicator_cols)

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
            *indicator_cols,
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
