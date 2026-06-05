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

AI_TIME_FEATURE_COLUMNS = [
    "ai_no_articles",
    "ai_articles_but_none_relevant",
    "ai_relevant_articles_available",
    "ai_enriched_article_count",
    "ai_relevant_article_count",
    "ai_relevant_share",
    "ai_avg_relevance_score",
    "ai_avg_sentiment",
    "ai_relevant_avg_sentiment",
    "ai_relevance_weighted_sentiment",
    "ai_avg_pressure_direction_score",
    "ai_relevant_avg_pressure_direction_score",
    "ai_relevance_weighted_pressure_direction_score",
    "ai_avg_pressure_strength",
    "ai_relevant_avg_pressure_strength",
    "ai_relevance_weighted_pressure_strength",
    "ai_upward_pressure_share",
    "ai_relevant_upward_pressure_share",
    "ai_downward_pressure_share",
    "ai_relevant_downward_pressure_share",
]


def _compute_sentiment(text: str, analyzer: object) -> float:
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
    indicator_features = {}
    for col in indicator_cols:
        pct_change = df[col].pct_change() * 100
        indicator_features[f"{col}_pct_change"] = pct_change
        indicator_features[f"{col}_lag_1"] = df[col].shift(1)
        indicator_features[f"{col}_pct_change_lag_1"] = pct_change.shift(1)

    if not indicator_features:
        return df
    return pd.concat([df, pd.DataFrame(indicator_features, index=df.index)], axis=1)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide two aligned Series and return 0 when the denominator is missing or 0.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def _add_ai_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit coverage flags for months with missing or irrelevant AI news.
    """
    if "ai_enriched_article_count" not in df.columns:
        df["ai_enriched_article_count"] = 0
    if "ai_relevant_article_count" not in df.columns:
        df["ai_relevant_article_count"] = 0

    enriched_count = pd.to_numeric(
        df["ai_enriched_article_count"],
        errors="coerce",
    ).fillna(0)
    relevant_count = pd.to_numeric(
        df["ai_relevant_article_count"],
        errors="coerce",
    ).fillna(0)

    coverage_features = pd.DataFrame(
        {
            "ai_no_articles": (enriched_count <= 0).astype(int),
            "ai_articles_but_none_relevant": (
                (enriched_count > 0) & (relevant_count <= 0)
            ).astype(int),
            "ai_relevant_articles_available": (relevant_count > 0).astype(int),
        },
        index=df.index,
    )
    return pd.concat([df, coverage_features], axis=1)


def _add_ai_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling versions of the AI article signals.
    """
    ai_cols = [col for col in AI_TIME_FEATURE_COLUMNS if col in df.columns]
    time_features = {}
    for col in ai_cols:
        for lag in (1, 2, 3):
            time_features[f"{col}_lag_{lag}"] = df[col].shift(lag)
        time_features[f"{col}_roll_3_mean"] = df[col].rolling(
            window=3,
            min_periods=1,
        ).mean()
        time_features[f"{col}_roll_3_std"] = df[col].rolling(
            window=3,
            min_periods=2,
        ).std()

    if time_features:
        df = pd.concat([df, pd.DataFrame(time_features, index=df.index)], axis=1)

    ai_time_cols = [
        col
        for col in df.columns
        if col.startswith("ai_") and ("_lag_" in col or "_roll_3_" in col)
    ]
    df[ai_time_cols] = df[ai_time_cols].fillna(0)
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

    # Load configuration and initialize optional legacy VADER sentiment.
    config = load_config(config_path)
    features_config = config.get("features", {})
    use_vader_sentiment = bool(features_config.get("use_vader_sentiment", False))
    analyzer = None
    if use_vader_sentiment:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()

    use_legacy_text_features = bool(features_config.get("use_legacy_text_features", False))

    # Load the cleaned news dataset and the FRED table from persistent storage.
    news_collection = config["storage"]["mongo"]["test_clean_news_collection"]
    enriched_collection = config["storage"]["mongo"].get("test_enriched_news_collection")
    news_projection = None
    if not use_legacy_text_features:
        news_projection = {
            "published_at": 1,
            "month": 1,
            "title": 1,
            "title_len": 1,
            "source": 1,
            "provider": 1,
            "content_char_count": 1,
            "linked_content_char_count": 1,
            "hn_points": 1,
            "hn_num_comments": 1,
        }
    news_df = load_dataframe_from_mongo(
        config,
        news_collection,
        sort_by="published_at",
        projection=news_projection,
    )
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

        # Compute optional legacy VADER sentiment. The AI-enriched sentiment lives
        # in ai_avg_sentiment and is the primary sentiment signal for modeling.
        clean_text = news_df.get("clean_text", pd.Series("", index=news_df.index)).fillna("")
        content_char_count = pd.to_numeric(
            news_df.get("content_char_count", pd.Series(0, index=news_df.index)),
            errors="coerce",
        ).fillna(0)
        linked_content_char_count = pd.to_numeric(
            news_df.get("linked_content_char_count", pd.Series(0, index=news_df.index)),
            errors="coerce",
        ).fillna(0)

        if use_legacy_text_features:
            news_df["full_text_len"] = clean_text.str.len()
            news_df["signal_text"] = clean_text.str.slice(0, TEXT_FEATURE_CHAR_LIMIT)
        else:
            news_df["full_text_len"] = content_char_count.where(
                content_char_count > 0,
                linked_content_char_count,
            )
            news_df["signal_text"] = ""

        if analyzer is not None and use_legacy_text_features:
            news_df["sentiment"] = news_df["signal_text"].apply(lambda x: _compute_sentiment(x, analyzer))
        else:
            news_df["sentiment"] = 0.0
        news_df["is_negative"] = (news_df["sentiment"] < 0).astype(int)
        news_df["text_len"] = news_df["signal_text"].str.len()
        body_text = news_df.get("content", pd.Series("", index=news_df.index)).fillna("")
        linked_body_text = news_df.get("linked_content", pd.Series("", index=news_df.index)).fillna("")
        news_df["has_article_body"] = (
            (body_text.str.strip() != "")
            | (content_char_count > 0)
            | (
                (news_df.get("provider", pd.Series("", index=news_df.index)).fillna("") == "hackernews")
                & ((linked_body_text.str.strip() != "") | (linked_content_char_count > 0))
            )
        ).astype(int)
        news_df["is_hackernews"] = (
            news_df.get("provider", pd.Series("", index=news_df.index)).fillna("") == "hackernews"
        ).astype(int)
        news_df["hn_points"] = pd.to_numeric(
            news_df.get("hn_points", pd.Series(0, index=news_df.index)),
            errors="coerce",
        ).fillna(0)
        news_df["hn_num_comments"] = pd.to_numeric(
            news_df.get("hn_num_comments", pd.Series(0, index=news_df.index)),
            errors="coerce",
        ).fillna(0)
        hn_cfg = config["news"].get("hackernews", {})
        hn_high_attention_min_points = int(hn_cfg.get("high_attention_min_points", 100))
        hn_high_attention_min_comments = int(hn_cfg.get("high_attention_min_comments", 50))
        news_df["hn_points_signal"] = news_df["hn_points"] * news_df["is_hackernews"]
        news_df["hn_comments_signal"] = news_df["hn_num_comments"] * news_df["is_hackernews"]
        news_df["hn_high_attention_story"] = (
            (news_df["is_hackernews"] == 1)
            & (
                (news_df["hn_points"] >= hn_high_attention_min_points)
                | (news_df["hn_num_comments"] >= hn_high_attention_min_comments)
            )
        ).astype(int)
        if use_legacy_text_features:
            news_df["relevance_score"] = news_df["signal_text"].apply(_bounded_relevance_score)
            news_df["pressure_direction_score"] = news_df["signal_text"].apply(_pressure_direction_score)
        else:
            news_df["relevance_score"] = 0.0
            news_df["pressure_direction_score"] = 0
        news_df["pressure_strength"] = news_df["pressure_direction_score"].abs().clip(upper=5)
        news_df["upward_pressure_article"] = (news_df["pressure_direction_score"] > 0).astype(int)
        news_df["downward_pressure_article"] = (news_df["pressure_direction_score"] < 0).astype(int)

        for topic, terms in TOPIC_TERMS.items():
            if use_legacy_text_features:
                news_df[f"topic_{topic}"] = news_df["signal_text"].apply(
                    lambda x, topic_terms=terms: int(_count_terms(x, topic_terms) > 0)
                )
            else:
                news_df[f"topic_{topic}"] = 0

        # Create one binary keyword column per configured keyword.
        keywords = config["features"]["keywords"]
        for kw in keywords:
            col_name = f"kw_{kw.replace(' ', '_')}"
            if use_legacy_text_features:
                news_df[col_name] = news_df["signal_text"].apply(
                    lambda x, keyword=kw: _contains_keyword(x, keyword)
                )
            else:
                news_df[col_name] = 0

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
            "is_hackernews": "sum",
            "hn_points_signal": "sum",
            "hn_comments_signal": "sum",
            "hn_high_attention_story": "sum",
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
                    "is_hackernews": "hn_story_count",
                    "hn_points_signal": "hn_total_points",
                    "hn_comments_signal": "hn_total_comments",
                    "hn_high_attention_story": "hn_high_attention_story_count",
                    "relevance_score": "avg_relevance_score",
                    "pressure_direction_score": "avg_pressure_direction_score",
                    "pressure_strength": "avg_pressure_strength",
                    "upward_pressure_article": "upward_pressure_share",
                    "downward_pressure_article": "downward_pressure_share",
                    "source": "unique_sources",
                }
            )
        )
        monthly_news["hn_avg_points"] = (
            monthly_news["hn_total_points"] / monthly_news["hn_story_count"].replace(0, np.nan)
        ).fillna(0)
        monthly_news["hn_avg_comments"] = (
            monthly_news["hn_total_comments"] / monthly_news["hn_story_count"].replace(0, np.nan)
        ).fillna(0)

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

            relevant_mask = enriched_df["is_relevant"] == 1
            enriched_df["ai_relevance_weight"] = enriched_df["relevance_score"].clip(lower=0, upper=1)
            enriched_df["ai_high_relevance_article"] = (
                enriched_df["relevance_score"] >= 0.7
            ).astype(int)
            enriched_df["ai_relevant_sentiment_sum"] = enriched_df["sentiment"].where(relevant_mask, 0)
            enriched_df["ai_relevant_pressure_direction_score_sum"] = enriched_df[
                "price_pressure_direction_score"
            ].where(relevant_mask, 0)
            enriched_df["ai_relevant_pressure_strength_sum"] = enriched_df[
                "price_pressure_strength"
            ].where(relevant_mask, 0)
            enriched_df["ai_sentiment_relevance_weighted_sum"] = (
                enriched_df["sentiment"] * enriched_df["ai_relevance_weight"]
            )
            enriched_df["ai_pressure_direction_relevance_weighted_sum"] = (
                enriched_df["price_pressure_direction_score"] * enriched_df["ai_relevance_weight"]
            )
            enriched_df["ai_pressure_strength_relevance_weighted_sum"] = (
                enriched_df["price_pressure_strength"] * enriched_df["ai_relevance_weight"]
            )
            enriched_df["ai_relevant_upward_pressure_article"] = (
                relevant_mask & (enriched_df["price_pressure_direction"] == "upward")
            ).astype(int)
            enriched_df["ai_relevant_downward_pressure_article"] = (
                relevant_mask & (enriched_df["price_pressure_direction"] == "downward")
            ).astype(int)
            enriched_df["ai_relevant_neutral_pressure_article"] = (
                relevant_mask & (enriched_df["price_pressure_direction"] == "neutral")
            ).astype(int)

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
                "ai_relevance_weight": "sum",
                "ai_high_relevance_article": "sum",
                "ai_relevant_sentiment_sum": "sum",
                "ai_relevant_pressure_direction_score_sum": "sum",
                "ai_relevant_pressure_strength_sum": "sum",
                "ai_sentiment_relevance_weighted_sum": "sum",
                "ai_pressure_direction_relevance_weighted_sum": "sum",
                "ai_pressure_strength_relevance_weighted_sum": "sum",
                "ai_relevant_upward_pressure_article": "sum",
                "ai_relevant_downward_pressure_article": "sum",
                "ai_relevant_neutral_pressure_article": "sum",
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
                "ai_relevance_weight_sum": "ai_total_relevance_weight",
                "ai_high_relevance_article_sum": "ai_high_relevance_article_count",
                "ai_relevant_upward_pressure_article_sum": "ai_relevant_upward_pressure_article_count",
                "ai_relevant_downward_pressure_article_sum": "ai_relevant_downward_pressure_article_count",
                "ai_relevant_neutral_pressure_article_sum": "ai_relevant_neutral_pressure_article_count",
            }
            for topic in TOPIC_TERMS:
                ai_rename_map[f"topic_{topic}_flag_sum"] = f"ai_topic_{topic}_article_count"
                ai_rename_map[f"topic_{topic}_count_sum"] = f"ai_topic_{topic}_term_count"

            monthly_ai = monthly_ai.reset_index().rename(columns=ai_rename_map)
            relevant_count = monthly_ai["ai_relevant_article_count"]
            relevance_weight = monthly_ai["ai_total_relevance_weight"]
            article_count = monthly_ai["ai_enriched_article_count"]

            monthly_ai["ai_high_relevance_share"] = _safe_divide(
                monthly_ai["ai_high_relevance_article_count"],
                article_count,
            )
            monthly_ai["ai_relevant_avg_sentiment"] = _safe_divide(
                monthly_ai["ai_relevant_sentiment_sum_sum"],
                relevant_count,
            )
            monthly_ai["ai_relevant_avg_pressure_direction_score"] = _safe_divide(
                monthly_ai["ai_relevant_pressure_direction_score_sum_sum"],
                relevant_count,
            )
            monthly_ai["ai_relevant_avg_pressure_strength"] = _safe_divide(
                monthly_ai["ai_relevant_pressure_strength_sum_sum"],
                relevant_count,
            )
            monthly_ai["ai_relevance_weighted_sentiment"] = _safe_divide(
                monthly_ai["ai_sentiment_relevance_weighted_sum_sum"],
                relevance_weight,
            )
            monthly_ai["ai_relevance_weighted_pressure_direction_score"] = _safe_divide(
                monthly_ai["ai_pressure_direction_relevance_weighted_sum_sum"],
                relevance_weight,
            )
            monthly_ai["ai_relevance_weighted_pressure_strength"] = _safe_divide(
                monthly_ai["ai_pressure_strength_relevance_weighted_sum_sum"],
                relevance_weight,
            )
            monthly_ai["ai_relevant_upward_pressure_share"] = _safe_divide(
                monthly_ai["ai_relevant_upward_pressure_article_count"],
                relevant_count,
            )
            monthly_ai["ai_relevant_downward_pressure_share"] = _safe_divide(
                monthly_ai["ai_relevant_downward_pressure_article_count"],
                relevant_count,
            )
            monthly_ai["ai_relevant_neutral_pressure_share"] = _safe_divide(
                monthly_ai["ai_relevant_neutral_pressure_article_count"],
                relevant_count,
            )
            monthly_ai = monthly_ai.drop(
                columns=[
                    "ai_relevant_sentiment_sum_sum",
                    "ai_relevant_pressure_direction_score_sum_sum",
                    "ai_relevant_pressure_strength_sum_sum",
                    "ai_sentiment_relevance_weighted_sum_sum",
                    "ai_pressure_direction_relevance_weighted_sum_sum",
                    "ai_pressure_strength_relevance_weighted_sum_sum",
                ],
                errors="ignore",
            )
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

    # Add explicit article coverage flags before time features are generated.
    df = _add_ai_coverage_features(df)

    # Add lagged and rolling AI features so delayed market reactions are learnable.
    df = _add_ai_time_features(df)

    # Create core PPI-based time-series features.
    ppi_pct_change = df["ppi_value"].pct_change() * 100
    ppi_features = pd.DataFrame(
        {
            "ppi_pct_change": ppi_pct_change,
            "ppi_lag_1": df["ppi_value"].shift(1),
            "ppi_pct_change_lag_1": ppi_pct_change.shift(1),
            "ppi_ma_3": df["ppi_value"].rolling(window=3).mean().shift(1),
            "ppi_std_3": df["ppi_value"].rolling(window=3).std().shift(1),
        },
        index=df.index,
    )
    df = pd.concat([df, ppi_features], axis=1)
    df = _add_indicator_features(df, indicator_cols)

    # Define the supervised learning target as the next month's percentage change.
    # This means the model will use information available at month t
    # to predict the class of month t+1.
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "target_pct_change_next_month": df["ppi_pct_change"].shift(-1),
                },
                index=df.index,
            ),
        ],
        axis=1,
    )

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
    target_class = df["target_pct_change_next_month"].apply(
        lambda x: _assign_target_class(x, low_cut, high_cut),
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame({"target_class": target_class}, index=df.index),
        ],
        axis=1,
    )

    target_class_lag_1 = df["target_class"].shift(1)
    target_class_lag_features = pd.DataFrame(
        {
            "target_class_lag_1": target_class_lag_1,
            "target_class_lag_1_score": target_class_lag_1.map(
                {"low": -1, "medium": 0, "high": 1}
            ),
            "target_class_lag_1_is_low": (target_class_lag_1 == "low").astype(int),
            "target_class_lag_1_is_medium": (target_class_lag_1 == "medium").astype(int),
            "target_class_lag_1_is_high": (target_class_lag_1 == "high").astype(int),
        },
        index=df.index,
    )
    df = pd.concat([df, target_class_lag_features], axis=1)

    # Save a scoring table before removing rows with unknown future targets.
    # This table is used for latest prediction, where the next month's target
    # is usually not available yet.
    prediction_df = df.dropna(
        subset=[
            "ppi_lag_1",
            "ppi_pct_change_lag_1",
            "ppi_ma_3",
            "ppi_std_3",
            "target_class_lag_1_score",
            *indicator_cols,
        ]
    )
    prediction_output_path = Path(config["paths"]["processed_dir"]) / "prediction_table.csv"
    prediction_df.to_csv(prediction_output_path, index=False)

    # Remove rows that cannot be used for training because lagged features
    # or target labels are missing.
    df = df.dropna(
        subset=[
            "ppi_lag_1",
            "ppi_pct_change_lag_1",
            "ppi_ma_3",
            "ppi_std_3",
            "target_class_lag_1_score",
            "target_class",
            *indicator_cols,
        ]
    )

    # Save the final feature table as a processed artifact.
    output_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Saved feature table to %s (%s rows)", output_path, len(df))
    logger.info("Saved prediction table to %s (%s rows)", prediction_output_path, len(prediction_df))
    logger.info("Target class distribution:\n%s", df["target_class"].value_counts(dropna=False).to_string())

    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual feature engineering tests.
    build_feature_table()
