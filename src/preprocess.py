from __future__ import annotations

# This script is responsible for preprocessing the raw news data.
#
# In the project pipeline, this file sits between raw news ingestion
# and feature engineering.
#
# The script:
# 1. loads the raw article-level CSV created during ingestion
# 2. cleans and standardizes text fields
# 3. parses article timestamps
# 4. removes duplicates and unusable rows
# 5. creates a month column for later aggregation
# 6. saves a cleaned article-level dataset
#
# The output of this script is the processed news dataset that is later used
# to compute sentiment features, keyword counts, article volumes, and source counts.

import logging

import pandas as pd

from src.storage import load_dataframe_from_mongo, save_dataframe_to_mongo
from src.text_cleaning import clean_news_text
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)


def preprocess_news(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Clean the raw news dataset and save a processed article-level CSV.

    Why this function exists:
    - Raw API output is often inconsistent and contains missing values,
      duplicate articles, and mixed formatting.
    - The downstream feature engineering step needs a stable and predictable schema.
    - This function standardizes the article data before aggregation.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Cleaned article-level DataFrame.
    """
    # Load shared project configuration to find storage locations.
    config = load_config(config_path)

    raw_collection = config["storage"]["mongo"]["test_news_collection"]
    clean_collection = config["storage"]["mongo"]["test_clean_news_collection"]

    df = load_dataframe_from_mongo(config, raw_collection, sort_by="seen_date")
    if df.empty:
        logger.warning("Raw news collection is empty: %s", raw_collection)
        empty_output = pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "title",
                "url",
                "source",
                "language",
                "seen_date",
                "content",
                "social_image",
                "source_country",
                "published_at",
                "clean_text",
                "month",
                "title_len",
            ]
        )
        save_dataframe_to_mongo(empty_output, config, clean_collection)
        return empty_output

    # Fill selected columns with empty strings to avoid errors in later text operations.
    df["title"] = df["title"].fillna("")
    df["source"] = df["source"].fillna("")
    df["language"] = df["language"].fillna("")
    if "content" not in df.columns:
        df["content"] = ""
    else:
        df["content"] = df["content"].fillna("")

    # Parse the article timestamp.
    #
    # GDELT's "seen_date" field often uses a format like:
    # YYYYMMDDTHHMMSSZ
    #
    # errors="coerce" converts invalid values to NaT instead of crashing.
    df["published_at"] = pd.to_datetime(df["seen_date"], errors="coerce", utc=True)

    # Remove timezone information so downstream handling is simpler.
    df["published_at"] = df["published_at"].dt.tz_convert(None)

    # Remove rows where the timestamp could not be parsed.
    df = df.dropna(subset=["published_at"])

    # Remove duplicate articles based on title, URL, and timestamp.
    # This helps reduce repeated items returned by the news API.
    df = df.drop_duplicates(subset=["title", "url", "published_at"])

    # Keep only English-language articles where possible.
    # Some rows may have an empty language field, so those are also kept.
    df = df[
        df["language"].isin(["English", "english", "EN", "en"])
        | (df["language"] == "")
    ]

    # Create the cleaned text field used later for keyword counts and sentiment analysis.
    # Prefer the article body when it is available, and fall back to the title.
    # The shared cleaner strips Guardian HTML while keeping visible text and link text.
    text_source = df["content"].where(df["content"].str.strip() != "", df["title"])
    df["clean_text"] = text_source.apply(clean_news_text)

    # Create a monthly timestamp key for later aggregation in feature engineering.
    df["month"] = df["published_at"].dt.to_period("M").dt.to_timestamp()

    # Store the article title length as a simple descriptive text feature.
    df["title_len"] = df["title"].str.len()

    # Sort chronologically so the output file is consistent and easier to inspect.
    df = df.sort_values("published_at").reset_index(drop=True)

    save_dataframe_to_mongo(df, config, clean_collection)

    logger.info("Saved cleaned news to MongoDB collection %s (%s rows)", clean_collection, len(df))
    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual preprocessing tests.
    setup_env()
    preprocess_news()
