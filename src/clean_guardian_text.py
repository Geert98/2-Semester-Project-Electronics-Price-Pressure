from __future__ import annotations

import logging

import pandas as pd

from src.storage import load_dataframe_from_mongo, save_dataframe_to_mongo
from src.text_cleaning import clean_news_text
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)


def clean_guardian_text(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    collection_name = config["storage"]["mongo"]["test_clean_news_collection"]

    df = load_dataframe_from_mongo(config, collection_name, sort_by="published_at")
    if df.empty:
        logger.warning("Clean news collection is empty: %s", collection_name)
        return df

    if "content" not in df.columns:
        df["content"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    df["content"] = df["content"].fillna("")
    df["title"] = df["title"].fillna("")

    text_source = df["content"].where(df["content"].str.strip() != "", df["title"])
    df["clean_text"] = text_source.apply(clean_news_text)

    save_dataframe_to_mongo(df, config, collection_name)
    logger.info("Updated clean_text in MongoDB collection %s (%s rows)", collection_name, len(df))
    return df


if __name__ == "__main__":
    setup_env()
    logging.basicConfig(level=logging.INFO)
    clean_guardian_text()