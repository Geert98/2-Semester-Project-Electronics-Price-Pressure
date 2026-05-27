from __future__ import annotations

# This script is responsible for ingesting news data from external news APIs.
#
# In this project, news articles are used as an external unstructured data source
# that complements the structured FRED time-series data.
#
# The script:
# 1. reads the news ingestion settings from the shared config file
# 2. creates monthly time windows for the requested period
# 3. queries the configured historical and recent news APIs
# 4. retries failed requests with exponential backoff
# 5. stores the raw article-level results in MongoDB

import logging

import pandas as pd

from src.news_providers.common import month_windows
from src.news_providers.gdelt import append_gdelt_rows
from src.news_providers.guardian import append_guardian_rows
from src.news_providers.newsapi import append_newsapi_rows
from src.news_providers.newsdata import append_newsdata_rows
from src.storage import upsert_dataframe_to_mongo
from src.utils import load_config

logger = logging.getLogger(__name__)


def ingest_news(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    news_cfg = config["news"]

    query = news_cfg["query"].strip()
    start_date = news_cfg["start_date"]
    end_date = news_cfg["end_date"]
    max_records = int(news_cfg["max_records_per_window"])
    sleep_seconds = float(news_cfg.get("sleep_seconds", 5))
    max_retries = int(news_cfg.get("max_retries", 5))
    retry_base_sleep = float(news_cfg.get("retry_base_sleep", 5))

    windows = month_windows(start_date, end_date)

    use_gdelt = bool(news_cfg.get("use_gdelt", True))
    use_guardian = bool(news_cfg.get("use_guardian", False))
    use_newsapi = bool(news_cfg.get("use_newsapi", False))
    use_newsdata = bool(news_cfg.get("use_newsdata", False))

    if not use_gdelt and not use_guardian and not use_newsapi and not use_newsdata:
        raise ValueError(
            "At least one news provider must be enabled: "
            "use_gdelt, use_guardian, use_newsapi, or use_newsdata."
        )

    rows: list[dict] = []

    if use_gdelt:
        append_gdelt_rows(
            rows=rows,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
            sleep_seconds=sleep_seconds,
        )

    if use_guardian:
        append_guardian_rows(
            rows=rows,
            news_cfg=news_cfg,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

    if use_newsapi:
        append_newsapi_rows(
            rows=rows,
            news_cfg=news_cfg,
            query=query,
            max_records=max_records,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

    if use_newsdata:
        append_newsdata_rows(
            rows=rows,
            news_cfg=news_cfg,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

    expected_columns = [
        "provider",
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
    ]

    df = pd.DataFrame(rows, columns=expected_columns)

    raw_collection = config["storage"]["mongo"]["test_news_collection"]
    changed_count = upsert_dataframe_to_mongo(df, config, raw_collection, key_columns=["url"])

    logger.info(
        "Upserted raw news to MongoDB collection %s (%s fetched rows, %s changed rows)",
        raw_collection,
        len(df),
        changed_count,
    )
    return df


if __name__ == "__main__":
    ingest_news()
