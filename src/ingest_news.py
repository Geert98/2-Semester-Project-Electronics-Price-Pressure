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
from argparse import ArgumentParser

import pandas as pd

from src.news_providers.common import month_windows
from src.news_providers.registry import build_provider_runners
from src.storage import upsert_dataframe_to_mongo
from src.utils import load_config

logger = logging.getLogger(__name__)


def _select_news_query(news_cfg: dict, query: str | None, query_mode: str | None) -> str:
    """
    Resolve the query used for news ingestion.

    The default config can define both a strict query and a broader backfill
    query. Runtime overrides make it possible to run historical backfills
    without editing the YAML file each time.
    """
    if query:
        return query.strip()

    mode = (query_mode or news_cfg.get("query_mode") or "").strip()
    if mode:
        mode_key = f"query_{mode}"
        if mode_key in news_cfg and str(news_cfg[mode_key]).strip():
            return str(news_cfg[mode_key]).strip()

    return str(news_cfg["query"]).strip()


def _news_config_with_overrides(
    news_cfg: dict,
    start_date: str | None,
    end_date: str | None,
    max_records_per_window: int | None,
    guardian_max_pages: int | None,
    guardian_only: bool,
) -> dict:
    """
    Copy the news config and apply safe runtime overrides.
    """
    resolved = dict(news_cfg)
    resolved["guardian"] = dict(news_cfg.get("guardian", {}))

    if start_date:
        resolved["start_date"] = start_date
    if end_date:
        resolved["end_date"] = end_date
    if max_records_per_window is not None:
        resolved["max_records_per_window"] = max_records_per_window
    if guardian_max_pages is not None:
        resolved["guardian"]["max_pages_per_window"] = guardian_max_pages

    if guardian_only:
        resolved["use_guardian"] = True
        resolved["use_gdelt"] = False
        resolved["use_newsapi"] = False
        resolved["use_newsdata"] = False
        resolved["use_webzio"] = False

    return resolved


def ingest_news(
    config_path: str = "configs/config.yaml",
    start_date: str | None = None,
    end_date: str | None = None,
    query: str | None = None,
    query_mode: str | None = None,
    max_records_per_window: int | None = None,
    guardian_max_pages: int | None = None,
    guardian_only: bool = False,
) -> pd.DataFrame:
    config = load_config(config_path)
    news_cfg = _news_config_with_overrides(
        news_cfg=config["news"],
        start_date=start_date,
        end_date=end_date,
        max_records_per_window=max_records_per_window,
        guardian_max_pages=guardian_max_pages,
        guardian_only=guardian_only,
    )

    query = _select_news_query(news_cfg, query=query, query_mode=query_mode)
    start_date = news_cfg["start_date"]
    end_date = news_cfg["end_date"]
    max_records = int(news_cfg["max_records_per_window"])
    sleep_seconds = float(news_cfg.get("sleep_seconds", 5))
    max_retries = int(news_cfg.get("max_retries", 5))
    retry_base_sleep = float(news_cfg.get("retry_base_sleep", 5))

    windows = month_windows(start_date, end_date)

    logger.info(
        "News ingestion configured for %s monthly windows from %s to %s",
        len(windows),
        start_date,
        end_date,
    )
    logger.info("Using news query: %s", query)

    provider_runners = build_provider_runners(
        news_cfg=news_cfg,
        query=query,
        max_records=max_records,
        windows=windows,
        max_retries=max_retries,
        retry_base_sleep=retry_base_sleep,
        sleep_seconds=sleep_seconds,
    )

    if not provider_runners:
        raise ValueError(
            "At least one news provider must be enabled: "
            "use_gdelt, use_guardian, use_newsapi, use_newsdata, or use_webzio."
        )

    rows: list[dict] = []

    for provider_name, run_provider in provider_runners:
        logger.info("Starting %s news provider", provider_name)
        run_provider(rows)

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
    parser = ArgumentParser(description="Ingest raw news articles into MongoDB.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start-date", help="Override news.start_date, e.g. 1998-01-01.")
    parser.add_argument("--end-date", help="Override news.end_date, e.g. 2005-12-31 or today.")
    parser.add_argument(
        "--query-mode",
        choices=["strict", "broad"],
        help="Use news.query_strict or news.query_broad from the config.",
    )
    parser.add_argument("--query", help="Use a custom API query instead of the configured query.")
    parser.add_argument(
        "--max-records-per-window",
        type=int,
        help="Override news.max_records_per_window.",
    )
    parser.add_argument(
        "--guardian-max-pages",
        type=int,
        help="Override news.guardian.max_pages_per_window.",
    )
    parser.add_argument(
        "--guardian-only",
        action="store_true",
        help="Use only Guardian for historical backfills.",
    )
    args = parser.parse_args()

    ingest_news(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
        query_mode=args.query_mode,
        max_records_per_window=args.max_records_per_window,
        guardian_max_pages=args.guardian_max_pages,
        guardian_only=args.guardian_only,
    )
