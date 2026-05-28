from __future__ import annotations

import logging
import time

import pandas as pd

from src.news_providers.common import fetch_json_with_retry, to_gdelt_dt

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def append_gdelt_rows(
    rows: list[dict],
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
    sleep_seconds: float,
) -> None:
    logger.info("Starting GDELT ingestion across %s monthly windows", len(windows))

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        logger.info(
            "Fetching GDELT window %s/%s: %s to %s",
            idx,
            len(windows),
            window_start.date(),
            window_end.date(),
        )

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "startdatetime": to_gdelt_dt(window_start, end_of_day=False),
            "enddatetime": to_gdelt_dt(window_end, end_of_day=True),
        }

        try:
            payload = fetch_json_with_retry(
                url=GDELT_URL,
                params=params,
                provider_name="GDELT",
                max_retries=max_retries,
                base_sleep=retry_base_sleep,
            )
        except Exception as exc:
            logger.warning(
                "Skipping GDELT window %s due to repeated failure: %s",
                window_start.date(),
                exc,
            )
            time.sleep(sleep_seconds)
            continue

        articles = payload.get("articles", [])
        logger.info("Fetched %s GDELT articles", len(articles))

        for article in articles:
            rows.append(
                {
                    "provider": "gdelt",
                    "window_start": window_start.date().isoformat(),
                    "window_end": window_end.date().isoformat(),
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "source": article.get("domain"),
                    "language": article.get("language"),
                    "seen_date": article.get("seendate"),
                    "content": article.get("content") or "",
                    "social_image": article.get("socialimage"),
                    "source_country": article.get("sourcecountry"),
                }
            )

        time.sleep(sleep_seconds)
