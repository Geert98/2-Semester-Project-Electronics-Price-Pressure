from __future__ import annotations

import logging

import pandas as pd

from src.news_providers.common import RequestThrottler, api_key_from_env, fetch_json_with_retry

logger = logging.getLogger(__name__)

NEWSDATA_BASE_URL = "https://newsdata.io/api/1"


def append_newsdata_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    newsdata_cfg = news_cfg.get("newsdata", {})
    api_key = api_key_from_env(news_cfg, "newsdata", "NEWSDATA_API_KEY")
    endpoint = newsdata_cfg.get("endpoint", "archive")
    language = newsdata_cfg.get("language", "en")
    min_interval = float(newsdata_cfg.get("min_request_interval_seconds", 31))
    size = min(max_records, 50)
    throttler = RequestThrottler(min_interval)

    if size < max_records:
        logger.warning(
            "NewsData size is capped at %s records per window. Requested %s.",
            size,
            max_records,
        )

    logger.info("Starting NewsData ingestion across %s monthly windows", len(windows))

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        logger.info(
            "Fetching NewsData window %s/%s: %s to %s",
            idx,
            len(windows),
            window_start.date(),
            window_end.date(),
        )

        params = {
            "apikey": api_key,
            "q": query,
            "language": language,
            "from_date": window_start.strftime("%Y-%m-%d"),
            "to_date": window_end.strftime("%Y-%m-%d"),
            "size": size,
        }

        try:
            payload = fetch_json_with_retry(
                url=f"{NEWSDATA_BASE_URL}/{endpoint.strip('/')}",
                params=params,
                provider_name="NewsData",
                max_retries=max_retries,
                base_sleep=retry_base_sleep,
                before_request=throttler.wait,
            )
        except Exception as exc:
            logger.warning(
                "Skipping NewsData window %s due to repeated failure: %s",
                window_start.date(),
                exc,
            )
            continue

        if payload.get("status") == "error":
            logger.warning(
                "Skipping NewsData window %s because API returned error: %s",
                window_start.date(),
                payload.get("message", payload),
            )
            continue

        articles = payload.get("results", [])
        logger.info("Fetched %s NewsData articles", len(articles))

        for article in articles:
            rows.append(
                {
                    "provider": "newsdata",
                    "window_start": window_start.date().isoformat(),
                    "window_end": window_end.date().isoformat(),
                    "title": article.get("title"),
                    "url": article.get("link"),
                    "source": article.get("source_id") or article.get("source_name"),
                    "language": article.get("language") or language,
                    "seen_date": article.get("pubDate"),
                    "content": article.get("content") or "",
                    "social_image": article.get("image_url"),
                    "source_country": ",".join(article.get("country", []))
                    if isinstance(article.get("country"), list)
                    else article.get("country"),
                }
            )
