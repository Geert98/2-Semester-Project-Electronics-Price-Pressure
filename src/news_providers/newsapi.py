from __future__ import annotations

import logging

import pandas as pd

from src.news_providers.common import RequestThrottler, api_key_from_env, fetch_json_with_retry

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def append_newsapi_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    newsapi_cfg = news_cfg.get("newsapi", {})
    api_key = api_key_from_env(news_cfg, "newsapi", "NEWSAPI_KEY")
    lookback_days = max(1, int(newsapi_cfg.get("lookback_days", 30)))
    language = newsapi_cfg.get("language", "en")
    sort_by = newsapi_cfg.get("sort_by", "publishedAt")
    min_interval = float(newsapi_cfg.get("min_request_interval_seconds", 1.2))
    max_pages = max(1, int(newsapi_cfg.get("max_pages", 1)))
    page_size = min(max_records, 100)
    throttler = RequestThrottler(min_interval)

    window_end = pd.Timestamp.now(tz="UTC").normalize()
    window_start = window_end - pd.Timedelta(days=lookback_days)

    logger.info(
        "Starting NewsAPI ingestion from %s to %s",
        window_start.date(),
        window_end.date(),
    )

    page = 1
    total_results = page_size * max_pages
    fetched_total = 0

    while page <= max_pages and (page - 1) * page_size < total_results:
        params = {
            "apiKey": api_key,
            "q": query,
            "from": window_start.strftime("%Y-%m-%d"),
            "to": window_end.strftime("%Y-%m-%d"),
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page,
        }

        try:
            payload = fetch_json_with_retry(
                url=NEWSAPI_URL,
                params=params,
                provider_name="NewsAPI",
                max_retries=max_retries,
                base_sleep=retry_base_sleep,
                before_request=throttler.wait,
            )
        except Exception as exc:
            logger.warning(
                "Stopping NewsAPI ingestion at page %s due to repeated failure: %s",
                page,
                exc,
            )
            break

        if payload.get("status") == "error":
            logger.warning(
                "Stopping NewsAPI ingestion because API returned error: %s",
                payload.get("message", payload),
            )
            break

        articles = payload.get("articles", [])
        total_results = int(payload.get("totalResults", 0))
        logger.info("Fetched %s NewsAPI articles from page %s", len(articles), page)

        if not articles:
            break

        for article in articles:
            source = article.get("source") or {}
            rows.append(
                {
                    "provider": "newsapi",
                    "window_start": window_start.date().isoformat(),
                    "window_end": window_end.date().isoformat(),
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "source": source.get("name") or source.get("id"),
                    "language": language,
                    "seen_date": article.get("publishedAt"),
                    "content": article.get("content") or "",
                    "social_image": article.get("urlToImage"),
                    "source_country": "",
                }
            )

        fetched_total += len(articles)
        page += 1

    logger.info("Fetched %s NewsAPI articles total", fetched_total)
