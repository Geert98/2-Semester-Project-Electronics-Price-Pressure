from __future__ import annotations

import logging

import pandas as pd

from src.news_providers.common import RequestThrottler, api_key_from_env, fetch_json_with_retry

logger = logging.getLogger(__name__)

WEBZIO_BASE_URL = "https://api.webz.io/newsApiLite"


def append_webzio_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    webzio_cfg = news_cfg.get("webzio", {})
    api_key = api_key_from_env(news_cfg, "webzio", "WEBZIO_API_KEY")
    language = webzio_cfg.get("language", "english")
    min_interval = float(webzio_cfg.get("min_request_interval_seconds", 1.5))
    max_pages = max(1, int(webzio_cfg.get("max_pages_per_window", 2)))
    deduplicate = webzio_cfg.get("performance", {}).get("deduplicate", True)
    timeout_seconds = float(webzio_cfg.get("performance", {}).get("timeout_seconds", 30))
    
    # Use webzio-specific query if available, otherwise use the main query
    webzio_query = news_cfg.get("query_webzio") or query
    
    throttler = RequestThrottler(min_interval)

    logger.info("Starting WebzIO ingestion across %s monthly windows", len(windows))

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        logger.info(
            "Fetching WebzIO window %s/%s: %s to %s",
            idx,
            len(windows),
            window_start.date(),
            window_end.date(),
        )

        # Build the query - newsApiLite doesn't support inline date filters
        # Use the ts parameter for date range instead
        # Convert start date to Unix timestamp in milliseconds (web.io expects milliseconds)
        ts_unix_ms = int(window_start.timestamp() * 1000)

        params = {
            "token": api_key,
            "q": webzio_query,
            "language": language,
            "ts": ts_unix_ms,
            # Note: newsApiLite doesn't support sorting parameter
        }

        fetched_total = 0
        next_url = None
        pages_fetched = 0

        while pages_fetched < max_pages:
            try:
                payload = fetch_json_with_retry(
                    url=WEBZIO_BASE_URL,
                    params=params,
                    provider_name="WebzIO",
                    max_retries=max_retries,
                    base_sleep=retry_base_sleep,
                    before_request=throttler.wait,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping WebzIO window %s due to repeated failure: %s",
                    window_start.date(),
                    exc,
                )
                break

            # Check for errors
            if payload.get("error"):
                logger.warning(
                    "Skipping WebzIO window %s because API returned error: %s",
                    window_start.date(),
                    payload.get("error"),
                )
                break

            articles = payload.get("posts", [])
            logger.info(
                "Fetched %s WebzIO articles from page %s",
                len(articles),
                pages_fetched + 1,
            )

            if not articles:
                break

            for article in articles:
                rows.append(
                    {
                        "provider": "webzio",
                        "window_start": window_start.date().isoformat(),
                        "window_end": window_end.date().isoformat(),
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "source": article.get("site") or article.get("thread", {}).get("site"),
                        "language": language,
                        "seen_date": article.get("crawled") or article.get("published"),
                        "content": article.get("text", "")[:5000],  # Limit text to 5000 chars
                        "source_country": article.get("country"),
                        "author": article.get("author"),
                        "sentiment": article.get("sentiment"),
                    }
                )
                fetched_total += 1

            # Check if we have more results
            next_url = payload.get("next")
            if not next_url:
                break

            # Extract the parameters from the next URL for the next request
            # WebzIO returns full URL with token and parameters
            try:
                params = {"token": api_key}  # Reset params with token
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(next_url)
                for key, value in parse_qs(parsed.query).items():
                    if key != "token":  # Don't override token
                        params[key] = value[0] if value else ""
            except Exception as exc:
                logger.warning("Failed to parse next URL from WebzIO: %s", exc)
                break

            pages_fetched += 1

        logger.info(
            "Completed WebzIO window %s/%s with %s total articles",
            idx,
            len(windows),
            fetched_total,
        )
