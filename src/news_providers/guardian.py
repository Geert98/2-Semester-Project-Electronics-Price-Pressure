from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import pandas as pd
import requests

from src.utils import setup_env

logger = logging.getLogger(__name__)

GUARDIAN_URL = "https://content.guardianapis.com/search"


def _to_guardian_date(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def _response_retry_sleep(response: requests.Response, fallback_sleep: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return fallback_sleep

    try:
        return max(float(retry_after), fallback_sleep)
    except ValueError:
        return fallback_sleep


def _fetch_json_with_retry(
    url: str,
    params: dict,
    provider_name: str,
    max_retries: int = 5,
    base_sleep: float = 5.0,
    before_request: Callable[[], None] | None = None,
) -> dict:
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            if before_request:
                before_request()

            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 429:
                fallback_sleep = base_sleep * (2**attempt)
                sleep_for = _response_retry_sleep(response, fallback_sleep)
                logger.warning(
                    "429 from %s. Sleeping %.1f seconds before retry.",
                    provider_name,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue

            if 400 <= response.status_code < 500:
                logger.warning(
                    "Non-retryable %s response from %s.",
                    response.status_code,
                    provider_name,
                )
                return response.json()

            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            last_exc = exc
            sleep_for = base_sleep * (2**attempt)
            logger.warning(
                "%s request failed (%s). Retrying in %.1f seconds. Attempt %s/%s",
                provider_name,
                exc,
                sleep_for,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_for)

    if last_exc:
        raise last_exc

    return {}


class RequestThrottler:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.last_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        elapsed = time.monotonic() - self.last_request_at
        sleep_for = self.min_interval_seconds - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        self.last_request_at = time.monotonic()


def _guardian_api_key(news_cfg: dict) -> str:
    setup_env()
    provider_cfg = news_cfg.get("guardian", {})
    api_key_env = provider_cfg.get("api_key_env", "GUARDIAN_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"guardian news ingestion is enabled, but {api_key_env} is not set. "
            "Add the key to your environment or .env file."
        )
    return api_key


def append_guardian_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    guardian_cfg = news_cfg.get("guardian", {})
    api_key = _guardian_api_key(news_cfg)
    min_interval = float(guardian_cfg.get("min_request_interval_seconds", 1.2))
    max_pages = max(1, int(guardian_cfg.get("max_pages_per_window", 1)))
    page_size = min(max_records, 50)
    throttler = RequestThrottler(min_interval)

    if page_size < max_records:
        logger.warning(
            "Guardian page-size is capped at %s records per window. Requested %s.",
            page_size,
            max_records,
        )

    logger.info("Starting Guardian ingestion across %s monthly windows", len(windows))

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        logger.info(
            "Fetching Guardian window %s/%s: %s to %s",
            idx,
            len(windows),
            window_start.date(),
            window_end.date(),
        )

        page = 1
        total_pages = max_pages
        fetched_for_window = 0

        while page <= max_pages and page <= total_pages:
            params = {
                "api-key": api_key,
                "q": query,
                "from-date": _to_guardian_date(window_start),
                "to-date": _to_guardian_date(window_end),
                "page": page,
                "page-size": page_size,
                "order-by": guardian_cfg.get("order_by", "newest"),
                "show-fields": guardian_cfg.get("show_fields", "bodyText,thumbnail"),
            }

            try:
                payload = _fetch_json_with_retry(
                    url=GUARDIAN_URL,
                    params=params,
                    provider_name="Guardian",
                    max_retries=max_retries,
                    base_sleep=retry_base_sleep,
                    before_request=throttler.wait,
                )
            except Exception as exc:
                logger.warning(
                    "Stopping Guardian window %s at page %s due to repeated failure: %s",
                    window_start.date(),
                    page,
                    exc,
                )
                break

            response = payload.get("response", {})
            articles = response.get("results", [])
            total_pages = min(int(response.get("pages", 1)), max_pages)
            logger.info(
                "Fetched %s Guardian articles from page %s/%s",
                len(articles),
                page,
                total_pages,
            )

            if not articles:
                break

            for article in articles:
                fields = article.get("fields", {})
                rows.append(
                    {
                        "provider": "guardian",
                        "window_start": window_start.date().isoformat(),
                        "window_end": window_end.date().isoformat(),
                        "title": article.get("webTitle"),
                        "url": article.get("webUrl"),
                        "source": "The Guardian",
                        "language": "en",
                        "seen_date": article.get("webPublicationDate"),
                        "content": fields.get("bodyText") or fields.get("body") or fields.get("trailText") or "",
                        "social_image": fields.get("thumbnail"),
                        "source_country": "",
                    }
                )

            fetched_for_window += len(articles)
            page += 1

        logger.info("Fetched %s Guardian articles for window", fetched_for_window)
