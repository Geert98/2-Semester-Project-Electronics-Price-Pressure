from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import pandas as pd
import requests

from src.utils import setup_env

logger = logging.getLogger(__name__)


def to_gdelt_dt(dt: pd.Timestamp, end_of_day: bool = False) -> str:
    if end_of_day:
        return dt.strftime("%Y%m%d") + "235959"
    return dt.strftime("%Y%m%d") + "000000"


def resolve_config_date(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"today", "now", "current_date"}:
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    return str(value)


def month_windows(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.to_datetime(resolve_config_date(start_date)).to_period("M").to_timestamp()
    end = pd.to_datetime(resolve_config_date(end_date)).to_period("M").to_timestamp()
    months = pd.date_range(start=start, end=end, freq="MS")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for month_start in months:
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        windows.append((month_start, month_end))

    return windows


def response_retry_sleep(response: requests.Response, fallback_sleep: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return fallback_sleep

    try:
        return max(float(retry_after), fallback_sleep)
    except ValueError:
        return fallback_sleep


def fetch_json_with_retry(
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
                sleep_for = response_retry_sleep(response, fallback_sleep)
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


def api_key_from_env(news_cfg: dict, provider: str, default_env: str) -> str:
    setup_env()
    provider_cfg = news_cfg.get(provider, {})
    api_key_env = provider_cfg.get("api_key_env", default_env)
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"{provider} news ingestion is enabled, but {api_key_env} is not set. "
            "Add the key to your environment or .env file."
        )
    return api_key
