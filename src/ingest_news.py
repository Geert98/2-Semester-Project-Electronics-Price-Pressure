from __future__ import annotations

# This script is responsible for ingesting news data from external news APIs.
#
# In this project, news articles are used as an external unstructured data source
# that complements the structured FRED time-series data.
#
# The script:
# 1. reads the news ingestion settings from the shared config file
# 2. creates monthly time windows for the requested period
# 3. queries the configured news API for each month
# 4. retries failed requests with exponential backoff
# 5. stores the raw article-level results in MongoDB
#
# Saving the raw news output is important for reproducibility,
# because it preserves the exact article snapshot used by the pipeline.

import logging
import os
import time
from collections.abc import Callable

import pandas as pd
import requests

from src.storage import save_dataframe_to_mongo
from src.utils import load_config, setup_env

logger = logging.getLogger(__name__)

# Base endpoint for the GDELT Document API.
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Base endpoint for The Guardian Open Platform Content API.
GUARDIAN_URL = "https://content.guardianapis.com/search"

# Base endpoint for the NewsData.io API.
NEWSDATA_BASE_URL = "https://newsdata.io/api/1"


def _to_gdelt_dt(dt: pd.Timestamp, end_of_day: bool = False) -> str:
    """
    Convert a pandas timestamp to the datetime format expected by GDELT.

    GDELT expects timestamps in the format:
    YYYYMMDDHHMMSS

    Parameters
    ----------
    dt : pd.Timestamp
        Timestamp to convert.
    end_of_day : bool
        If True, use 23:59:59 for the time component.
        Otherwise use 00:00:00.

    Returns
    -------
    str
        Timestamp string in GDELT format.
    """
    if end_of_day:
        return dt.strftime("%Y%m%d") + "235959"
    return dt.strftime("%Y%m%d") + "000000"


def _to_guardian_date(dt: pd.Timestamp) -> str:
    """
    Convert a pandas timestamp to the date format expected by The Guardian API.
    """
    return dt.strftime("%Y-%m-%d")


def _month_windows(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate monthly start/end windows between two dates.

    Why this function exists:
    - The project aggregates news data at the monthly level.
    - Querying GDELT month by month makes it easier to align article data
      with the monthly FRED indicator used later in the pipeline.

    Parameters
    ----------
    start_date : str
        Start date as a string.
    end_date : str
        End date as a string.

    Returns
    -------
    list[tuple[pd.Timestamp, pd.Timestamp]]
        List of monthly (start, end) tuples.
    """
    start = pd.to_datetime(start_date).to_period("M").to_timestamp()
    end = pd.to_datetime(end_date).to_period("M").to_timestamp()
    months = pd.date_range(start=start, end=end, freq="MS")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for month_start in months:
        # MonthEnd(1) gives the end of the same month.
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        windows.append((month_start, month_end))

    return windows


def _response_retry_sleep(response: requests.Response, fallback_sleep: float) -> float:
    """
    Read an HTTP Retry-After value when the API provides one.
    """
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
    """
    Query a news API with retry logic and exponential backoff.

    Why this function exists:
    - News APIs can return rate-limit errors (HTTP 429).
    - Instead of failing immediately, the pipeline waits and retries.
    - Exponential backoff reduces pressure on the API and improves robustness.

    Parameters
    ----------
    url : str
        API endpoint URL.
    params : dict
        Query parameters sent to the API.
    provider_name : str
        Human-readable provider name used in log messages.
    max_retries : int
        Maximum number of retry attempts.
    base_sleep : float
        Base number of seconds used for exponential backoff.
    before_request : Callable[[], None] | None
        Optional hook used to throttle requests before they are sent.

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    requests.RequestException
        If all retries fail.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            if before_request:
                before_request()

            response = requests.get(url, params=params, timeout=60)

            # If the API responds with HTTP 429, wait and retry.
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

            # Raise an exception for any other HTTP error codes.
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

    # If all attempts fail, raise the last captured exception.
    if last_exc:
        raise last_exc

    return {}


def _fetch_gdelt_with_retry(
    params: dict,
    max_retries: int = 5,
    base_sleep: float = 5.0,
) -> dict:
    """
    Query the GDELT API with retry logic and exponential backoff.
    """
    return _fetch_json_with_retry(
        url=GDELT_URL,
        params=params,
        provider_name="GDELT",
        max_retries=max_retries,
        base_sleep=base_sleep,
    )


class RequestThrottler:
    """
    Keep requests spaced out so provider rate limits are respected.
    """

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


def _fetch_guardian_with_retry(
    params: dict,
    max_retries: int = 5,
    base_sleep: float = 5.0,
    throttler: RequestThrottler | None = None,
) -> dict:
    """
    Query The Guardian API with retry logic and request throttling.
    """
    return _fetch_json_with_retry(
        url=GUARDIAN_URL,
        params=params,
        provider_name="Guardian",
        max_retries=max_retries,
        base_sleep=base_sleep,
        before_request=throttler.wait if throttler else None,
    )


def _fetch_newsdata_with_retry(
    endpoint: str,
    params: dict,
    max_retries: int = 5,
    base_sleep: float = 5.0,
    throttler: RequestThrottler | None = None,
) -> dict:
    """
    Query NewsData.io with retry logic and request throttling.
    """
    endpoint = endpoint.strip("/")
    return _fetch_json_with_retry(
        url=f"{NEWSDATA_BASE_URL}/{endpoint}",
        params=params,
        provider_name="NewsData",
        max_retries=max_retries,
        base_sleep=base_sleep,
        before_request=throttler.wait if throttler else None,
    )


def _api_key_from_env(news_cfg: dict, provider: str, default_env: str) -> str:
    """
    Read a provider API key from the environment.
    """
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


def _guardian_api_key(news_cfg: dict) -> str:
    """
    Read The Guardian API key from the environment.
    """
    return _api_key_from_env(news_cfg, "guardian", "GUARDIAN_API_KEY")


def _newsdata_api_key(news_cfg: dict) -> str:
    """
    Read the NewsData.io API key from the environment.
    """
    return _api_key_from_env(news_cfg, "newsdata", "NEWSDATA_API_KEY")


def _append_gdelt_rows(
    rows: list[dict],
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
    sleep_seconds: float,
) -> None:
    """
    Fetch GDELT rows into the shared raw news schema.
    """
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
            "startdatetime": _to_gdelt_dt(window_start, end_of_day=False),
            "enddatetime": _to_gdelt_dt(window_end, end_of_day=True),
        }

        try:
            payload = _fetch_gdelt_with_retry(
                params=params,
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
                    "social_image": article.get("socialimage"),
                    "source_country": article.get("sourcecountry"),
                }
            )

        time.sleep(sleep_seconds)


def _append_guardian_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    """
    Fetch Guardian rows into the shared raw news schema.
    """
    guardian_cfg = news_cfg.get("guardian", {})
    api_key = _guardian_api_key(news_cfg)
    min_interval = float(guardian_cfg.get("min_request_interval_seconds", 1.2))
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

        params = {
            "api-key": api_key,
            "q": query,
            "from-date": _to_guardian_date(window_start),
            "to-date": _to_guardian_date(window_end),
            "page": 1,
            "page-size": page_size,
            "order-by": guardian_cfg.get("order_by", "newest"),
            "show-fields": guardian_cfg.get("show_fields", "trailText,thumbnail"),
        }

        try:
            payload = _fetch_guardian_with_retry(
                params=params,
                max_retries=max_retries,
                base_sleep=retry_base_sleep,
                throttler=throttler,
            )
        except Exception as exc:
            logger.warning(
                "Skipping Guardian window %s due to repeated failure: %s",
                window_start.date(),
                exc,
            )
            continue

        response = payload.get("response", {})
        articles = response.get("results", [])
        logger.info("Fetched %s Guardian articles", len(articles))

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
                    "social_image": fields.get("thumbnail"),
                    "source_country": "",
                }
            )


def _append_newsdata_rows(
    rows: list[dict],
    news_cfg: dict,
    query: str,
    max_records: int,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    """
    Fetch NewsData.io rows into the shared raw news schema.
    """
    newsdata_cfg = news_cfg.get("newsdata", {})
    api_key = _newsdata_api_key(news_cfg)
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
            "from_date": _to_guardian_date(window_start),
            "to_date": _to_guardian_date(window_end),
            "size": size,
        }

        try:
            payload = _fetch_newsdata_with_retry(
                endpoint=endpoint,
                params=params,
                max_retries=max_retries,
                base_sleep=retry_base_sleep,
                throttler=throttler,
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
                    "social_image": article.get("image_url"),
                    "source_country": ",".join(article.get("country", []))
                    if isinstance(article.get("country"), list)
                    else article.get("country"),
                }
            )


def ingest_news(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Download monthly news article lists from configured APIs and save them.

    Why this function exists:
    - The project uses news titles as an unstructured data source for later
      preprocessing, keyword extraction, and sentiment analysis.
    - The article data is pulled month by month to align with the downstream
      monthly feature engineering process.
    - The raw article-level dataset is stored in MongoDB for reproducibility.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Raw article-level DataFrame.
    """
    # Load project configuration so the ingestion behavior can be controlled
    # centrally from config.yaml.
    config = load_config(config_path)
    news_cfg = config["news"]

    # Read configuration values for the news ingestion process.
    query = news_cfg["query"].strip()
    start_date = news_cfg["start_date"]
    end_date = news_cfg["end_date"]
    max_records = int(news_cfg["max_records_per_window"])
    sleep_seconds = float(news_cfg.get("sleep_seconds", 5))
    max_retries = int(news_cfg.get("max_retries", 5))
    retry_base_sleep = float(news_cfg.get("retry_base_sleep", 5))

    # Create monthly windows covering the configured period.
    windows = _month_windows(start_date, end_date)

    use_gdelt = bool(news_cfg.get("use_gdelt", True))
    use_guardian = bool(news_cfg.get("use_guardian", False))
    use_newsdata = bool(news_cfg.get("use_newsdata", False))

    if not use_gdelt and not use_guardian and not use_newsdata:
        raise ValueError(
            "At least one news provider must be enabled: "
            "use_gdelt, use_guardian, or use_newsdata."
        )

    # This list will collect one dictionary per article across all providers.
    rows: list[dict] = []

    if use_gdelt:
        _append_gdelt_rows(
            rows=rows,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
            sleep_seconds=sleep_seconds,
        )

    if use_guardian:
        _append_guardian_rows(
            rows=rows,
            news_cfg=news_cfg,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

    if use_newsdata:
        _append_newsdata_rows(
            rows=rows,
            news_cfg=news_cfg,
            query=query,
            max_records=max_records,
            windows=windows,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

    # Define the expected schema explicitly so the collection still has columns
    # even if no articles were fetched.
    expected_columns = [
        "provider",
        "window_start",
        "window_end",
        "title",
        "url",
        "source",
        "language",
        "seen_date",
        "social_image",
        "source_country",
    ]

    df = pd.DataFrame(rows, columns=expected_columns)

    raw_collection = config["storage"]["mongo"]["raw_news_collection"]
    save_dataframe_to_mongo(df, config, raw_collection)

    logger.info("Saved raw news to MongoDB collection %s (%s rows)", raw_collection, len(df))
    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual ingestion testing.
    ingest_news()
