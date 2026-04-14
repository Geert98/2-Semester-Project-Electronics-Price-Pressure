from __future__ import annotations

# This script is responsible for ingesting news data from the GDELT API.
#
# In this project, news articles are used as an external unstructured data source
# that complements the structured FRED time-series data.
#
# The script:
# 1. reads the news ingestion settings from the shared config file
# 2. creates monthly time windows for the requested period
# 3. queries the GDELT Doc API for each month
# 4. retries failed requests with exponential backoff
# 5. stores the raw article-level results as a CSV file
#
# Saving the raw news output locally is important for reproducibility,
# because it preserves the exact article snapshot used by the pipeline.

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils import load_config

logger = logging.getLogger(__name__)

# Base endpoint for the GDELT Document API.
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


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


def _fetch_gdelt_with_retry(
    params: dict,
    max_retries: int = 5,
    base_sleep: float = 5.0,
) -> dict:
    """
    Query the GDELT API with retry logic and exponential backoff.

    Why this function exists:
    - The GDELT API can return rate-limit errors (HTTP 429).
    - Instead of failing immediately, the pipeline waits and retries.
    - Exponential backoff reduces pressure on the API and improves robustness.

    Parameters
    ----------
    params : dict
        Query parameters sent to the GDELT API.
    max_retries : int
        Maximum number of retry attempts.
    base_sleep : float
        Base number of seconds used for exponential backoff.

    Returns
    -------
    dict
        Parsed JSON response from GDELT.

    Raises
    ------
    requests.RequestException
        If all retries fail.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=60)

            # If GDELT responds with HTTP 429, wait and retry.
            if response.status_code == 429:
                sleep_for = base_sleep * (2**attempt)
                logger.warning("429 from GDELT. Sleeping %.1f seconds before retry.", sleep_for)
                time.sleep(sleep_for)
                continue

            # Raise an exception for any other HTTP error codes.
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            last_exc = exc
            sleep_for = base_sleep * (2**attempt)
            logger.warning(
                "Request failed (%s). Retrying in %.1f seconds. Attempt %s/%s",
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


def ingest_news(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Download monthly news article lists from the GDELT API and save them locally.

    Why this function exists:
    - The project uses news titles as an unstructured data source for later
      preprocessing, keyword extraction, and sentiment analysis.
    - The article data is pulled month by month to align with the downstream
      monthly feature engineering process.
    - The raw article-level dataset is stored as a CSV artifact for reproducibility.

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

    # This list will collect one dictionary per article across all months.
    rows: list[dict] = []

    logger.info("Starting news ingestion across %s monthly windows", len(windows))

    for idx, (window_start, window_end) in enumerate(windows, start=1):
        logger.info(
            "Fetching window %s/%s: %s to %s",
            idx,
            len(windows),
            window_start.date(),
            window_end.date(),
        )

        # Build the query parameters for the GDELT API.
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
            # If repeated retries still fail, skip the month instead of crashing
            # the entire pipeline. This keeps the ingestion step operational.
            logger.warning("Skipping window %s due to repeated failure: %s", window_start.date(), exc)
            time.sleep(sleep_seconds)
            continue

        articles = payload.get("articles", [])
        logger.info("Fetched %s articles", len(articles))

        # Extract only the fields needed for downstream preprocessing and analysis.
        for article in articles:
            rows.append(
                {
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

        # Small delay between successful month requests to reduce API pressure.
        time.sleep(sleep_seconds)

    # Define the expected schema explicitly so the CSV still has columns
    # even if no articles were fetched.
    expected_columns = [
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

    output_path = Path(config["paths"]["raw_dir"]) / "raw_news.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Saved raw news to %s (%s rows)", output_path, len(df))
    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual ingestion testing.
    ingest_news()
