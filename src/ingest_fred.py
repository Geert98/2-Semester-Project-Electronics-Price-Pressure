from __future__ import annotations

# This script is responsible for downloading the external market indicator
# used in the project: a FRED producer price index (PPI) series.
#
# In this project, the FRED series acts as the core structured time-series input
# and is later used for feature engineering and target construction.
#
# The script:
# 1. reads the series ID and download URL from the shared config file
# 2. downloads the CSV from FRED
# 3. stores the raw external file locally
# 4. loads it into a pandas DataFrame
# 5. standardizes column names and data types
#
# Saving the downloaded file locally is useful for reproducibility,
# because the pipeline keeps a copy of the external data it used.

import logging
import io
from typing import Any

import pandas as pd
import requests

from src.storage import get_sqlite_path, save_dataframe_to_sqlite
from src.utils import load_config

logger = logging.getLogger(__name__)


def _download_fred_series(series_id: str, url_template: str, value_name: str) -> pd.DataFrame:
    """
    Download one FRED series and return standardized date/value columns.
    """
    url = url_template.format(series_id=series_id)
    logger.info("Downloading FRED data from %s", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"No data returned for FRED series: {series_id}")

    df = df.iloc[:, :2].copy()
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date", value_name]).sort_values("date").reset_index(drop=True)
    return df


def _download_ny_fed_gscpi(url: str, value_name: str) -> pd.DataFrame:
    """
    Download the New York Fed GSCPI spreadsheet and return standardized columns.
    """
    logger.info("Downloading New York Fed GSCPI data from %s", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_excel(io.BytesIO(response.content), sheet_name="GSCPI Monthly Data")
    df = df[["Date", "GSCPI"]].copy()
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date", value_name]).sort_values("date").reset_index(drop=True)
    return df


def _build_monthly_indicators(
    indicator_cfg: list[dict[str, Any]],
    url_template: str,
) -> pd.DataFrame:
    """
    Download configured FRED indicator series and aggregate them to monthly rows.
    """
    monthly_frames: list[pd.DataFrame] = []

    for indicator in indicator_cfg:
        source = indicator.get("source", "fred")
        series_id = indicator.get("series_id", source)
        name = indicator.get("name", str(series_id).lower())

        if source == "ny_fed_gscpi":
            raw_df = _download_ny_fed_gscpi(indicator["url"], name)
        else:
            raw_df = _download_fred_series(series_id, url_template, name)

        raw_df["month"] = raw_df["date"].dt.to_period("M").dt.to_timestamp()
        monthly_df = (
            raw_df.groupby("month", as_index=False)[name]
            .mean()
            .sort_values("month")
            .reset_index(drop=True)
        )
        monthly_frames.append(monthly_df)

    if not monthly_frames:
        return pd.DataFrame()

    indicators_df = monthly_frames[0]
    for monthly_df in monthly_frames[1:]:
        indicators_df = indicators_df.merge(monthly_df, on="month", how="outer")

    return indicators_df.sort_values("month").reset_index(drop=True)


def ingest_fred(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Download the configured FRED time series as CSV and save it locally.

    Why this function exists:
    - The project needs a structured market indicator that can be updated
      automatically as part of the pipeline.
    - FRED provides a simple CSV endpoint that is easy to integrate.
    - The downloaded file is stored in the external data folder so that
      the pipeline has a local copy of the source data.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing:
        - date
        - ppi_value
    """
    # Load project configuration so the script can read
    # the correct FRED series ID and output directory.
    config = load_config(config_path)

    fred_cfg = config["fred"]

    # Read the configured target series identifier, for example PCU33443344.
    series_id = fred_cfg["series_id"]
    url_template = fred_cfg["url_template"]

    sqlite_path = get_sqlite_path(config)
    table_name = "fred_series"

    df = _download_fred_series(series_id, url_template, "ppi_value")

    save_dataframe_to_sqlite(df, sqlite_path, table_name)

    logger.info("Saved FRED data to SQLite database %s (table=%s, %s rows)", sqlite_path, table_name, len(df))

    indicators_cfg = fred_cfg.get("indicators", [])
    if indicators_cfg:
        indicators_df = _build_monthly_indicators(indicators_cfg, url_template)
        save_dataframe_to_sqlite(indicators_df, sqlite_path, "fred_indicators")
        logger.info(
            "Saved FRED indicators to SQLite database %s (table=fred_indicators, %s rows)",
            sqlite_path,
            len(indicators_df),
        )

    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual testing.
    ingest_fred()
