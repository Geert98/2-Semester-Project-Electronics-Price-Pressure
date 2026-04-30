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

import pandas as pd
import requests

from src.storage import get_sqlite_path, save_dataframe_to_sqlite
from src.utils import load_config

logger = logging.getLogger(__name__)


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

    # Read the configured series identifier, for example PCU33443344.
    series_id = config["fred"]["series_id"]

    # Build the full CSV download URL using the template from config.
    url = config["fred"]["url_template"].format(series_id=series_id)

    sqlite_path = get_sqlite_path(config)
    table_name = "fred_series"

    logger.info("Downloading FRED data from %s", url)

    # Send the HTTP request to FRED and fail immediately if the request is invalid.
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Load the downloaded CSV into pandas for downstream processing.
    df = pd.read_csv(io.BytesIO(response.content))

    # Standardize column names so the rest of the pipeline can rely on a fixed schema.
    df.columns = ["date", "ppi_value"]

    # Convert the date column into pandas datetime objects.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert the index values to numeric values.
    # Non-numeric rows become NaN and are removed below.
    df["ppi_value"] = pd.to_numeric(df["ppi_value"], errors="coerce")

    # Remove invalid rows and sort chronologically.
    df = df.dropna(subset=["date", "ppi_value"]).sort_values("date").reset_index(drop=True)

    save_dataframe_to_sqlite(df, sqlite_path, table_name)

    logger.info("Saved FRED data to SQLite database %s (table=%s, %s rows)", sqlite_path, table_name, len(df))
    return df


if __name__ == "__main__":
    # Allow the script to be run directly for manual testing.
    ingest_fred()