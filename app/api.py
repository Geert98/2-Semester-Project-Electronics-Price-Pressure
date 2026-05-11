from __future__ import annotations

# This file defines the FastAPI backend for the project.
#
# In the overall system architecture, this file acts as the serving layer.
# It exposes the pipeline and its artifacts through HTTP endpoints so the
# system can be interacted with programmatically.
#
# The API currently supports:
# - a health check endpoint
# - retrieval of the latest prediction artifact
# - retrieval of saved training metrics
# - triggering the full pipeline on demand
#
# This is important for the MLOps setup because it demonstrates that the
# pipeline is not only a local script, but can also be operationalized
# as an API-based service.

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.feature_engineering import build_feature_table
from src.ingest_fred import ingest_fred
from src.ingest_news import ingest_news
from src.predict import predict_latest
from src.preprocess import preprocess_news
from src.storage import load_dataframe_from_mongo
from src.train import train_model
from src.utils import ensure_directories, get_log_level, load_config, setup_env, setup_logging
from src.generate_pages_report import generate_pages_report

# Load environment variables and configure logging when the API starts.
# This ensures the API uses the same environment and logging style
# as the rest of the project.
setup_env()
setup_logging(get_log_level())

# Initialize the FastAPI application.
# The metadata here is used by the automatic Swagger/OpenAPI docs.
app = FastAPI(
    title="Electronics Price Pressure API",
    description="API for running the electronics price pressure pipeline and serving predictions.",
    version="1.0.0",
)


def _prepare_environment() -> dict:
    """
    Load project configuration and ensure required directories exist.

    Why this function exists:
    - Several endpoints need access to the same config and folder structure.
    - Keeping this logic in one helper avoids repetition.
    - It also ensures output folders exist before the API tries to read or write artifacts.

    Returns
    -------
    dict
        The loaded project configuration.
    """
    config = load_config()
    ensure_directories(config["paths"])
    return config


@app.get("/health")
def health() -> dict:
    """
    Simple health check endpoint.

    Why this endpoint exists:
    - It provides a minimal way to verify that the API service is running.
    - It is useful for testing, deployment checks, and monitoring.

    Returns
    -------
    dict
        Basic service status information.
    """
    return {
        "status": "ok",
        "service": "electronics-price-pressure-api",
    }


@app.get("/latest-prediction")
def latest_prediction() -> dict:
    """
    Return the latest saved prediction artifact.

    Why this endpoint exists:
    - The pipeline saves its most recent prediction to a CSV file.
    - This endpoint makes that prediction available through the API,
      so it can be consumed by the frontend or by other systems.

    Returns
    -------
    dict
        JSON response containing the latest prediction record.

    Raises
    ------
    HTTPException
        If the prediction file does not exist or is empty.
    """
    config = _prepare_environment()
    pred_path = Path(config["paths"]["predictions_dir"]) / "latest_prediction.csv"

    # If the prediction file has not been created yet, return a clear API error.
    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No prediction file found. Run the pipeline first via POST /run-pipeline.",
        )

    # Load the saved prediction artifact.
    df = pd.read_csv(pred_path)

    # Protect against the case where the file exists but contains no rows.
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Prediction file exists but is empty.",
        )

    # Convert the first row to a plain dictionary for JSON output.
    record = df.iloc[0].to_dict()

    return {
        "status": "success",
        "prediction": record,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    """
    Return the saved training metrics artifact.

    Why this endpoint exists:
    - The model training step saves evaluation metrics as JSON.
    - This endpoint exposes those metrics through the API so they can be
      displayed in the frontend or inspected by a user.

    Returns
    -------
    dict
        JSON response containing the saved training metrics.

    Raises
    ------
    HTTPException
        If the metrics file does not exist.
    """
    config = _prepare_environment()
    metrics_path = Path(config["paths"]["metrics_dir"]) / "train_metrics.json"

    # If the metrics artifact does not exist yet, return a clear API error.
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics file found. Run the pipeline first via POST /run-pipeline.",
        )

    # Load the saved metrics JSON artifact.
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "status": "success",
        "metrics": metrics,
    }


@app.get("/news-articles")
def get_news_articles(
    limit: int = Query(default=50, ge=1, le=500),
    cleaned: bool = Query(default=True),
) -> dict:
    """
    Return recent ingested news articles.

    Why this endpoint exists:
    - The news articles are a core input to the feature pipeline.
    - Exposing a compact article list makes ingestion easier to inspect
      without connecting directly to MongoDB.

    Parameters
    ----------
    limit : int
        Maximum number of articles to return.
    cleaned : bool
        If True, read from the cleaned news collection. Otherwise read raw news.

    Returns
    -------
    dict
        JSON response containing recent article metadata.
    """
    config = _prepare_environment()
    mongo_cfg = config["storage"]["mongo"]
    collection_name = (
        mongo_cfg["clean_news_collection"]
        if cleaned
        else mongo_cfg["raw_news_collection"]
    )

    sort_by = "published_at" if cleaned else "seen_date"
    df = load_dataframe_from_mongo(config, collection_name, sort_by=sort_by)

    if df.empty:
        return {
            "status": "success",
            "collection": collection_name,
            "count": 0,
            "articles": [],
        }

    preferred_columns = [
        "provider",
        "published_at",
        "seen_date",
        "month",
        "title",
        "url",
        "source",
        "language",
        "source_country",
        "clean_text",
    ]
    columns = [col for col in preferred_columns if col in df.columns]
    article_df = df[columns].tail(limit).iloc[::-1].copy()

    for date_col in ["published_at", "seen_date", "month"]:
        if date_col in article_df.columns:
            article_df[date_col] = pd.to_datetime(article_df[date_col], errors="coerce")
            article_df[date_col] = article_df[date_col].dt.strftime("%Y-%m-%d")
            article_df[date_col] = article_df[date_col].fillna("")

    article_df = article_df.fillna("")

    return {
        "status": "success",
        "collection": collection_name,
        "count": int(len(article_df)),
        "articles": article_df.to_dict(orient="records"),
    }


@app.post("/run-pipeline")
def run_pipeline() -> dict:
    """
    Run the full pipeline end-to-end through the API.

    Why this endpoint exists:
    - It allows the pipeline to be triggered on demand instead of only
      from the command line.
    - This demonstrates an API-based deployment scenario, which aligns well
      with the MLOps assignment requirements.

    Pipeline order
    --------------
    1. Ingest FRED data
    2. Ingest news data
    3. Preprocess news
    4. Build the feature table
    5. Train the model
    6. Generate the latest prediction

    Returns
    -------
    dict
        JSON response containing a pipeline summary and key artifacts.

    Raises
    ------
    HTTPException
        If any pipeline step fails.
    """
    _prepare_environment()

    try:
        # Step 1: Structured external data ingestion.
        fred_df = ingest_fred()

        # Step 2: Unstructured external data ingestion.
        news_df = ingest_news()

        # Step 3: News preprocessing.
        clean_df = preprocess_news()

        # Step 4: Feature engineering and target creation.
        model_table_df = build_feature_table()

        # Step 5: Model training and evaluation.
        metrics = train_model()

        # Step 6: Final prediction artifact creation.
        prediction_df = predict_latest()

        # Step 7: Generate the static dashboard for GitHub Pages.
        report_path = generate_pages_report()

        # Return a compact execution summary to the caller.
        return {
            "status": "success",
            "message": "Pipeline completed successfully.",
            "artifacts": {
                "fred_rows": int(len(fred_df)),
                "raw_news_rows": int(len(news_df)),
                "clean_news_rows": int(len(clean_df)),
                "model_table_rows": int(len(model_table_df)),
                "latest_prediction": prediction_df.iloc[0].to_dict(),
                "train_metrics_summary": {
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "n_rows_train": metrics.get("n_rows_train"),
                    "n_rows_test": metrics.get("n_rows_test"),
                },
            },
        }

    except Exception as exc:
        # Wrap internal errors as a standard API response.
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc
