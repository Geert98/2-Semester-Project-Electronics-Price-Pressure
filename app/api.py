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


def _load_latest_prediction_record(config: dict) -> dict:
    """
    Load the latest prediction record from the prediction artifact CSV.

    Raises
    ------
    HTTPException
        If the prediction artifact is missing or empty.
    """
    pred_path = Path(config["paths"]["predictions_dir"]) / "latest_prediction.csv"

    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No prediction file found. Run the pipeline first via POST /run-pipeline.",
        )

    df = pd.read_csv(pred_path)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Prediction file exists but is empty.",
        )

    return df.iloc[0].to_dict()


def _load_metrics(config: dict) -> dict:
    """
    Load the training metrics artifact from disk.

    Raises
    ------
    HTTPException
        If the metrics artifact does not exist.
    """
    metrics_path = Path(config["paths"]["metrics_dir"]) / "train_metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics file found. Run the pipeline first via POST /run-pipeline.",
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics_table(config: dict, filename: str) -> list[dict]:
    """
    Load a CSV metrics artifact and return JSON-safe row records.

    Raises
    ------
    HTTPException
        If the file is missing.
    """
    metrics_path = Path(config["paths"]["metrics_dir"]) / filename

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics artifact '{filename}' was not found.",
        )

    df = pd.read_csv(metrics_path)
    if df.empty:
        return []

    return df.where(pd.notna(df), None).to_dict(orient="records")


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
    record = _load_latest_prediction_record(config)

    return {
        "status": "success",
        "prediction": record,
    }


@app.get("/metrics")
def get_metrics(include: list[str] | None = Query(default=None)) -> dict:
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
    metrics = _load_metrics(config)

    if include:
        metrics = {key: metrics.get(key) for key in include if key in metrics}

    return {
        "status": "success",
        "metrics": metrics,
    }


@app.get("/results")
def get_results(include_metrics: bool = Query(default=True)) -> dict:
    """
    Return the latest prediction and (optionally) training metrics in one call.

    Why this endpoint exists:
    - Dashboard clients often need both artifacts at the same time.
    - A combined endpoint reduces request overhead and simplifies polling.
    """
    config = _prepare_environment()

    payload = {
        "status": "success",
        "prediction": _load_latest_prediction_record(config),
    }

    if include_metrics:
        payload["metrics"] = _load_metrics(config)

    return payload


@app.get("/model-comparisons")
def get_model_comparisons() -> dict:
    """
    Return model-comparison metric tables used by the Streamlit comparison page.

    Why this endpoint exists:
    - Keeps comparison dashboards API-driven in docker/networked deployments.
    - Avoids duplicate CSV loading logic across frontend pages.
    """
    config = _prepare_environment()

    return {
        "status": "success",
        "tables": {
            "model_comparison": _load_metrics_table(config, "model_comparison.csv"),
            "ai_lift_comparison": _load_metrics_table(config, "ai_lift_comparison.csv"),
            "recent_period_model_comparison": _load_metrics_table(config, "recent_period_model_comparison.csv"),
            "recent_period_ai_lift_comparison": _load_metrics_table(config, "recent_period_ai_lift_comparison.csv"),
        },
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
