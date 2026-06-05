from __future__ import annotations

# This script is responsible for generating the latest prediction from the
# trained model artifacts.
#
# In the pipeline, this file is used after model training and feature engineering.
# It loads:
# - the latest model-ready feature table
# - the saved trained model pipeline
# - the saved feature column list
#
# It then:
# 1. selects the most recent available row in the feature table
# 2. applies the trained model to that row
# 3. extracts class probabilities
# 4. saves the result as a prediction artifact
#
# This prediction artifact is later consumed by the FastAPI service and
# the Streamlit dashboard.

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.utils import load_config

logger = logging.getLogger(__name__)


def predict_latest(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Generate a prediction for the latest available feature row.

    Why this function exists:
    - The project needs a reusable prediction step that can run after training.
    - The prediction output should be stored as an artifact so it can be served
      by the API and displayed in the frontend.
    - The function uses the exact same feature columns and preprocessing pipeline
      that were used during training.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    pd.DataFrame
        A one-row DataFrame containing:
        - month
        - feature_month
        - predicted_next_month_pressure
        - proba_low
        - proba_medium
        - proba_high
    """
    # Load the shared project configuration.
    config = load_config(config_path)

    # Define paths for the model-ready feature table, trained model artifacts,
    # and output prediction artifact.
    processed_dir = Path(config["paths"]["processed_dir"])
    prediction_input_path = processed_dir / "prediction_table.csv"
    training_input_path = processed_dir / "model_table.csv"
    model_path = Path(config["paths"]["model_dir"]) / "price_pressure_model.joblib"
    features_path = Path(config["paths"]["model_dir"]) / "feature_columns.joblib"
    output_path = Path(config["paths"]["predictions_dir"]) / "latest_prediction.csv"

    # Load the scoring table produced during feature engineering. It keeps the
    # latest feature row even when the next-month target is not known yet.
    input_path = prediction_input_path if prediction_input_path.exists() else training_input_path
    df = pd.read_csv(input_path)

    # Parse the month column and sort chronologically to ensure the latest row
    # is truly the most recent available observation.
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    # Load the trained sklearn pipeline and the exact feature column list
    # used during training.
    pipeline = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    # Select the most recent row in the feature table.
    # This row represents the latest known state of the system and is used
    # to predict the next month's pressure class.
    latest_row = df.iloc[[-1]].copy()

    # Keep only the feature columns expected by the trained model.
    X_latest = latest_row[feature_cols]

    # Predict the most likely class label.
    predicted_class = pipeline.predict(X_latest)[0]

    # Predict class probabilities.
    predicted_proba = pipeline.predict_proba(X_latest)[0]

    # Read the class names directly from the trained classifier.
    class_names = list(pipeline.named_steps["model"].classes_)

    # Initialize a full probability dictionary for all expected classes.
    # This ensures the output always contains low, medium, and high,
    # even if one class is missing from the fitted classifier.
    proba_map = {cls: 0.0 for cls in ["low", "medium", "high"]}

    # Fill in the probabilities returned by the model.
    for cls, prob in zip(class_names, predicted_proba):
        proba_map[cls] = float(prob)

    feature_month = latest_row["month"].iloc[0]
    predicted_month = feature_month + pd.DateOffset(months=1)

    # Build the final one-row prediction DataFrame. The month field is the
    # predicted month; feature_month documents which source row was scored.
    result = pd.DataFrame(
        {
            "month": [predicted_month.date().isoformat()],
            "feature_month": [feature_month.date().isoformat()],
            "predicted_next_month_pressure": [predicted_class],
            "proba_low": [proba_map["low"]],
            "proba_medium": [proba_map["medium"]],
            "proba_high": [proba_map["high"]],
        }
    )

    # Ensure the predictions directory exists before saving the output artifact.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the latest prediction as a CSV artifact.
    result.to_csv(output_path, index=False)

    logger.info("Saved latest prediction to %s", output_path)
    return result


if __name__ == "__main__":
    # Allow the script to be run directly for manual prediction testing.
    print(predict_latest())
