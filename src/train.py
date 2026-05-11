from __future__ import annotations

# This script is responsible for training the baseline classification model.
#
# In the pipeline, this file takes the model-ready feature table produced during
# feature engineering and turns it into:
# - a trained model artifact
# - a saved feature list
# - an evaluation metrics artifact
#
# The script:
# 1. loads the monthly model table
# 2. performs a time-based train/test split
# 3. builds preprocessing + model pipelines
# 4. trains candidate classifiers
# 5. evaluates each model on the holdout test period
# 6. saves the best trained model, feature list, and metrics
#
# This step is essential for reproducibility because the model artifact and
# evaluation outputs are saved to disk and can be reused later by the API
# and frontend.

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def _build_preprocessor(feature_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    """
    Build a numeric preprocessing pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=steps),
                feature_cols,
            )
        ]
    )


def _build_model(model_name: str, random_state: int):
    """
    Create a classifier by configured model name.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"Unsupported model candidate: {model_name}")


def _evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels_order: list[str],
) -> dict:
    """
    Evaluate one trained model pipeline on the holdout data.
    """
    y_pred = pipeline.predict(X_test)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "predicted_class_distribution": pd.Series(y_pred).value_counts().to_dict(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels_order,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=labels_order,
        ).tolist(),
    }


def train_model(config_path: str = "configs/config.yaml") -> dict:
    """
    Train the baseline classification model using the monthly feature table.

    Why this function exists:
    - The project needs a reproducible model training step that can be rerun
      whenever new data is ingested.
    - The function trains a baseline classifier and stores the results as artifacts.
    - A time-based split is used instead of a random split because the project
      is forecasting future periods from past data.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics and training metadata.
    """
    # Load configuration so paths and model settings come from the shared config file.
    config = load_config(config_path)

    # Define the input model table and output artifact locations.
    input_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    model_dir = Path(config["paths"]["model_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])

    # Load the model-ready dataset.
    df = pd.read_csv(input_path)

    # Ensure the time column is parsed correctly and sorted chronologically.
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    # Read the number of months to keep as the test set.
    test_size_months = int(config["model"]["test_size_months"])

    # Select feature columns by excluding non-feature and target columns.
    feature_cols = [
        col for col in df.columns
        if col not in {
            "month",
            "date",
            "target_class",
            "target_pct_change_next_month",
        }
    ]

    # Split the dataset into features (X) and target labels (y).
    X = df[feature_cols]
    y = df["target_class"]

    # Ensure the dataset is large enough for the chosen train/test split.
    # This protects the training step from failing silently on extremely small datasets.
    if len(df) <= test_size_months + 6:
        raise ValueError(
            f"Dataset is too small for training/test split. "
            f"Rows={len(df)}, requested test_size_months={test_size_months}"
        )

    # Use a chronological split:
    # - earlier rows for training
    # - last N months for testing
    #
    # This is more realistic than a random split for time-dependent data.
    split_index = len(df) - test_size_months
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_months = df["month"].iloc[split_index:]

    # Log class balance in both train and test sets for transparency.
    logger.info("Train class distribution:\n%s", y_train.value_counts().to_string())
    logger.info("Test class distribution:\n%s", y_test.value_counts().to_string())

    # Define a fixed class order for evaluation outputs.
    # This makes the confusion matrix and report easier to interpret consistently.
    labels_order = ["low", "medium", "high"]

    random_state = int(config["model"]["random_state"])
    model_candidates = config["model"].get("candidates", ["logistic_regression"])
    selection_metric = config["model"].get("selection_metric", "macro_f1")

    model_results: dict[str, dict] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name in model_candidates:
        preprocessor = _build_preprocessor(
            feature_cols=feature_cols,
            scale_numeric=(model_name == "logistic_regression"),
        )
        model = _build_model(model_name, random_state=random_state)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        result = _evaluate_model(model_name, pipeline, X_test, y_test, labels_order)
        model_results[model_name] = result
        trained_pipelines[model_name] = pipeline

        logger.info(
            "%s | Accuracy: %.4f | Macro F1: %.4f",
            model_name,
            result["accuracy"],
            result["macro_f1"],
        )

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name].get(selection_metric, float("-inf")),
    )
    best_result = model_results[best_model_name]
    best_pipeline = trained_pipelines[best_model_name]

    # Collect evaluation metrics and metadata in one dictionary.
    metrics = {
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "accuracy": best_result["accuracy"],
        "macro_f1": best_result["macro_f1"],
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "predicted_class_distribution": best_result["predicted_class_distribution"],
        "labels_in_test": sorted(y_test.unique().tolist()),
        "classification_report": best_result["classification_report"],
        "confusion_matrix": best_result["confusion_matrix"],
        "model_results": model_results,
        "test_months": [str(x.date()) for x in test_months],
    }

    # Log the two main summary metrics so they are visible directly in the terminal.
    logger.info("Selected model: %s", best_model_name)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])

    # Ensure output directories exist before saving artifacts.
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Define artifact file paths.
    model_path = model_dir / "price_pressure_model.joblib"
    features_path = model_dir / "feature_columns.joblib"
    metrics_path = metrics_dir / "train_metrics.json"

    # Save the trained sklearn pipeline.
    # This includes both preprocessing and model logic.
    joblib.dump(best_pipeline, model_path)

    # Save the exact feature column list used during training.
    # This is needed later during prediction to ensure correct feature alignment.
    joblib.dump(feature_cols, features_path)

    # Save evaluation metrics and metadata as a JSON artifact.
    save_json(metrics, str(metrics_path))

    logger.info("Saved model to %s", model_path)
    logger.info("Saved feature columns to %s", features_path)
    logger.info("Saved metrics to %s", metrics_path)

    return metrics


if __name__ == "__main__":
    # Allow the script to be run directly for manual model training tests.
    train_model()
