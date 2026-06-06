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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "month",
    "date",
    "target_class",
    "target_class_lag_1",
    "target_pct_change_next_month",
}

BASELINE_FEATURE_PREFIXES = (
    "ppi_",
    "gscpi",
    "wti_oil_price",
)

BASELINE_FEATURE_COLUMNS = {
    "ppi_value",
}

PERSISTENCE_FEATURE_COLUMNS = {
    "target_class_lag_1_score",
    "target_class_lag_1_is_low",
    "target_class_lag_1_is_medium",
    "target_class_lag_1_is_high",
}

AI_ENHANCED_BASE_COLUMNS = {
    "ai_no_articles",
    "ai_articles_but_none_relevant",
    "ai_relevant_articles_available",
    "ai_enriched_article_count",
    "ai_relevant_article_count",
    "ai_relevant_share",
    "ai_high_relevance_share",
    "ai_avg_relevance_score",
    "ai_relevant_avg_sentiment",
    "ai_relevance_weighted_sentiment",
    "ai_relevant_avg_pressure_direction_score",
    "ai_relevance_weighted_pressure_direction_score",
    "ai_relevant_avg_pressure_strength",
    "ai_relevance_weighted_pressure_strength",
    "ai_relevant_upward_pressure_share",
    "ai_relevant_downward_pressure_share",
    "ai_relevant_neutral_pressure_share",
    "ai_topic_shortage_article_count",
    "ai_topic_tariff_article_count",
    "ai_topic_demand_article_count",
    "ai_topic_supply_chain_article_count",
    "ai_topic_export_controls_article_count",
    "ai_topic_oversupply_article_count",
}

AI_COMPACT_CURRENT_COLUMNS = {
    "ai_no_articles",
    "ai_articles_but_none_relevant",
    "ai_relevant_articles_available",
    "ai_enriched_article_count",
    "ai_relevant_article_count",
    "ai_relevant_share",
    "ai_high_relevance_share",
    "ai_relevance_weighted_sentiment",
    "ai_relevance_weighted_pressure_direction_score",
    "ai_relevance_weighted_pressure_strength",
    "ai_relevant_upward_pressure_share",
    "ai_relevant_downward_pressure_share",
}

AI_COMPACT_TIME_BASE_COLUMNS = {
    "ai_relevant_article_count",
    "ai_relevance_weighted_sentiment",
    "ai_relevance_weighted_pressure_direction_score",
    "ai_relevance_weighted_pressure_strength",
    "ai_relevant_upward_pressure_share",
}


class EncodedTargetClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrap estimators that require numeric target labels.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        y_encoded = self.estimator_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(y_encoded)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


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
            solver="lbfgs",
            C=0.5,
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ImportError(
                "xgboost is not installed. Install project requirements before using this candidate."
            ) from exc

        return EncodedTargetClassifier(
            XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
            )
        )

    raise ValueError(f"Unsupported model candidate: {model_name}")


def _candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return every numeric feature column that is not a target or identifier.
    """
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def _is_baseline_feature(col: str) -> bool:
    """
    Baseline features are structured economic and PPI time-series signals only.
    """
    return col in BASELINE_FEATURE_COLUMNS or col.startswith(BASELINE_FEATURE_PREFIXES)


def _is_ai_enhanced_feature(col: str) -> bool:
    """
    Curated AI features include relevance-filtered news signals and their lags.
    """
    for base_col in AI_ENHANCED_BASE_COLUMNS:
        if col == base_col:
            return True
        if col.startswith(f"{base_col}_lag_"):
            return True
        if col.startswith(f"{base_col}_roll_3_"):
            return True
    return False


def _is_ai_compact_feature(col: str) -> bool:
    """
    Compact AI features keep the strongest and most explainable news signals.
    """
    if col in AI_COMPACT_CURRENT_COLUMNS:
        return True

    for base_col in AI_COMPACT_TIME_BASE_COLUMNS:
        if col.startswith(f"{base_col}_lag_"):
            return True
        if col.startswith(f"{base_col}_roll_3_"):
            return True
    return False


def _is_persistence_feature(col: str) -> bool:
    """
    Persistence features summarize the latest known target class.
    """
    return col in PERSISTENCE_FEATURE_COLUMNS


def _select_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """
    Select the feature columns for a named experiment.
    """
    candidates = _candidate_feature_columns(df)

    if feature_set == "baseline":
        selected = [col for col in candidates if _is_baseline_feature(col)]
    elif feature_set == "baseline_with_persistence":
        selected = [
            col
            for col in candidates
            if _is_baseline_feature(col) or _is_persistence_feature(col)
        ]
    elif feature_set == "ai_compact":
        selected = [
            col
            for col in candidates
            if _is_baseline_feature(col) or _is_ai_compact_feature(col)
        ]
    elif feature_set == "ai_compact_with_persistence":
        selected = [
            col
            for col in candidates
            if (
                _is_baseline_feature(col)
                or _is_ai_compact_feature(col)
                or _is_persistence_feature(col)
            )
        ]
    elif feature_set == "ai_enhanced":
        selected = [
            col
            for col in candidates
            if _is_baseline_feature(col) or _is_ai_enhanced_feature(col)
        ]
    elif feature_set == "ai_enhanced_with_persistence":
        selected = [
            col
            for col in candidates
            if (
                _is_baseline_feature(col)
                or _is_ai_enhanced_feature(col)
                or _is_persistence_feature(col)
            )
        ]
    elif feature_set == "all_features":
        selected = candidates
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    if not selected:
        raise ValueError(f"No feature columns selected for feature set: {feature_set}")

    return selected


def _evaluate_model(
    model_name: str,
    feature_set: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels_order: list[str],
) -> dict:
    """
    Evaluate one trained model pipeline on the holdout data.
    """
    y_pred = pipeline.predict(X_test)
    return _evaluate_predictions(
        model_name=model_name,
        feature_set=feature_set,
        y_true=y_test,
        y_pred=pd.Series(y_pred, index=y_test.index),
        labels_order=labels_order,
        feature_count=int(X_test.shape[1]),
    )


def _evaluate_predictions(
    model_name: str,
    feature_set: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    labels_order: list[str],
    feature_count: int,
) -> dict:
    """
    Evaluate already-computed predictions.
    """
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "status": "ok",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "feature_count": int(feature_count),
        "predicted_class_distribution": pd.Series(y_pred).value_counts().to_dict(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels_order,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels_order,
        ).tolist(),
    }


def _skipped_result(model_name: str, feature_set: str, reason: str, feature_count: int) -> dict:
    """
    Return a metrics-shaped object for candidates that cannot be trained.
    """
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "status": "skipped",
        "skip_reason": reason,
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "feature_count": int(feature_count),
        "predicted_class_distribution": {},
        "classification_report": {},
        "confusion_matrix": [],
    }


def _evaluate_naive_baselines(
    y_reference: pd.Series,
    y_all: pd.Series,
    start_index: int,
    end_index: int,
    labels_order: list[str],
) -> dict[str, dict]:
    """
    Evaluate simple non-ML baselines for context.
    """
    y_true = y_all.iloc[start_index:end_index]
    majority_class = y_reference.value_counts().idxmax()

    majority_pred = pd.Series(
        [majority_class] * len(y_true),
        index=y_true.index,
    )
    previous_pred = y_all.shift(1).iloc[start_index:end_index].fillna(majority_class)

    return {
        "naive::train_majority": _evaluate_predictions(
            model_name="train_majority",
            feature_set="naive",
            y_true=y_true,
            y_pred=majority_pred,
            labels_order=labels_order,
            feature_count=0,
        ),
        "naive::previous_month_class": _evaluate_predictions(
            model_name="previous_month_class",
            feature_set="naive",
            y_true=y_true,
            y_pred=previous_pred,
            labels_order=labels_order,
            feature_count=0,
        ),
    }


def _build_model_comparison_rows(
    validation_model_results: dict[str, dict],
    test_model_results: dict[str, dict],
) -> list[dict]:
    """
    Build one flat row per trained candidate for easier reporting.
    """
    rows = []
    for result_id, test_result in test_model_results.items():
        validation_result = validation_model_results.get(result_id, {})
        rows.append(
            {
                "result_id": result_id,
                "feature_set": test_result.get("feature_set"),
                "model_name": test_result.get("model_name"),
                "status": test_result.get("status"),
                "feature_count": test_result.get("feature_count"),
                "validation_accuracy": validation_result.get("accuracy"),
                "validation_balanced_accuracy": validation_result.get("balanced_accuracy"),
                "validation_macro_f1": validation_result.get("macro_f1"),
                "test_accuracy": test_result.get("accuracy"),
                "test_balanced_accuracy": test_result.get("balanced_accuracy"),
                "test_macro_f1": test_result.get("macro_f1"),
                "skip_reason": test_result.get("skip_reason"),
            }
        )
    return rows


def _build_ai_lift_rows(model_comparison_rows: list[dict]) -> list[dict]:
    """
    Compare each model family with and without AI features.
    """
    by_feature_model = {
        (row["feature_set"], row["model_name"]): row
        for row in model_comparison_rows
        if row.get("status") == "ok"
    }
    model_names = sorted({row["model_name"] for row in model_comparison_rows})
    comparison_pairs = [
        ("baseline", "ai_compact", "compact_without_persistence"),
        (
            "baseline_with_persistence",
            "ai_compact_with_persistence",
            "compact_with_persistence",
        ),
        ("baseline", "ai_enhanced", "without_persistence"),
        (
            "baseline_with_persistence",
            "ai_enhanced_with_persistence",
            "with_persistence",
        ),
    ]

    rows = []
    for model_name in model_names:
        for baseline_feature_set, ai_feature_set, comparison_type in comparison_pairs:
            baseline_row = by_feature_model.get((baseline_feature_set, model_name))
            ai_row = by_feature_model.get((ai_feature_set, model_name))
            if not baseline_row or not ai_row:
                continue

            rows.append(
                {
                    "model_name": model_name,
                    "comparison_type": comparison_type,
                    "baseline_feature_set": baseline_feature_set,
                    "ai_feature_set": ai_feature_set,
                    "baseline_validation_macro_f1": baseline_row["validation_macro_f1"],
                    "ai_validation_macro_f1": ai_row["validation_macro_f1"],
                    "validation_macro_f1_lift": (
                        ai_row["validation_macro_f1"]
                        - baseline_row["validation_macro_f1"]
                    ),
                    "baseline_test_macro_f1": baseline_row["test_macro_f1"],
                    "ai_test_macro_f1": ai_row["test_macro_f1"],
                    "test_macro_f1_lift": (
                        ai_row["test_macro_f1"] - baseline_row["test_macro_f1"]
                    ),
                    "baseline_test_accuracy": baseline_row["test_accuracy"],
                    "ai_test_accuracy": ai_row["test_accuracy"],
                    "test_accuracy_lift": (
                        ai_row["test_accuracy"] - baseline_row["test_accuracy"]
                    ),
                }
            )
    return rows


def _build_feature_set_column_rows(feature_columns_by_result: dict[str, list[str]]) -> list[dict]:
    """
    Build a transparent feature-set-to-column mapping for reporting.

    The selected feature columns only depend on the feature set, not on the
    model candidate. Store each feature set once even though each result_id is
    feature_set::model_name.
    """
    rows = []
    seen_feature_sets = set()

    for result_id, feature_cols in feature_columns_by_result.items():
        feature_set = result_id.split("::", maxsplit=1)[0]
        if feature_set in seen_feature_sets:
            continue

        seen_feature_sets.add(feature_set)
        for feature_index, feature in enumerate(feature_cols, start=1):
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_index": feature_index,
                    "feature": feature,
                }
            )

    return rows


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

    # Read the number of months to keep as validation and final test sets.
    test_size_months = int(config["model"]["test_size_months"])
    validation_size_months = int(config["model"].get("validation_size_months", 24))

    # Split the dataset into target labels (y). Feature columns are selected per
    # experiment below so baseline and AI-enhanced runs can be compared fairly.
    y = df["target_class"]

    # Ensure the dataset is large enough for the chosen train/validation/test split.
    # This protects the training step from failing silently on extremely small datasets.
    if len(df) <= test_size_months + validation_size_months + 12:
        raise ValueError(
            f"Dataset is too small for training/validation/test split. "
            f"Rows={len(df)}, validation_size_months={validation_size_months}, "
            f"test_size_months={test_size_months}"
        )

    # Use a chronological split:
    # - earlier rows for training
    # - next N months for validation/model selection
    # - last N months for final testing
    #
    # This is more realistic than a random split for time-dependent data.
    test_start_index = len(df) - test_size_months
    validation_start_index = test_start_index - validation_size_months
    y_train = y.iloc[:validation_start_index]
    y_validation = y.iloc[validation_start_index:test_start_index]
    y_train_validation = y.iloc[:test_start_index]
    y_test = y.iloc[test_start_index:]
    validation_months = df["month"].iloc[validation_start_index:test_start_index]
    test_months = df["month"].iloc[test_start_index:]

    # Log class balance in train, validation, and test sets for transparency.
    logger.info("Train class distribution:\n%s", y_train.value_counts().to_string())
    logger.info("Validation class distribution:\n%s", y_validation.value_counts().to_string())
    logger.info("Test class distribution:\n%s", y_test.value_counts().to_string())

    # Define a fixed class order for evaluation outputs.
    # This makes the confusion matrix and report easier to interpret consistently.
    labels_order = ["low", "medium", "high"]

    random_state = int(config["model"]["random_state"])
    model_candidates = config["model"].get("candidates", ["logistic_regression"])
    feature_sets = config["model"].get("feature_sets", ["baseline", "ai_enhanced"])
    selection_metric = config["model"].get("selection_metric", "macro_f1")

    validation_model_results: dict[str, dict] = {}
    test_model_results: dict[str, dict] = {}
    trained_final_pipelines: dict[str, Pipeline] = {}
    feature_columns_by_result: dict[str, list[str]] = {}

    for feature_set in feature_sets:
        feature_cols = _select_feature_columns(df, feature_set)
        X = df[feature_cols]
        X_train = X.iloc[:validation_start_index]
        X_validation = X.iloc[validation_start_index:test_start_index]
        X_train_validation = X.iloc[:test_start_index]
        X_test = X.iloc[test_start_index:]

        for model_name in model_candidates:
            result_id = f"{feature_set}::{model_name}"
            feature_columns_by_result[result_id] = feature_cols
            try:
                preprocessor = _build_preprocessor(
                    feature_cols=feature_cols,
                    scale_numeric=(model_name == "logistic_regression"),
                )
                model = _build_model(model_name, random_state=random_state)
                validation_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )
                validation_pipeline.fit(X_train, y_train)
                validation_result = _evaluate_model(
                    model_name,
                    feature_set,
                    validation_pipeline,
                    X_validation,
                    y_validation,
                    labels_order,
                )

                final_preprocessor = _build_preprocessor(
                    feature_cols=feature_cols,
                    scale_numeric=(model_name == "logistic_regression"),
                )
                final_model = _build_model(model_name, random_state=random_state)
                final_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", final_preprocessor),
                        ("model", final_model),
                    ]
                )
                final_pipeline.fit(X_train_validation, y_train_validation)
                test_result = _evaluate_model(
                    model_name,
                    feature_set,
                    final_pipeline,
                    X_test,
                    y_test,
                    labels_order,
                )

                validation_model_results[result_id] = validation_result
                test_model_results[result_id] = test_result
                trained_final_pipelines[result_id] = final_pipeline

                logger.info(
                    "%s | validation %s: %.4f | test %s: %.4f",
                    result_id,
                    selection_metric,
                    validation_result[selection_metric],
                    selection_metric,
                    test_result[selection_metric],
                )
            except Exception as exc:
                skipped = _skipped_result(
                    model_name=model_name,
                    feature_set=feature_set,
                    reason=str(exc),
                    feature_count=len(feature_cols),
                )
                validation_model_results[result_id] = skipped
                test_model_results[result_id] = skipped
                logger.warning("%s skipped: %s", result_id, exc)

    validation_naive_results = _evaluate_naive_baselines(
        y_reference=y_train,
        y_all=y,
        start_index=validation_start_index,
        end_index=test_start_index,
        labels_order=labels_order,
    )
    test_naive_results = _evaluate_naive_baselines(
        y_reference=y_train_validation,
        y_all=y,
        start_index=test_start_index,
        end_index=len(df),
        labels_order=labels_order,
    )

    selectable_results = {
        result_id: result
        for result_id, result in validation_model_results.items()
        if result.get("status") == "ok" and result.get(selection_metric) is not None
    }
    if not selectable_results:
        raise RuntimeError("No model candidates trained successfully.")

    best_result_id = max(
        selectable_results,
        key=lambda name: selectable_results[name].get(selection_metric, float("-inf")),
    )
    best_validation_result = validation_model_results[best_result_id]
    best_test_result = test_model_results[best_result_id]
    best_pipeline = trained_final_pipelines[best_result_id]
    best_feature_cols = feature_columns_by_result[best_result_id]
    model_comparison_rows = _build_model_comparison_rows(
        validation_model_results=validation_model_results,
        test_model_results=test_model_results,
    )
    ai_lift_rows = _build_ai_lift_rows(model_comparison_rows)
    feature_set_column_rows = _build_feature_set_column_rows(feature_columns_by_result)

    # Collect evaluation metrics and metadata in one dictionary.
    metrics = {
        "best_model": best_test_result["model_name"],
        "best_feature_set": best_test_result["feature_set"],
        "best_result_id": best_result_id,
        "selection_metric": selection_metric,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(y_train)),
        "n_rows_validation": int(len(y_validation)),
        "n_rows_train_validation": int(len(y_train_validation)),
        "n_rows_test": int(len(y_test)),
        "feature_count": int(len(best_feature_cols)),
        "validation_accuracy": best_validation_result["accuracy"],
        "validation_balanced_accuracy": best_validation_result["balanced_accuracy"],
        "validation_macro_f1": best_validation_result["macro_f1"],
        "accuracy": best_test_result["accuracy"],
        "balanced_accuracy": best_test_result["balanced_accuracy"],
        "macro_f1": best_test_result["macro_f1"],
        "train_class_distribution": y_train.value_counts().to_dict(),
        "validation_class_distribution": y_validation.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "predicted_class_distribution": best_test_result["predicted_class_distribution"],
        "labels_in_test": sorted(y_test.unique().tolist()),
        "classification_report": best_test_result["classification_report"],
        "confusion_matrix": best_test_result["confusion_matrix"],
        "validation_model_results": validation_model_results,
        "validation_naive_results": validation_naive_results,
        "test_model_results": test_model_results,
        "test_naive_results": test_naive_results,
        "model_comparison": model_comparison_rows,
        "ai_lift_comparison": ai_lift_rows,
        "feature_set_columns": feature_set_column_rows,
        "model_results": test_model_results,
        "validation_months": [str(x.date()) for x in validation_months],
        "test_months": [str(x.date()) for x in test_months],
    }

    # Log the two main summary metrics so they are visible directly in the terminal.
    logger.info("Selected model: %s", best_result_id)
    logger.info("Validation Macro F1: %.4f", metrics["validation_macro_f1"])
    logger.info("Test Accuracy: %.4f", metrics["accuracy"])
    logger.info("Test Macro F1: %.4f", metrics["macro_f1"])

    # Ensure output directories exist before saving artifacts.
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Define artifact file paths.
    model_path = model_dir / "price_pressure_model.joblib"
    features_path = model_dir / "feature_columns.joblib"
    metrics_path = metrics_dir / "train_metrics.json"
    model_comparison_path = metrics_dir / "model_comparison.csv"
    ai_lift_path = metrics_dir / "ai_lift_comparison.csv"
    feature_set_columns_path = metrics_dir / "feature_set_columns.csv"

    # Save the trained sklearn pipeline.
    # This includes both preprocessing and model logic.
    joblib.dump(best_pipeline, model_path)

    # Save the exact feature column list used during training.
    # This is needed later during prediction to ensure correct feature alignment.
    joblib.dump(best_feature_cols, features_path)

    # Save evaluation metrics and metadata as a JSON artifact.
    save_json(metrics, str(metrics_path))
    pd.DataFrame(model_comparison_rows).to_csv(model_comparison_path, index=False)
    pd.DataFrame(ai_lift_rows).to_csv(ai_lift_path, index=False)
    pd.DataFrame(feature_set_column_rows).to_csv(feature_set_columns_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved feature columns to %s", features_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved model comparison to %s", model_comparison_path)
    logger.info("Saved AI lift comparison to %s", ai_lift_path)
    logger.info("Saved feature set columns to %s", feature_set_columns_path)

    return metrics


if __name__ == "__main__":
    # Allow the script to be run directly for manual model training tests.
    train_model()
