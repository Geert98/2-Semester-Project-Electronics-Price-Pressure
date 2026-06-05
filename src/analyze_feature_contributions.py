from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.train import _build_model, _build_preprocessor, _select_feature_columns
from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def _feature_group(feature_name: str) -> str:
    """
    Map a feature name into a compact explanatory group.
    """
    if feature_name.startswith("target_class_lag_1"):
        return "persistence"
    if feature_name.startswith("ppi"):
        return "ppi_history"
    if feature_name.startswith("gscpi"):
        return "supply_chain_index"
    if feature_name.startswith("wti_oil_price"):
        return "oil_price"

    if feature_name.startswith("ai_no_articles"):
        return "ai_coverage"
    if feature_name.startswith("ai_articles_but_none_relevant"):
        return "ai_coverage"
    if feature_name.startswith("ai_relevant_articles_available"):
        return "ai_coverage"
    if "sentiment" in feature_name:
        return "ai_sentiment"
    if "pressure_strength" in feature_name:
        return "ai_pressure_strength"
    if "pressure_direction" in feature_name:
        return "ai_pressure_direction"
    if "upward_pressure" in feature_name or "downward_pressure" in feature_name:
        return "ai_pressure_direction"
    if "article_count" in feature_name:
        return "ai_volume"
    if "relevant_share" in feature_name or "relevance" in feature_name:
        return "ai_relevance"
    if "topic_" in feature_name:
        return "ai_topic"
    if feature_name.startswith("ai_"):
        return "ai_other"
    return "other"


def _fit_candidate_pipeline(
    model_name: str,
    feature_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                _build_preprocessor(
                    feature_cols=feature_cols,
                    scale_numeric=(model_name == "logistic_regression"),
                ),
            ),
            ("model", _build_model(model_name, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _macro_f1_scorer(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Score with macro-F1 for string multiclass labels.
    """
    y_pred = estimator.predict(X)
    return float(f1_score(y, y_pred, average="macro"))


def analyze_feature_contributions(config_path: str = "configs/config.yaml") -> dict:
    """
    Calculate permutation-based feature contribution artifacts.
    """
    config = load_config(config_path)
    input_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    metrics_dir = Path(config["paths"]["metrics_dir"])

    df = pd.read_csv(input_path)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    test_size_months = int(config["model"]["test_size_months"])
    test_start_index = len(df) - test_size_months

    y = df["target_class"]
    y_train = y.iloc[:test_start_index]
    y_test = y.iloc[test_start_index:]
    test_months = df["month"].iloc[test_start_index:]

    feature_sets = config["model"].get("feature_sets", ["baseline", "ai_compact"])
    model_candidates = config["model"].get("candidates", ["logistic_regression"])
    random_state = int(config["model"].get("random_state", 42))

    importance_rows: list[dict] = []

    for feature_set in feature_sets:
        feature_cols = _select_feature_columns(df, feature_set)
        X = df[feature_cols]
        X_train = X.iloc[:test_start_index]
        X_test = X.iloc[test_start_index:]

        for model_name in model_candidates:
            result_id = f"{feature_set}::{model_name}"
            logger.info("Calculating permutation importance for %s", result_id)
            try:
                pipeline = _fit_candidate_pipeline(
                    model_name=model_name,
                    feature_cols=feature_cols,
                    X_train=X_train,
                    y_train=y_train,
                    random_state=random_state,
                )
                y_pred = pipeline.predict(X_test)
                baseline_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
                permutation = permutation_importance(
                    pipeline,
                    X_test,
                    y_test,
                    scoring=_macro_f1_scorer,
                    n_repeats=30,
                    random_state=random_state,
                    n_jobs=1,
                )
            except Exception as exc:
                logger.warning("%s skipped: %s", result_id, exc)
                continue

            for feature_name, importance_mean, importance_std in zip(
                feature_cols,
                permutation.importances_mean,
                permutation.importances_std,
                strict=True,
            ):
                importance_rows.append(
                    {
                        "result_id": result_id,
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "feature": feature_name,
                        "feature_group": _feature_group(feature_name),
                        "baseline_macro_f1": baseline_macro_f1,
                        "importance_mean": float(importance_mean),
                        "importance_std": float(importance_std),
                        "positive_importance": float(max(0.0, importance_mean)),
                    }
                )

    importance_df = pd.DataFrame(importance_rows)
    if importance_df.empty:
        raise RuntimeError("No feature contribution rows were produced.")

    group_df = (
        importance_df.groupby(["result_id", "feature_set", "model_name", "feature_group"], as_index=False)
        .agg(
            feature_count=("feature", "count"),
            baseline_macro_f1=("baseline_macro_f1", "first"),
            importance_mean_sum=("importance_mean", "sum"),
            positive_importance_sum=("positive_importance", "sum"),
            mean_importance=("importance_mean", "mean"),
        )
        .sort_values(["result_id", "positive_importance_sum"], ascending=[True, False])
    )

    top_features_df = importance_df.sort_values(
        ["result_id", "importance_mean"],
        ascending=[True, False],
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    importance_path = metrics_dir / "feature_permutation_importance.csv"
    group_path = metrics_dir / "feature_group_importance.csv"
    summary_path = metrics_dir / "feature_contribution_summary.json"

    top_features_df.to_csv(importance_path, index=False)
    group_df.to_csv(group_path, index=False)

    summary = {
        "input_path": str(input_path),
        "importance_path": str(importance_path),
        "group_importance_path": str(group_path),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(y_train)),
        "n_rows_test": int(len(y_test)),
        "test_start_month": str(test_months.iloc[0].date()),
        "test_end_month": str(test_months.iloc[-1].date()),
        "n_repeats": 30,
        "scoring": "macro_f1",
    }
    save_json(summary, str(summary_path))
    return summary


if __name__ == "__main__":
    print(analyze_feature_contributions())
