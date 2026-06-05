from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.train import (
    _build_ai_lift_rows,
    _build_model,
    _build_preprocessor,
    _evaluate_model,
    _evaluate_naive_baselines,
    _select_feature_columns,
    _skipped_result,
)
from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


def _period_result_row(
    period_start_year: int,
    period_df: pd.DataFrame,
    validation_months: pd.Series,
    test_months: pd.Series,
    result: dict,
    validation_result: dict | None,
    n_rows_train: int,
    n_rows_validation: int,
    n_rows_test: int,
) -> dict:
    return {
        "period_start_year": period_start_year,
        "period_start_month": str(period_df["month"].iloc[0].date()),
        "period_end_month": str(period_df["month"].iloc[-1].date()),
        "validation_start_month": str(validation_months.iloc[0].date()),
        "validation_end_month": str(validation_months.iloc[-1].date()),
        "test_start_month": str(test_months.iloc[0].date()),
        "test_end_month": str(test_months.iloc[-1].date()),
        "n_rows_total": int(len(period_df)),
        "n_rows_train": int(n_rows_train),
        "n_rows_validation": int(n_rows_validation),
        "n_rows_test": int(n_rows_test),
        "result_id": f"{result.get('feature_set')}::{result.get('model_name')}",
        "feature_set": result.get("feature_set"),
        "model_name": result.get("model_name"),
        "status": result.get("status"),
        "feature_count": result.get("feature_count"),
        "validation_accuracy": (
            validation_result.get("accuracy") if validation_result else None
        ),
        "validation_balanced_accuracy": (
            validation_result.get("balanced_accuracy") if validation_result else None
        ),
        "validation_macro_f1": (
            validation_result.get("macro_f1") if validation_result else None
        ),
        "test_accuracy": result.get("accuracy"),
        "test_balanced_accuracy": result.get("balanced_accuracy"),
        "test_macro_f1": result.get("macro_f1"),
        "skip_reason": result.get("skip_reason"),
    }


def _skip_period_row(
    period_start_year: int,
    period_df: pd.DataFrame,
    reason: str,
) -> dict:
    if period_df.empty:
        period_start_month = None
        period_end_month = None
    else:
        period_start_month = str(period_df["month"].iloc[0].date())
        period_end_month = str(period_df["month"].iloc[-1].date())

    return {
        "period_start_year": period_start_year,
        "period_start_month": period_start_month,
        "period_end_month": period_end_month,
        "validation_start_month": None,
        "validation_end_month": None,
        "test_start_month": None,
        "test_end_month": None,
        "n_rows_total": int(len(period_df)),
        "n_rows_train": None,
        "n_rows_validation": None,
        "n_rows_test": None,
        "result_id": "period::skipped",
        "feature_set": None,
        "model_name": None,
        "status": "skipped",
        "feature_count": None,
        "validation_accuracy": None,
        "validation_balanced_accuracy": None,
        "validation_macro_f1": None,
        "test_accuracy": None,
        "test_balanced_accuracy": None,
        "test_macro_f1": None,
        "skip_reason": reason,
    }


def _evaluate_period(
    df: pd.DataFrame,
    period_start_year: int,
    feature_sets: list[str],
    model_candidates: list[str],
    random_state: int,
    validation_size_months: int,
    test_size_months: int,
    min_train_months: int,
) -> tuple[list[dict], dict]:
    period_start = pd.Timestamp(f"{period_start_year}-01-01")
    period_df = df[df["month"] >= period_start].reset_index(drop=True)

    if len(period_df) < min_train_months + validation_size_months + test_size_months:
        reason = (
            f"Not enough rows for split. rows={len(period_df)}, "
            f"required={min_train_months + validation_size_months + test_size_months}"
        )
        return [_skip_period_row(period_start_year, period_df, reason)], {}

    test_start_index = len(period_df) - test_size_months
    validation_start_index = test_start_index - validation_size_months

    y = period_df["target_class"]
    y_train = y.iloc[:validation_start_index]
    y_validation = y.iloc[validation_start_index:test_start_index]
    y_train_validation = y.iloc[:test_start_index]
    y_test = y.iloc[test_start_index:]
    validation_months = period_df["month"].iloc[validation_start_index:test_start_index]
    test_months = period_df["month"].iloc[test_start_index:]
    labels_order = ["low", "medium", "high"]

    rows: list[dict] = []
    validation_results: dict[str, dict] = {}
    test_results: dict[str, dict] = {}

    for feature_set in feature_sets:
        feature_cols = _select_feature_columns(period_df, feature_set)
        X = period_df[feature_cols]
        X_train = X.iloc[:validation_start_index]
        X_validation = X.iloc[validation_start_index:test_start_index]
        X_train_validation = X.iloc[:test_start_index]
        X_test = X.iloc[test_start_index:]

        for model_name in model_candidates:
            result_id = f"{feature_set}::{model_name}"
            try:
                validation_pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            _build_preprocessor(
                                feature_cols=feature_cols,
                                scale_numeric=(model_name == "logistic_regression"),
                            ),
                        ),
                        ("model", _build_model(model_name, random_state)),
                    ]
                )
                validation_pipeline.fit(X_train, y_train)
                validation_result = _evaluate_model(
                    model_name=model_name,
                    feature_set=feature_set,
                    pipeline=validation_pipeline,
                    X_test=X_validation,
                    y_test=y_validation,
                    labels_order=labels_order,
                )

                final_pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            _build_preprocessor(
                                feature_cols=feature_cols,
                                scale_numeric=(model_name == "logistic_regression"),
                            ),
                        ),
                        ("model", _build_model(model_name, random_state)),
                    ]
                )
                final_pipeline.fit(X_train_validation, y_train_validation)
                test_result = _evaluate_model(
                    model_name=model_name,
                    feature_set=feature_set,
                    pipeline=final_pipeline,
                    X_test=X_test,
                    y_test=y_test,
                    labels_order=labels_order,
                )
            except Exception as exc:
                validation_result = _skipped_result(
                    model_name=model_name,
                    feature_set=feature_set,
                    reason=str(exc),
                    feature_count=len(feature_cols),
                )
                test_result = validation_result
                logger.warning("%s %s skipped: %s", period_start_year, result_id, exc)

            validation_results[result_id] = validation_result
            test_results[result_id] = test_result
            rows.append(
                _period_result_row(
                    period_start_year=period_start_year,
                    period_df=period_df,
                    validation_months=validation_months,
                    test_months=test_months,
                    result=test_result,
                    validation_result=validation_result,
                    n_rows_train=len(y_train),
                    n_rows_validation=len(y_validation),
                    n_rows_test=len(y_test),
                )
            )

    for split_name, naive_results in {
        "validation": _evaluate_naive_baselines(
            y_reference=y_train,
            y_all=y,
            start_index=validation_start_index,
            end_index=test_start_index,
            labels_order=labels_order,
        ),
        "test": _evaluate_naive_baselines(
            y_reference=y_train_validation,
            y_all=y,
            start_index=test_start_index,
            end_index=len(period_df),
            labels_order=labels_order,
        ),
    }.items():
        for result_id, result in naive_results.items():
            rows.append(
                {
                    **_period_result_row(
                        period_start_year=period_start_year,
                        period_df=period_df,
                        validation_months=validation_months,
                        test_months=test_months,
                        result=result,
                        validation_result=result if split_name == "validation" else None,
                        n_rows_train=len(y_train),
                        n_rows_validation=len(y_validation),
                        n_rows_test=len(y_test),
                    ),
                    "evaluation_split": split_name,
                    "result_id": result_id,
                }
            )

    metadata = {
        "validation_results": validation_results,
        "test_results": test_results,
    }
    return rows, metadata


def evaluate_recent_period_models(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    input_path = processed_dir / "model_table.csv"

    df = pd.read_csv(input_path)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    experiment_config = config["model"].get("recent_period_experiment", {})
    start_years = experiment_config.get("start_years", [2020, 2021, 2022, 2023])
    validation_size_months = int(experiment_config.get("validation_size_months", 12))
    test_size_months = int(experiment_config.get("test_size_months", 12))
    min_train_months = int(experiment_config.get("min_train_months", 24))

    feature_sets = config["model"].get(
        "feature_sets",
        [
            "baseline",
            "ai_enhanced",
            "baseline_with_persistence",
            "ai_enhanced_with_persistence",
        ],
    )
    model_candidates = config["model"].get("candidates", ["logistic_regression"])
    random_state = int(config["model"].get("random_state", 42))

    all_rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for start_year in start_years:
        rows, period_metadata = _evaluate_period(
            df=df,
            period_start_year=int(start_year),
            feature_sets=feature_sets,
            model_candidates=model_candidates,
            random_state=random_state,
            validation_size_months=validation_size_months,
            test_size_months=test_size_months,
            min_train_months=min_train_months,
        )
        all_rows.extend(rows)
        metadata[str(start_year)] = period_metadata

    comparison_df = pd.DataFrame(all_rows)
    model_rows = comparison_df[
        comparison_df["feature_set"].notna()
        & (comparison_df["feature_set"] != "naive")
        & (comparison_df.get("evaluation_split", "test") != "validation")
    ].copy()

    lift_rows = []
    for start_year, period_rows in model_rows.groupby("period_start_year"):
        period_lift_rows = _build_ai_lift_rows(period_rows.to_dict("records"))
        for row in period_lift_rows:
            row["period_start_year"] = int(start_year)
        lift_rows.extend(period_lift_rows)

    lift_df = pd.DataFrame(lift_rows)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = metrics_dir / "recent_period_model_comparison.csv"
    lift_path = metrics_dir / "recent_period_ai_lift_comparison.csv"
    summary_path = metrics_dir / "recent_period_model_summary.json"

    comparison_df.to_csv(comparison_path, index=False)
    lift_df.to_csv(lift_path, index=False)

    summary = {
        "input_path": str(input_path),
        "comparison_path": str(comparison_path),
        "ai_lift_path": str(lift_path),
        "start_years": [int(x) for x in start_years],
        "validation_size_months": validation_size_months,
        "test_size_months": test_size_months,
        "min_train_months": min_train_months,
    }
    save_json(summary, str(summary_path))

    logger.info("Saved recent period comparison to %s", comparison_path)
    logger.info("Saved recent period AI lift comparison to %s", lift_path)
    return summary


if __name__ == "__main__":
    print(evaluate_recent_period_models())
