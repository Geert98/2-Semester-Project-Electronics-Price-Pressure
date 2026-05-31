from __future__ import annotations

# This script compares the trained model candidates using the saved evaluation
# metrics from the training step.

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils import get_log_level, load_config, save_json, setup_env, setup_logging

logger = logging.getLogger(__name__)


def compare_models(config_path: str = "configs/config.yaml") -> dict:
    """
    Compare the trained model candidates using the saved training metrics.
    """
    config = load_config(config_path)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_path = metrics_dir / "train_metrics.json"
    output_path = metrics_dir / "model_comparison.json"

    with open(metrics_path, "r", encoding="utf-8") as f:
        train_metrics = json.load(f)

    model_results = train_metrics.get("model_results", {})
    if len(model_results) < 2:
        raise ValueError(
            "Need at least two trained model results in train_metrics.json to compare models."
        )

    rows = []
    for model_name, result in model_results.items():
        rows.append(
            {
                "model_name": model_name,
                "accuracy": float(result.get("accuracy", 0.0)),
                "macro_f1": float(result.get("macro_f1", 0.0)),
                "predicted_class_distribution": result.get("predicted_class_distribution", {}),
                "classification_report": result.get("classification_report", {}),
                "confusion_matrix": result.get("confusion_matrix", []),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    summary = {
        "selection_metric": train_metrics.get("selection_metric", "macro_f1"),
        "best_model": train_metrics.get("best_model"),
        "comparison": comparison_df.to_dict(orient="records"),
    }

    save_json(summary, str(output_path))

    logger.info("Loaded comparison data from %s", metrics_path)
    logger.info("Saved model comparison to %s", output_path)
    print(comparison_df[["model_name", "accuracy", "macro_f1"]].to_string(index=False))

    return summary


if __name__ == "__main__":
    setup_env()
    setup_logging(get_log_level())
    compare_models()