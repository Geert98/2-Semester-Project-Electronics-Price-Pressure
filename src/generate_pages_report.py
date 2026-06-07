from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from src.utils import ensure_directories, load_config

PREFERRED_FEATURE_ORDER = [
    "baseline",
    "baseline_with_persistence",
    "ai_compact",
    "ai_compact_with_persistence",
]
PREFERRED_MODEL_ORDER = ["logistic_regression", "xgboost"]


def _format_float(value: object, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_percent(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _humanize_token(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _dict_rows_html(data: dict[str, object]) -> str:
    if not data:
        return "<tr><td colspan='2'>No data available</td></tr>"
    rows = []
    for key, value in data.items():
        rows.append(f"<tr><td>{escape(_humanize_token(key))}</td><td>{escape(str(value))}</td></tr>")
    return "\n".join(rows)


def _confusion_matrix_html(confusion: list[list[int]], labels: list[str] | None = None) -> str:
    if not confusion:
        return "<p>No confusion matrix available.</p>"

    labels = labels or ["low", "medium", "high"]
    labels = [_humanize_token(label) for label in labels]
    header = "".join(f"<th>{escape(label)}</th>" for label in labels)

    body_rows = []
    for row_label, row_values in zip(labels, confusion):
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row_values)
        body_rows.append(f"<tr><th>{escape(row_label)}</th>{cells}</tr>")

    return (
        "<table class='matrix-table'><thead><tr><th>Actual / Predicted</th>"
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def _load_prediction(pred_path: Path) -> dict[str, object]:
    if not pred_path.exists():
        return {}
    df = pd.read_csv(pred_path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _load_train_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_table_signals(model_table_path: Path, limit: int = 18) -> tuple[list[dict], list[dict]]:
    if not model_table_path.exists():
        return [], []

    df = pd.read_csv(model_table_path)
    if df.empty:
        return [], []

    news_trend: list[dict] = []
    external_trend: list[dict] = []

    pressure_candidates = [
        "ai_relevance_weighted_pressure_direction_score",
        "ai_relevant_avg_pressure_direction_score",
        "avg_pressure_direction_score",
    ]
    pressure_col = next((col for col in pressure_candidates if col in df.columns), None)

    if "month" in df.columns and "article_count" in df.columns:
        news_cols = ["month", "article_count"]
        if pressure_col:
            news_cols.append(pressure_col)
        elif {"upward_pressure_share", "downward_pressure_share"}.issubset(df.columns):
            news_cols.extend(["upward_pressure_share", "downward_pressure_share"])

        news_df = df[news_cols].tail(limit).copy()
        news_df["month"] = pd.to_datetime(news_df["month"], errors="coerce").dt.strftime("%Y-%m")
        news_df["article_count"] = pd.to_numeric(news_df["article_count"], errors="coerce").fillna(0).astype(int)

        if pressure_col:
            news_df["news_pressure_signal"] = pd.to_numeric(news_df[pressure_col], errors="coerce").fillna(0.0)
        elif {"upward_pressure_share", "downward_pressure_share"}.issubset(news_df.columns):
            up = pd.to_numeric(news_df["upward_pressure_share"], errors="coerce").fillna(0.0)
            down = pd.to_numeric(news_df["downward_pressure_share"], errors="coerce").fillna(0.0)
            news_df["news_pressure_signal"] = up - down
        else:
            news_df["news_pressure_signal"] = 0.0

        keep_cols = ["month", "article_count", "news_pressure_signal"]
        news_df = news_df[keep_cols]
        news_trend = news_df.to_dict(orient="records")

    indicator_cols = [col for col in ["ppi_value", "gscpi", "wti_oil_price"] if col in df.columns]
    if indicator_cols and "month" in df.columns:
        ext_df = df[["month", *indicator_cols]].tail(limit).copy()
        ext_df["month"] = pd.to_datetime(ext_df["month"], errors="coerce").dt.strftime("%Y-%m")
        for col in indicator_cols:
            ext_df[col] = pd.to_numeric(ext_df[col], errors="coerce")
        external_trend = ext_df.fillna("").to_dict(orient="records")

    return news_trend, external_trend


def _load_comparison_tables(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = metrics_dir / "model_comparison.csv"
    lift_path = metrics_dir / "ai_lift_comparison.csv"

    model_df = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    lift_df = pd.read_csv(lift_path) if lift_path.exists() else pd.DataFrame()
    return model_df, lift_df


def _ordered_grouped_comparison(model_df: pd.DataFrame, metric_column: str = "test_macro_f1") -> pd.DataFrame:
    if model_df.empty:
        return pd.DataFrame()

    filtered = model_df[model_df["status"] == "ok"].copy()
    if filtered.empty:
        return pd.DataFrame()

    grouped_idx = filtered.groupby(["feature_set", "model_name"])[metric_column].idxmax()
    grouped = filtered.loc[grouped_idx].copy()

    grouped = grouped[grouped["feature_set"].isin(PREFERRED_FEATURE_ORDER)].copy()
    if grouped.empty:
        return pd.DataFrame()

    grouped["feature_order"] = grouped["feature_set"].apply(
        lambda feature: PREFERRED_FEATURE_ORDER.index(feature)
        if feature in PREFERRED_FEATURE_ORDER
        else len(PREFERRED_FEATURE_ORDER)
    )
    grouped["model_order"] = grouped["model_name"].apply(
        lambda model: PREFERRED_MODEL_ORDER.index(model)
        if model in PREFERRED_MODEL_ORDER
        else len(PREFERRED_MODEL_ORDER)
    )

    grouped = grouped.sort_values(["feature_order", "model_order", metric_column], ascending=[True, True, False])
    return grouped.drop(columns=["feature_order", "model_order"])


def _prediction_prob_rows(prediction: dict[str, object]) -> str:
    if not prediction:
        return "<p>No prediction probabilities available.</p>"

    probs = [
        ("Low", prediction.get("proba_low", 0.0)),
        ("Medium", prediction.get("proba_medium", 0.0)),
        ("High", prediction.get("proba_high", 0.0)),
    ]

    html_rows = []
    for label, value in probs:
        try:
            width = max(0.0, min(100.0, float(value) * 100.0))
        except (TypeError, ValueError):
            width = 0.0
        html_rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'><span>{label}</span><span>{_format_percent(value)}</span></div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
            "</div>"
        )
    return "".join(html_rows)


def _model_leaderboard_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='7'>No model comparison data available.</td></tr>"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(_humanize_token(row.get('feature_set', '')))}</td>"
            f"<td>{escape(_humanize_token(row.get('model_name', '')))}</td>"
            f"<td>{escape(str(int(float(row.get('feature_count', 0) or 0))))}</td>"
            f"<td>{_format_float(row.get('validation_accuracy'), 4)}</td>"
            f"<td>{_format_float(row.get('validation_macro_f1'), 4)}</td>"
            f"<td>{_format_float(row.get('test_accuracy'), 4)}</td>"
            f"<td>{_format_float(row.get('test_macro_f1'), 4)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _ai_lift_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='6'>No AI lift data available.</td></tr>"

    rows = []
    for _, row in df.iterrows():
        lift = row.get("test_macro_f1_lift", 0.0)
        lift_class = "lift-positive" if float(lift) >= 0 else "lift-negative"
        rows.append(
            "<tr>"
            f"<td>{escape(_humanize_token(row.get('model_name', '')))}</td>"
            f"<td>{escape(_humanize_token(row.get('comparison_type', '')))}</td>"
            f"<td>{escape(_humanize_token(row.get('baseline_feature_set', '')))}</td>"
            f"<td>{escape(_humanize_token(row.get('ai_feature_set', '')))}</td>"
            f"<td class='{lift_class}'>{_format_float(row.get('test_macro_f1_lift'), 4)}</td>"
            f"<td>{_format_float(row.get('test_accuracy_lift'), 4)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _news_signal_rows(news_signal: list[dict]) -> str:
    if not news_signal:
        return "<tr><td colspan='4'>No news signal data available.</td></tr>"
    max_count = max(int(item.get("article_count", 0)) for item in news_signal) or 1

    rows = []
    for item in news_signal:
        count = int(item.get("article_count", 0))
        pressure_signal = float(item.get("news_pressure_signal", 0.0))
        width = min(100.0, (count / max_count) * 100)
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('month', '')))}</td>"
            f"<td>{count}</td>"
            "<td><div class='mini-bar-track'><div class='mini-bar-fill volume-fill' "
            f"style='width:{width:.1f}%'></div></div></td>"
            f"<td>{_format_float(pressure_signal, 3)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _indicator_rows(indicators: list[dict]) -> str:
    if not indicators:
        return "<tr><td colspan='4'>No external indicator data available.</td></tr>"

    rows = []
    for item in indicators:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('month', '')))}</td>"
            f"<td>{_format_float(item.get('ppi_value'), 2)}</td>"
            f"<td>{_format_float(item.get('gscpi'), 2)}</td>"
            f"<td>{_format_float(item.get('wti_oil_price'), 2)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _base_layout(title: str, subtitle: str, nav: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>{escape(title)}</title>
    <style>
        :root {{
            --bg: #0b1020;
            --panel: #121a2b;
            --panel-2: #182338;
            --text: #f3f4f6;
            --muted: #b6c2d1;
            --accent: #4ade80;
            --accent-2: #60a5fa;
            --border: #263247;
            --danger: #f87171;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .container {{ max-width: 1300px; margin: 0 auto; padding: 30px 20px 48px; }}
        .header h1 {{ margin: 0 0 8px; font-size: 2.3rem; }}
        .header p {{ margin: 0; color: var(--muted); }}
        .nav {{ margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }}
        .nav a {{
            color: #dbeafe;
            background: #1f2a44;
            border: 1px solid var(--border);
            border-radius: 999px;
            text-decoration: none;
            padding: 8px 14px;
            font-size: 0.92rem;
        }}
        .status {{
            margin-top: 18px;
            background: rgba(74, 222, 128, 0.15);
            border: 1px solid rgba(74, 222, 128, 0.35);
            color: #d1fae5;
            padding: 12px 14px;
            border-radius: 12px;
            font-weight: 600;
        }}
        .grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 24px; margin-top: 24px; }}
        .panel {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            margin-top: 24px;
        }}
        .panel h2 {{ margin: 0 0 16px; font-size: 1.65rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
        .metric-card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }}
        .metric-label {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 7px; }}
        .metric-value {{ font-size: 1.7rem; font-weight: 700; }}
        .subsection-title {{ margin: 20px 0 10px; font-size: 1.1rem; font-weight: 700; }}
        .three-col {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        .table-card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 12px; padding: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
        th {{ color: var(--muted); font-weight: 600; }}
        .bar-row {{ margin-bottom: 12px; }}
        .bar-label {{ display: flex; justify-content: space-between; margin-bottom: 6px; color: var(--muted); }}
        .bar-track {{ background: #0f172a; border: 1px solid var(--border); border-radius: 999px; height: 18px; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent-2), var(--accent)); }}
        .matrix-table th, .matrix-table td {{ text-align: center; }}
        .mini-bar-track {{ height: 12px; background: #0f172a; border: 1px solid var(--border); border-radius: 999px; overflow: hidden; }}
        .mini-bar-fill {{ height: 100%; border-radius: 999px; }}
        .volume-fill {{ background: var(--accent-2); }}
        .lift-positive {{ color: #2E8B57; font-weight: 700; }}
        .lift-negative {{ color: #B22222; font-weight: 700; }}
        .footer {{ margin-top: 26px; color: var(--muted); font-size: 0.95rem; }}
        @media (max-width: 1000px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .three-col {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 640px) {{
            .metrics-grid {{ grid-template-columns: 1fr; }}
            .metric-value {{ font-size: 1.35rem; }}
        }}
    </style>
</head>
<body>
    <div class='container'>
        <div class='header'>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
            <div class='nav'>{nav}</div>
            <div class='status'>Static pages generated from pipeline artifacts.</div>
        </div>
        {body}
        <div class='footer'></div>
    </div>
</body>
</html>"""


def _render_main_page(
    prediction: dict[str, object],
    metrics: dict[str, object],
    news_signal: list[dict],
    indicators: list[dict],
) -> str:
    pred_month = prediction.get("month", "N/A")
    pred_class = _humanize_token(str(prediction.get("predicted_next_month_pressure", "N/A")))
    proba_low = _format_percent(prediction.get("proba_low"))
    proba_medium = _format_percent(prediction.get("proba_medium"))
    proba_high = _format_percent(prediction.get("proba_high"))

    metric_cards = {
        "Accuracy": _format_float(metrics.get("accuracy")),
        "Macro f1": _format_float(metrics.get("macro_f1")),
        "Train rows": metrics.get("n_rows_train", "N/A"),
        "Test rows": metrics.get("n_rows_test", "N/A"),
    }

    model_results_html = "<tr><td colspan='4'>No model results available.</td></tr>"
    model_results = metrics.get("model_results", {})
    if isinstance(model_results, dict) and model_results:
        model_rows = []
        for model_key, values in model_results.items():
            feature_set = ""
            model_name = str(model_key)
            if "::" in model_name:
                feature_set, model_name = model_name.split("::", maxsplit=1)
            feature_set = feature_set or "n/a"
            model_rows.append(
                "<tr>"
                f"<td>{escape(_humanize_token(feature_set))}</td>"
                f"<td>{escape(_humanize_token(model_name))}</td>"
                f"<td>{_format_float(values.get('accuracy'))}</td>"
                f"<td>{_format_float(values.get('macro_f1'))}</td>"
                "</tr>"
            )
        model_results_html = "\n".join(model_rows)

    train_dist_html = _dict_rows_html(metrics.get("train_class_distribution", {}))
    test_dist_html = _dict_rows_html(metrics.get("test_class_distribution", {}))
    pred_dist_html = _dict_rows_html(metrics.get("predicted_class_distribution", {}))
    confusion_html = _confusion_matrix_html(
        metrics.get("confusion_matrix", []),
        metrics.get("class_labels") if isinstance(metrics.get("class_labels"), list) else None,
    )

    nav = "".join(
        [
            "<a href='index.html'>Main dashboard</a>",
            "<a href='model-comparisons.html'>Model comparisons</a>",
        ]
    )

    body = f"""
    <div class='grid'>
        <section class='panel'>
            <h2>Latest prediction</h2>
            <div class='metrics-grid'>
                <div class='metric-card'><div class='metric-label'>Prediction month</div><div class='metric-value'>{escape(str(pred_month))}</div></div>
                <div class='metric-card'><div class='metric-label'>Predicted pressure</div><div class='metric-value'>{escape(str(pred_class))}</div></div>
                <div class='metric-card'><div class='metric-label'>Medium probability</div><div class='metric-value'>{proba_medium}</div></div>
                <div class='metric-card'><div class='metric-label'>High probability</div><div class='metric-value'>{proba_high}</div></div>
            </div>
            <div class='subsection-title'>Prediction probabilities</div>
            {_prediction_prob_rows(prediction)}

            <div class='subsection-title'>Class distributions</div>
            <div class='three-col'>
                <div class='table-card'><table><thead><tr><th colspan='2'>Train distribution</th></tr></thead><tbody>{train_dist_html}</tbody></table></div>
                <div class='table-card'><table><thead><tr><th colspan='2'>Test distribution</th></tr></thead><tbody>{test_dist_html}</tbody></table></div>
                <div class='table-card'><table><thead><tr><th colspan='2'>Predicted distribution</th></tr></thead><tbody>{pred_dist_html}</tbody></table></div>
            </div>

            <div class='subsection-title'>Confusion matrix</div>
            {confusion_html}

            <div class='subsection-title'>Source</div>
            <p style='color:#b6c2d1;margin:0;'>artifacts/predictions/latest_prediction.csv</p>
        </section>

        <section class='panel'>
            <h2>Model metrics</h2>
            <div class='metrics-grid'>
                <div class='metric-card'><div class='metric-label'>Accuracy</div><div class='metric-value'>{metric_cards['Accuracy']}</div></div>
                <div class='metric-card'><div class='metric-label'>Macro f1</div><div class='metric-value'>{metric_cards['Macro f1']}</div></div>
                <div class='metric-card'><div class='metric-label'>Train rows</div><div class='metric-value'>{metric_cards['Train rows']}</div></div>
                <div class='metric-card'><div class='metric-label'>Test rows</div><div class='metric-value'>{metric_cards['Test rows']}</div></div>
            </div>

            <div class='subsection-title'>Model selection</div>
            <div class='table-card'>
                <p style='margin:0 0 10px;color:#b6c2d1;'>Selected model: {escape(_humanize_token(str(metrics.get('best_model', 'N/A'))))}</p>
                <table><thead><tr><th>Feature set</th><th>Model name</th><th>Accuracy</th><th>Macro f1</th></tr></thead><tbody>{model_results_html}</tbody></table>
            </div>
        </section>
    </div>

    <section class='panel'>
        <h2>Monthly news signal</h2>
        <table><thead><tr><th>Month</th><th>Articles</th><th>Volume</th><th>Net pressure signal</th></tr></thead><tbody>{_news_signal_rows(news_signal)}</tbody></table>
        <p style='color:#b6c2d1;margin:10px 0 0;'>Signal uses pressure-direction features.</p>
        <p style='color:#b6c2d1;margin:6px 0 0;'>Guide: above 0.30 indicates upward pressure, between -0.30 and 0.30 indicates mixed signal, below -0.30 indicates downward pressure.</p>
    </section>

    <section class='panel'>
        <h2>External market indicators</h2>
        <table><thead><tr><th>Month</th><th>PPI value</th><th>GSCPI</th><th>WTI oil price</th></tr></thead><tbody>{_indicator_rows(indicators)}</tbody></table>
    </section>
    """

    return _base_layout(
        title="Electronics price pressure dashboard",
        subtitle="Static report generated from prediction and metrics artifacts.",
        nav=nav,
        body=body,
    )


def _render_model_comparison_page(model_df: pd.DataFrame, ai_lift_df: pd.DataFrame) -> str:
    ordered_grouped = _ordered_grouped_comparison(model_df)
    highest_score = pd.DataFrame()
    if not model_df.empty:
        highest_score = model_df[model_df["status"] == "ok"].copy().sort_values("test_macro_f1", ascending=False)

    positive_lift = int((ai_lift_df["test_macro_f1_lift"] > 0).sum()) if not ai_lift_df.empty else 0
    neutral_lift = int((ai_lift_df["test_macro_f1_lift"] == 0).sum()) if not ai_lift_df.empty else 0
    negative_lift = int((ai_lift_df["test_macro_f1_lift"] < 0).sum()) if not ai_lift_df.empty else 0

    nav = "".join(
        [
            "<a href='index.html'>Main dashboard</a>",
            "<a href='model-comparisons.html'>Model comparisons</a>",
        ]
    )

    body = f"""
    <section class='panel'>
        <h2>Grouped by feature + model</h2>
        <p style='color:#b6c2d1;margin-top:0;'>Order follows feature set and model pairing used in the Streamlit app.</p>
        <table>
            <thead>
                <tr>
                    <th>Feature set</th><th>Model name</th><th>Feature count</th>
                    <th>Validation accuracy</th><th>Validation macro f1</th><th>Test accuracy</th><th>Test macro f1</th>
                </tr>
            </thead>
            <tbody>{_model_leaderboard_rows(ordered_grouped)}</tbody>
        </table>
    </section>

    <section class='panel'>
        <h2>Highest score leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature set</th><th>Model name</th><th>Feature count</th>
                    <th>Validation accuracy</th><th>Validation macro f1</th><th>Test accuracy</th><th>Test macro f1</th>
                </tr>
            </thead>
            <tbody>{_model_leaderboard_rows(highest_score.head(20))}</tbody>
        </table>
    </section>

    <section class='panel'>
        <h2>AI lift summary</h2>
        <div class='metrics-grid'>
            <div class='metric-card'><div class='metric-label'>Positive AI test macro f1 lift</div><div class='metric-value'>{positive_lift}</div></div>
            <div class='metric-card'><div class='metric-label'>Neutral lift</div><div class='metric-value'>{neutral_lift}</div></div>
            <div class='metric-card'><div class='metric-label'>Negative lift</div><div class='metric-value'>{negative_lift}</div></div>
            <div class='metric-card'><div class='metric-label'>Rows</div><div class='metric-value'>{int(len(ai_lift_df)) if not ai_lift_df.empty else 0}</div></div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Model name</th><th>Comparison type</th><th>Baseline feature set</th><th>AI feature set</th>
                    <th>Test macro f1 lift</th><th>Test accuracy lift</th>
                </tr>
            </thead>
            <tbody>{_ai_lift_rows(ai_lift_df)}</tbody>
        </table>
    </section>
    """

    return _base_layout(
        title="Model comparisons",
        subtitle="Static comparison report generated from metrics artifacts.",
        nav=nav,
        body=body,
    )


def generate_pages_report(config_path: str = "configs/config.yaml") -> Path:
    config = load_config(config_path)
    ensure_directories(config["paths"])

    pred_path = Path(config["paths"]["predictions_dir"]) / "latest_prediction.csv"
    metrics_path = Path(config["paths"]["metrics_dir"]) / "train_metrics.json"
    model_table_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    metrics_dir = Path(config["paths"]["metrics_dir"])

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    index_html_path = docs_dir / "index.html"
    model_html_path = docs_dir / "model-comparisons.html"

    prediction = _load_prediction(pred_path)
    train_metrics = _load_train_metrics(metrics_path)
    news_signal, indicators = _load_model_table_signals(model_table_path, limit=18)
    model_comparison_df, ai_lift_df = _load_comparison_tables(metrics_dir)

    with open(docs_dir / "latest_prediction.json", "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)
    with open(docs_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)
    with open(docs_dir / "news_signal_trend.json", "w", encoding="utf-8") as f:
        json.dump(news_signal, f, indent=2)
    with open(docs_dir / "external_indicators.json", "w", encoding="utf-8") as f:
        json.dump(indicators, f, indent=2)
    with open(docs_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(model_comparison_df.where(pd.notna(model_comparison_df), None).to_dict(orient="records"), f, indent=2)
    with open(docs_dir / "ai_lift_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ai_lift_df.where(pd.notna(ai_lift_df), None).to_dict(orient="records"), f, indent=2)

    # Explicitly remove article-level export from the docs output scope.
    legacy_articles_json = docs_dir / "news_articles.json"
    if legacy_articles_json.exists():
        legacy_articles_json.unlink()

    index_html_path.write_text(
        _render_main_page(prediction, train_metrics, news_signal, indicators),
        encoding="utf-8",
    )
    model_html_path.write_text(
        _render_model_comparison_page(model_comparison_df, ai_lift_df),
        encoding="utf-8",
    )

    return index_html_path


if __name__ == "__main__":
    output_file = generate_pages_report()
    print(f"Static report generated: {output_file}")
