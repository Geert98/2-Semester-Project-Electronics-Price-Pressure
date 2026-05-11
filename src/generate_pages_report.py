from __future__ import annotations

# This script generates a static HTML dashboard for GitHub Pages.
#
# In the final project architecture, this file is used after the pipeline has run.
# It reads the latest saved artifacts and turns them into a simple static report.
#
# The script:
# 1. loads the latest prediction artifact
# 2. loads the saved training metrics artifact
# 3. creates a styled HTML dashboard
# 4. writes the output to docs/index.html
#
# This allows GitHub Pages to serve the latest model outputs as a lightweight
# public frontend without requiring a live Python backend.

import json
from html import escape
from pathlib import Path

import pandas as pd

from src.storage import load_dataframe_from_mongo
from src.utils import ensure_directories, load_config


def _safe_get(d: dict, key: str, default="N/A"):
    """Safely get a value from a dictionary."""
    return d.get(key, default) if isinstance(d, dict) else default


def _format_float(value, decimals: int = 3) -> str:
    """Format floats consistently for display in HTML."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_percent(value) -> str:
    """Format probabilities as percentages."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _dict_to_html_rows(data: dict) -> str:
    """Convert a simple dictionary to HTML rows."""
    if not data:
        return "<tr><td colspan='2'>No data available</td></tr>"

    rows = []
    for key, value in data.items():
        rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    return "\n".join(rows)


def _model_results_to_html_rows(model_results: dict) -> str:
    """Render model comparison metrics as HTML rows."""
    if not model_results:
        return "<tr><td colspan='3'>No model comparison available.</td></tr>"

    rows = []
    for model_name, result in model_results.items():
        rows.append(
            "<tr>"
            f"<td>{escape(str(model_name))}</td>"
            f"<td>{_format_float(result.get('accuracy'))}</td>"
            f"<td>{_format_float(result.get('macro_f1'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _confusion_matrix_html(confusion: list[list[int]]) -> str:
    """Render the confusion matrix as an HTML table."""
    if not confusion or len(confusion) != 3:
        return "<p>No confusion matrix available.</p>"

    labels = ["low", "medium", "high"]

    header = "".join(f"<th>{label}</th>" for label in labels)

    body_rows = []
    for row_label, row_values in zip(labels, confusion):
        cells = "".join(f"<td>{value}</td>" for value in row_values)
        body_rows.append(f"<tr><th>{row_label}</th>{cells}</tr>")

    return f"""
    <table class="matrix-table">
        <thead>
            <tr>
                <th>Actual \\ Predicted</th>
                {header}
            </tr>
        </thead>
        <tbody>
            {''.join(body_rows)}
        </tbody>
    </table>
    """


def _load_recent_articles(config: dict, limit: int = 20) -> list[dict]:
    """Load a compact recent-articles sample from MongoDB."""
    try:
        collection = config["storage"]["mongo"]["clean_news_collection"]
        df = load_dataframe_from_mongo(config, collection, sort_by="published_at")
    except Exception:
        return []

    if df.empty:
        return []

    columns = [
        col
        for col in ["published_at", "provider", "source", "title", "url", "language"]
        if col in df.columns
    ]
    article_df = df[columns].tail(limit).iloc[::-1].copy()

    if "published_at" in article_df.columns:
        article_df["published_at"] = pd.to_datetime(
            article_df["published_at"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    article_df = article_df.fillna("")
    return article_df.to_dict(orient="records")


def _articles_to_html_rows(articles: list[dict]) -> str:
    """Render recent articles as HTML table rows."""
    if not articles:
        return "<tr><td colspan='5'>No ingested articles available.</td></tr>"

    rows = []
    for article in articles:
        published = escape(str(article.get("published_at", "")))
        provider = escape(str(article.get("provider", "")))
        source = escape(str(article.get("source", "")))
        title = escape(str(article.get("title", "")))
        url = escape(str(article.get("url", "")))

        link = f"<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">{title}</a>" if url else title
        rows.append(
            "<tr>"
            f"<td>{published}</td>"
            f"<td>{provider}</td>"
            f"<td>{source}</td>"
            f"<td>{link}</td>"
            "</tr>"
        )

    return "\n".join(rows)


def _load_news_signal_trend(config: dict, limit: int = 18) -> list[dict]:
    """Load monthly news volume and sentiment from the processed model table."""
    model_table_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    if not model_table_path.exists():
        return []

    df = pd.read_csv(model_table_path)
    required_cols = {"month", "article_count", "avg_sentiment"}
    if df.empty or not required_cols.issubset(df.columns):
        return []

    trend_df = df[["month", "article_count", "avg_sentiment"]].tail(limit).copy()
    trend_df["month"] = pd.to_datetime(trend_df["month"], errors="coerce").dt.strftime("%Y-%m")
    trend_df["article_count"] = pd.to_numeric(
        trend_df["article_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    trend_df["avg_sentiment"] = pd.to_numeric(
        trend_df["avg_sentiment"],
        errors="coerce",
    ).fillna(0.0)

    return trend_df.to_dict(orient="records")


def _news_signal_to_html_rows(trend: list[dict]) -> str:
    """Render monthly news signal rows with compact in-table bars."""
    if not trend:
        return "<tr><td colspan='4'>No monthly news signal data available.</td></tr>"

    max_count = max(int(row.get("article_count", 0)) for row in trend) or 1
    rows = []
    for row in trend:
        month = escape(str(row.get("month", "")))
        article_count = int(row.get("article_count", 0))
        sentiment = float(row.get("avg_sentiment", 0.0))
        volume_width = min(100.0, (article_count / max_count) * 100)
        sentiment_width = min(100.0, abs(sentiment) * 100)
        sentiment_class = "sentiment-positive" if sentiment >= 0 else "sentiment-negative"

        rows.append(
            "<tr>"
            f"<td>{month}</td>"
            f"<td>{article_count}</td>"
            "<td>"
            "<div class=\"mini-bar-track\">"
            f"<div class=\"mini-bar-fill volume-fill\" style=\"width:{volume_width:.1f}%\"></div>"
            "</div>"
            "</td>"
            "<td>"
            f"<span class=\"sentiment-value\">{sentiment:.3f}</span>"
            "<div class=\"mini-bar-track sentiment-track\">"
            f"<div class=\"mini-bar-fill {sentiment_class}\" style=\"width:{sentiment_width:.1f}%\"></div>"
            "</div>"
            "</td>"
            "</tr>"
        )

    return "\n".join(rows)


def _load_external_indicator_trend(config: dict, limit: int = 18) -> list[dict]:
    """Load recent external structured indicator values from the model table."""
    model_table_path = Path(config["paths"]["processed_dir"]) / "model_table.csv"
    if not model_table_path.exists():
        return []

    df = pd.read_csv(model_table_path)
    indicator_cols = [
        col
        for col in ["gscpi", "wti_oil_price"]
        if col in df.columns
    ]
    if df.empty or not indicator_cols:
        return []

    trend_df = df[["month", *indicator_cols]].tail(limit).copy()
    trend_df["month"] = pd.to_datetime(trend_df["month"], errors="coerce").dt.strftime("%Y-%m")

    for col in indicator_cols:
        trend_df[col] = pd.to_numeric(trend_df[col], errors="coerce")

    trend_df = trend_df.fillna("")
    return trend_df.to_dict(orient="records")


def _external_indicators_to_html_rows(indicators: list[dict]) -> str:
    """Render recent external indicator values as HTML table rows."""
    if not indicators:
        return "<tr><td colspan='3'>No external indicator data available.</td></tr>"

    rows = []
    for row in indicators:
        month = escape(str(row.get("month", "")))
        gscpi = _format_float(row.get("gscpi"), decimals=2)
        wti = _format_float(row.get("wti_oil_price"), decimals=2)
        rows.append(
            "<tr>"
            f"<td>{month}</td>"
            f"<td>{gscpi}</td>"
            f"<td>{wti}</td>"
            "</tr>"
        )

    return "\n".join(rows)


def generate_pages_report(config_path: str = "configs/config.yaml") -> Path:
    """
    Generate a static HTML dashboard from saved pipeline artifacts.

    Why this function exists:
    - GitHub Pages can only serve static files.
    - The pipeline already saves its results as CSV and JSON artifacts.
    - This function transforms those artifacts into a human-readable webpage.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    config = load_config(config_path)

    # Ensure the standard project directories exist.
    ensure_directories(config["paths"])

    # Define artifact input paths.
    pred_path = Path(config["paths"]["predictions_dir"]) / "latest_prediction.csv"
    metrics_path = Path(config["paths"]["metrics_dir"]) / "train_metrics.json"

    # Define docs output folder for GitHub Pages.
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_path = docs_dir / "index.html"
    prediction_json_path = docs_dir / "latest_prediction.json"
    metrics_json_path = docs_dir / "train_metrics.json"
    articles_json_path = docs_dir / "news_articles.json"
    news_signal_json_path = docs_dir / "news_signal_trend.json"
    external_indicators_json_path = docs_dir / "external_indicators.json"

    # Load latest prediction artifact if it exists.
    prediction = {}
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if not pred_df.empty:
            prediction = pred_df.iloc[0].to_dict()

    # Load metrics artifact if it exists.
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    recent_articles = _load_recent_articles(config, limit=20)
    news_signal_trend = _load_news_signal_trend(config, limit=18)
    external_indicator_trend = _load_external_indicator_trend(config, limit=18)

    # Save copies of the key artifacts into docs/ as well.
    # This is useful for transparency and possible future frontend extensions.
    with open(prediction_json_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(articles_json_path, "w", encoding="utf-8") as f:
        json.dump(recent_articles, f, indent=2)

    with open(news_signal_json_path, "w", encoding="utf-8") as f:
        json.dump(news_signal_trend, f, indent=2)

    with open(external_indicators_json_path, "w", encoding="utf-8") as f:
        json.dump(external_indicator_trend, f, indent=2)

    # Extract prediction fields.
    pred_month = prediction.get("month", "N/A")
    pred_class = str(prediction.get("predicted_next_month_pressure", "N/A")).capitalize()
    proba_low = _format_percent(prediction.get("proba_low"))
    proba_medium = _format_percent(prediction.get("proba_medium"))
    proba_high = _format_percent(prediction.get("proba_high"))

    # Extract metric fields.
    accuracy = _format_float(metrics.get("accuracy"))
    macro_f1 = _format_float(metrics.get("macro_f1"))
    n_rows_train = _safe_get(metrics, "n_rows_train")
    n_rows_test = _safe_get(metrics, "n_rows_test")
    best_model = _safe_get(metrics, "best_model")

    train_dist_html = _dict_to_html_rows(metrics.get("train_class_distribution", {}))
    test_dist_html = _dict_to_html_rows(metrics.get("test_class_distribution", {}))
    pred_dist_html = _dict_to_html_rows(metrics.get("predicted_class_distribution", {}))
    model_results_html = _model_results_to_html_rows(metrics.get("model_results", {}))

    confusion_html = _confusion_matrix_html(metrics.get("confusion_matrix", []))
    articles_html = _articles_to_html_rows(recent_articles)
    news_signal_html = _news_signal_to_html_rows(news_signal_trend)
    external_indicators_html = _external_indicators_to_html_rows(external_indicator_trend)

    # Create the static HTML page.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Electronics Price Pressure Dashboard</title>
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

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}

        .container {{
            max-width: 1300px;
            margin: 0 auto;
            padding: 32px 20px 48px;
        }}

        .header {{
            margin-bottom: 24px;
        }}

        .header h1 {{
            margin: 0 0 8px;
            font-size: 2.6rem;
        }}

        .header p {{
            margin: 0;
            color: var(--muted);
            font-size: 1.05rem;
        }}

        .status {{
            margin-top: 20px;
            background: rgba(74, 222, 128, 0.15);
            border: 1px solid rgba(74, 222, 128, 0.35);
            color: #d1fae5;
            padding: 14px 16px;
            border-radius: 14px;
            font-weight: 600;
        }}

        .grid {{
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 24px;
            margin-top: 28px;
        }}

        .panel {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }}

        .panel h2 {{
            margin-top: 0;
            margin-bottom: 18px;
            font-size: 2rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 18px;
        }}

        .metric-card {{
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
        }}

        .metric-label {{
            font-size: 0.95rem;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .probabilities {{
            margin-top: 18px;
        }}

        .bar-row {{
            margin-bottom: 14px;
        }}

        .bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            margin-bottom: 6px;
            color: var(--muted);
        }}

        .bar-track {{
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 999px;
            height: 18px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-2), var(--accent));
            border-radius: 999px;
        }}

        .subsection-title {{
            margin: 22px 0 12px;
            font-size: 1.2rem;
            font-weight: 700;
        }}

        .three-col {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 10px;
        }}

        .table-card {{
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px;
        }}

        .table-card h3 {{
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 1rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}

        th {{
            color: var(--muted);
            font-weight: 600;
        }}

        a {{
            color: #93c5fd;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .wide-panel {{
            margin-top: 24px;
        }}

        .mini-bar-track {{
            width: 100%;
            min-width: 90px;
            height: 12px;
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 999px;
            overflow: hidden;
        }}

        .mini-bar-fill {{
            height: 100%;
            border-radius: 999px;
        }}

        .volume-fill {{
            background: var(--accent-2);
        }}

        .sentiment-positive {{
            background: var(--accent);
        }}

        .sentiment-negative {{
            background: var(--danger);
        }}

        .sentiment-value {{
            display: inline-block;
            min-width: 58px;
            margin-bottom: 5px;
            color: var(--muted);
            font-variant-numeric: tabular-nums;
        }}

        .matrix-table th,
        .matrix-table td {{
            text-align: center;
        }}

        .footer {{
            margin-top: 28px;
            color: var(--muted);
            font-size: 0.95rem;
        }}

        @media (max-width: 1000px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}

            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}

            .three-col {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 640px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 2rem;
            }}

            .metric-value {{
                font-size: 1.6rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Electronics Price Pressure Dashboard</h1>
            <p>Static report generated from the latest pipeline artifacts.</p>
            <div class="status">Latest report generated successfully from saved prediction and metrics artifacts.</div>
        </div>

        <div class="grid">
            <section class="panel">
                <h2>Latest Prediction</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Prediction Month</div>
                        <div class="metric-value">{pred_month}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Predicted Pressure</div>
                        <div class="metric-value">{pred_class}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Medium Probability</div>
                        <div class="metric-value">{proba_medium}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">High Probability</div>
                        <div class="metric-value">{proba_high}</div>
                    </div>
                </div>

                <div class="subsection-title">Prediction Probabilities</div>
                <div class="probabilities">
                    <div class="bar-row">
                        <div class="bar-label"><span>Low</span><span>{proba_low}</span></div>
                        <div class="bar-track"><div class="bar-fill" style="width:{prediction.get('proba_low', 0) * 100 if prediction else 0}%"></div></div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label"><span>Medium</span><span>{proba_medium}</span></div>
                        <div class="bar-track"><div class="bar-fill" style="width:{prediction.get('proba_medium', 0) * 100 if prediction else 0}%"></div></div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label"><span>High</span><span>{proba_high}</span></div>
                        <div class="bar-track"><div class="bar-fill" style="width:{prediction.get('proba_high', 0) * 100 if prediction else 0}%"></div></div>
                    </div>
                </div>
            </section>

            <section class="panel">
                <h2>Model Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{accuracy}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Macro F1</div>
                        <div class="metric-value">{macro_f1}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Train Rows</div>
                        <div class="metric-value">{n_rows_train}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Test Rows</div>
                        <div class="metric-value">{n_rows_test}</div>
                    </div>
                </div>

                <div class="subsection-title">Model Selection</div>
                <div class="table-card">
                    <h3>Selected Model: {best_model}</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Macro F1</th>
                            </tr>
                        </thead>
                        <tbody>
                            {model_results_html}
                        </tbody>
                    </table>
                </div>

                <div class="subsection-title">Class Distributions</div>
                <div class="three-col">
                    <div class="table-card">
                        <h3>Train Distribution</h3>
                        <table>
                            <tbody>
                                {train_dist_html}
                            </tbody>
                        </table>
                    </div>

                    <div class="table-card">
                        <h3>Test Distribution</h3>
                        <table>
                            <tbody>
                                {test_dist_html}
                            </tbody>
                        </table>
                    </div>

                    <div class="table-card">
                        <h3>Predicted Distribution</h3>
                        <table>
                            <tbody>
                                {pred_dist_html}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="subsection-title">Confusion Matrix</div>
                {confusion_html}
            </section>
        </div>

        <section class="panel wide-panel">
            <h2>Monthly News Signal</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Articles</th>
                        <th>Volume</th>
                        <th>Avg Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    {news_signal_html}
                </tbody>
            </table>
        </section>

        <section class="panel wide-panel">
            <h2>External Market Indicators</h2>
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>GSCPI</th>
                        <th>WTI Oil Price</th>
                    </tr>
                </thead>
                <tbody>
                    {external_indicators_html}
                </tbody>
            </table>
        </section>

        <section class="panel wide-panel">
            <h2>Recent Ingested Articles</h2>
            <table>
                <thead>
                    <tr>
                        <th>Published</th>
                        <th>Provider</th>
                        <th>Source</th>
                        <th>Article</th>
                    </tr>
                </thead>
                <tbody>
                    {articles_html}
                </tbody>
            </table>
        </section>

        <div class="footer">
            <p>
                This page is generated automatically from the latest saved pipeline artifacts.
                It is intended to be published through GitHub Pages as a lightweight frontend.
            </p>
        </div>
    </div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


if __name__ == "__main__":
    output_file = generate_pages_report()
    print(f"Static report generated: {output_file}")
