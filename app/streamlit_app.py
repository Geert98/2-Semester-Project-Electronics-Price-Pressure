from __future__ import annotations

# This file defines the Streamlit frontend for the project.
#
# In the overall system architecture, Streamlit acts as the user-facing layer.
# It does not run the machine learning logic directly. Instead, it communicates
# with the FastAPI backend, which serves the pipeline and saved artifacts.
#
# The dashboard currently supports:
# - checking whether the backend API is running
# - displaying the latest prediction
# - displaying model evaluation metrics
# - triggering the full pipeline from a button
# - displaying a summary of the latest pipeline run
#
# This makes the project easier to demonstrate and interact with than using
# API documentation alone.

import requests
import streamlit as st
import pandas as pd

# Base URL of the FastAPI backend.
# The Streamlit app expects the API to be running locally on port 8000.
API_BASE_URL = "http://127.0.0.1:8000"


# Configure the overall Streamlit page.
# layout="wide" gives more horizontal space for charts and metric boxes.
st.set_page_config(
    page_title="Electronics Price Pressure Dashboard",
    page_icon="📈",
    layout="wide",
)


def fetch_json(endpoint: str, method: str = "GET", timeout: int = 300) -> dict:
    """
    Send a request to the FastAPI backend and return the JSON response.

    Why this function exists:
    - Multiple UI components need to call backend endpoints.
    - Wrapping the request logic in one function avoids repetition.
    - It also makes the code easier to maintain if the backend URL or logic changes.

    Parameters
    ----------
    endpoint : str
        API endpoint path, for example "/health" or "/metrics".
    method : str
        HTTP method to use, typically "GET" or "POST".
    timeout : int
        Maximum time in seconds to wait for the request.

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    ValueError
        If an unsupported HTTP method is provided.
    requests.HTTPError
        If the API returns an unsuccessful status code.
    """
    url = f"{API_BASE_URL}{endpoint}"

    if method == "GET":
        response = requests.get(url, timeout=timeout)
    elif method == "POST":
        response = requests.post(url, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Raise an exception if the HTTP request failed.
    response.raise_for_status()
    return response.json()


def render_prediction(prediction: dict) -> None:
    """
    Render the latest prediction section of the dashboard.

    Why this function exists:
    - The latest prediction is one of the key outputs of the system.
    - This function presents the prediction label and probabilities in a
      more user-friendly format than raw JSON.

    Parameters
    ----------
    prediction : dict
        Prediction record returned by the backend.

    Returns
    -------
    None
    """
    st.subheader("Latest Prediction")

    # Read the main prediction fields from the API response.
    pressure = prediction.get("predicted_next_month_pressure", "unknown")
    month = prediction.get("month", "unknown")

    proba_low = float(prediction.get("proba_low", 0.0))
    proba_medium = float(prediction.get("proba_medium", 0.0))
    proba_high = float(prediction.get("proba_high", 0.0))

    # Display the key prediction values as summary metric cards.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Prediction Month", month)
    col2.metric("Predicted Pressure", pressure.capitalize())
    col3.metric("Medium Probability", f"{proba_medium:.2%}")
    col4.metric("High Probability", f"{proba_high:.2%}")

    # Prepare a small table for the probability chart.
    probs_df = pd.DataFrame(
        {
            "class": ["low", "medium", "high"],
            "probability": [proba_low, proba_medium, proba_high],
        }
    )

    st.write("Prediction probabilities")

    # Visualize the class probabilities as a simple bar chart.
    st.bar_chart(probs_df.set_index("class"))


def render_metrics(metrics: dict) -> None:
    """
    Render the model metrics section of the dashboard.

    Why this function exists:
    - A prediction alone is not enough for a useful demonstration.
    - The dashboard should also show how the model performed on the test set.
    - This function displays summary metrics, class distributions,
      and the confusion matrix.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary returned by the backend.

    Returns
    -------
    None
    """
    st.subheader("Model Metrics")

    # Extract the main summary metrics.
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    n_rows_train = metrics.get("n_rows_train")
    n_rows_test = metrics.get("n_rows_test")

    # Display the main model summary values in metric cards.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}" if accuracy is not None else "N/A")
    col2.metric("Macro F1", f"{macro_f1:.3f}" if macro_f1 is not None else "N/A")
    col3.metric("Train Rows", n_rows_train if n_rows_train is not None else "N/A")
    col4.metric("Test Rows", n_rows_test if n_rows_test is not None else "N/A")

    st.write("Class distributions")

    # Create three columns to show class balance in train, test, and predictions.
    dist_cols = st.columns(3)

    train_dist = metrics.get("train_class_distribution", {})
    test_dist = metrics.get("test_class_distribution", {})
    pred_dist = metrics.get("predicted_class_distribution", {})

    with dist_cols[0]:
        st.markdown("**Train distribution**")
        st.json(train_dist)

    with dist_cols[1]:
        st.markdown("**Test distribution**")
        st.json(test_dist)

    with dist_cols[2]:
        st.markdown("**Predicted distribution**")
        st.json(pred_dist)

    # Display the confusion matrix if it is available.
    confusion = metrics.get("confusion_matrix")
    if confusion:
        st.markdown("**Confusion matrix**")
        confusion_df = pd.DataFrame(
            confusion,
            index=["low", "medium", "high"],
            columns=["low", "medium", "high"],
        )
        st.dataframe(confusion_df, use_container_width=True)


def render_pipeline_summary(summary: dict) -> None:
    """
    Render a compact summary of the latest pipeline run.

    Why this function exists:
    - When the user triggers the pipeline from the frontend, it is useful
      to immediately see what happened.
    - This function displays row counts and the latest prediction returned
      by the pipeline execution endpoint.

    Parameters
    ----------
    summary : dict
        Response returned by POST /run-pipeline.

    Returns
    -------
    None
    """
    st.subheader("Latest Pipeline Run")

    artifacts = summary.get("artifacts", {})

    # Display a few key row counts from the latest run.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FRED Rows", artifacts.get("fred_rows", "N/A"))
    col2.metric("Raw News Rows", artifacts.get("raw_news_rows", "N/A"))
    col3.metric("Clean News Rows", artifacts.get("clean_news_rows", "N/A"))
    col4.metric("Model Table Rows", artifacts.get("model_table_rows", "N/A"))

    # Show the returned prediction artifact if available.
    latest_pred = artifacts.get("latest_prediction")
    if latest_pred:
        st.markdown("**Prediction returned by pipeline**")
        st.json(latest_pred)

    # Show a small training summary if available.
    metric_summary = artifacts.get("train_metrics_summary")
    if metric_summary:
        st.markdown("**Training summary**")
        st.json(metric_summary)


def render_news_articles(response: dict) -> None:
    """
    Render a compact table of recently ingested news articles.
    """
    st.subheader("Recent Ingested Articles")

    articles = response.get("articles", [])
    if not articles:
        st.info("No news articles found yet. Run the pipeline first.")
        return

    articles_df = pd.DataFrame(articles)
    visible_columns = [
        col
        for col in ["published_at", "provider", "source", "title", "url", "language"]
        if col in articles_df.columns
    ]
    articles_df = articles_df[visible_columns]

    st.dataframe(
        articles_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "published_at": st.column_config.TextColumn("Published"),
            "provider": st.column_config.TextColumn("Provider"),
            "source": st.column_config.TextColumn("Source"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "url": st.column_config.LinkColumn("Article"),
            "language": st.column_config.TextColumn("Lang"),
        },
    )


# Main page title and short description.
st.title("Electronics Price Pressure Dashboard")
st.write("Interactive dashboard for the electronics price pressure pipeline and API.")

# Sidebar controls for interacting with the backend.
with st.sidebar:
    st.header("Controls")
    st.caption("Make sure the FastAPI server is running on http://127.0.0.1:8000")

    # Button to trigger the full pipeline through the API.
    if st.button("Run Pipeline", use_container_width=True):
        with st.spinner("Running full pipeline... this can take a while."):
            try:
                pipeline_result = fetch_json("/run-pipeline", method="POST", timeout=1200)
                st.session_state["pipeline_result"] = pipeline_result
                st.success("Pipeline completed successfully.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    # Button to refresh the dashboard state.
    # The data is loaded again automatically during the next app rerun.
    if st.button("Refresh Prediction & Metrics", use_container_width=True):
        st.session_state["refresh_requested"] = True
        st.success("Data refreshed.")

# First verify that the backend API is reachable.
try:
    health_data = fetch_json("/health")
    if health_data.get("status") == "ok":
        st.success("API is running.")
    else:
        st.warning("API responded, but health status was unexpected.")
except Exception as exc:
    st.error(f"Could not connect to API: {exc}")
    st.stop()

# Create the main two-column dashboard layout.
left_col, right_col = st.columns([1.2, 1])

# Left column: latest prediction and class probabilities.
with left_col:
    try:
        prediction_response = fetch_json("/latest-prediction")
        render_prediction(prediction_response["prediction"])
    except Exception as exc:
        st.warning(f"Could not load latest prediction: {exc}")

# Right column: model evaluation metrics.
with right_col:
    try:
        metrics_response = fetch_json("/metrics")
        render_metrics(metrics_response["metrics"])
    except Exception as exc:
        st.warning(f"Could not load metrics: {exc}")

# If the pipeline was run from the dashboard during this session,
# display a summary of that pipeline execution below the main panels.
if "pipeline_result" in st.session_state:
    st.divider()
    render_pipeline_summary(st.session_state["pipeline_result"])

st.divider()
try:
    articles_response = fetch_json("/news-articles?limit=50&cleaned=true")
    render_news_articles(articles_response)
except Exception as exc:
    st.warning(f"Could not load ingested articles: {exc}")
