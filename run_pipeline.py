from __future__ import annotations

# This script is the main orchestration entry point for the full project pipeline.
#
# It connects all pipeline components into one end-to-end workflow.
#
# The script:
# 1. loads environment variables
# 2. sets up logging
# 3. loads the shared configuration
# 4. ensures required directories exist
# 5. refreshes structured/news data
# 6. runs preprocessing
# 7. builds features from imported AI-LAB labels
# 8. trains the model
# 9. generates the latest prediction
# 10. generates the static GitHub Pages report
#
# In practice, this file is the easiest way to execute the full pipeline locally.
# It is also useful as the baseline command behind automation, scheduling,
# or API-triggered execution.

from src.feature_engineering import build_feature_table
from src.generate_pages_report import generate_pages_report
from src.ingest_fred import ingest_fred
from src.ingest_news import ingest_news
from src.predict import predict_latest
from src.preprocess import preprocess_news
from src.train import train_model
from src.utils import ensure_directories, get_log_level, load_config, setup_env, setup_logging


def main() -> None:
    """
    Run the full pipeline from ingestion to prediction and report generation.

    Why this function exists:
    - The project consists of multiple modular scripts, each handling one stage
      of the MLOps pipeline.
    - This function stitches them together into one reproducible end-to-end run.
    - It provides a simple single command for local execution, testing,
      automation, and scheduled runs.

    Pipeline order
    --------------
    1. Load environment variables
    2. Configure logging
    3. Load config and create required directories
    4. Ingest structured FRED data
    5. Ingest unstructured news data
    6. Preprocess raw news
    7. Build the monthly feature table from imported AI-LAB labels
    8. Train the comparison models
    9. Generate the latest prediction
    10. Generate the static GitHub Pages report

    Returns
    -------
    None
    """
    # Load environment variables from .env so the project can access
    # external settings such as the configured log level.
    setup_env()

    # Set up consistent logging across the full pipeline run.
    setup_logging(get_log_level())

    # Load the shared YAML configuration and ensure all configured output
    # directories exist before any script tries to save files.
    config = load_config()
    ensure_directories(config["paths"])

    # Step 1: Ingest the structured external market data from FRED.
    ingest_fred()

    # Step 2: Ingest unstructured news data from the enabled providers.
    ingest_news()

    # Step 3: Clean and standardize the raw news data.
    preprocess_news()

    # AI enrichment is intentionally handled as a separate AI-LAB batch process.
    # Import those labels before this step with:
    #   python -m ai_lab.import_enrichment --input <upload_bundle>/results

    # Step 4: Combine processed news, FRED data, and imported AI labels.
    build_feature_table()

    # Step 5: Train model candidates and save model artifacts.
    train_model()

    # Step 6: Generate the latest available prediction using the trained model.
    prediction = predict_latest()

    # Step 7: Generate the static dashboard for GitHub Pages.
    report_path = generate_pages_report()

    # Print a short completion message and key output locations.
    print("\nPipeline completed successfully.\n")
    print(prediction.to_string(index=False))
    print(f"\nStatic report generated at: {report_path}\n")


if __name__ == "__main__":
    # Allow the full pipeline to be run directly from the command line.
    main()
