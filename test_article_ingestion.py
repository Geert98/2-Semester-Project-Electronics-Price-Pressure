from __future__ import annotations



from src.feature_engineering import build_feature_table
from src.generate_pages_report import generate_pages_report
from src.ingest_fred import ingest_fred
from src.ingest_news import ingest_news
from src.predict import predict_latest
from src.preprocess import preprocess_news
from src.train import train_model
from src.utils import ensure_directories, get_log_level, load_config, setup_env, setup_logging

def main() -> None:
    """ only test the news ingestion component, using a small sample of articles """

    setup_env()
    #adding logging, as i dont think it will work otherwise, and autocomplete tells me it's needed for the ingest_news function
    setup_logging(get_log_level())
    config = load_config()

    ensure_directories(config["paths"])

    ingest_news()

    preprocess_news()
    print("Preprocessing complete. Check mongoDB atlass for the cleaned news collection to verify results.")
if __name__ == "__main__":
    main()
