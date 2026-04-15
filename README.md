


# Repository Structure
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── artifacts/
│   ├── metrics/
│   │   └── train_metrics.json
│   ├── models/
│   │   ├── feature_columns.joblib
│   │   └── price_pressure_model.joblib
│   └── predictions/
│       └── latest_prediction.csv
├── configs/
│   └── config.yaml
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── index.html
│   ├── latest_prediction.json
│   └── train_metrics.json
├── src/
│   ├── feature_engineering.py
│   ├── generate_pages_report.py
│   ├── ingest_fred.py
│   ├── ingest_news.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── run_pipeline.py
└── README.md

	•	To reproduce the project, build and run it locally or with Docker.
	•	To inspect the latest frontend output, open the GitHub Pages dashboard.
	•	To inspect the automated execution, see the GitHub Actions workflow history.
