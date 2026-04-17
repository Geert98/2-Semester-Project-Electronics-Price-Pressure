# Electronics Price Pressure Monitoring Pipeline

An end-to-end MLOps-style pipeline for monitoring and predicting short-term **electronics price pressure** using:

- structured market indicator data from **FRED**
- unstructured news data from **GDELT**
- feature engineering across both sources
- a baseline classification model
- automated artifact generation
- scheduled execution with **GitHub Actions**
- a static frontend published through **GitHub Pages**
- containerized execution with **Docker**

The system predicts whether next-period electronics price pressure is:

- `low`
- `medium`
- `high`

The focus of the project is on **technical implementation, reproducibility, and operationalization** rather than perfect predictive performance.

---

## Live Frontend

A Github pages with a static report, made with Github actions can be found in the link below to show a visual presentation.
GitHub Pages dashboard:

**[Open the latest dashboard](https://geert98.github.io/Data-engineering-exam-April-2026/)**

---

## Repository Structure

```text
.
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
```	
---

## Main Components

Data ingestion:
• src/ingest_fred.py
• src/ingest_news.py

Preprocessing
	•	src/preprocess.py

Feature engineering
	•	src/feature_engineering.py

Model training
	•	src/train.py

Prediction
	•	src/predict.py

Static report generation
	•	src/generate_pages_report.py

Pipeline orchestration
	•	run_pipeline.py

API layer
	•	app/api.py

Prototype frontend
	•	app/streamlit_app.py

Automation
	•	.github/workflows/pipeline.yml

---

## Pipeline Overview

The full pipeline performs the following steps:
1.	Download the FRED producer price index series
2.	Download monthly news data from GDELT
3.	Clean and preprocess the news articles
4.	Aggregate article-level data to monthly features
5.	Build lagged and rolling structured features from FRED
6.	Construct the next-period target class
7.	Train a baseline Logistic Regression model
8.	Evaluate the model on a time-based holdout test set
9.	Generate the latest prediction artifact
10.	Generate a static HTML report for GitHub Pages

---

## Build, Run, and Reproduce

### Option A — Run locally with Python

#### 1. Clone the repository (local copy)

```bash
git clone PASTE_YOUR_REPOSITORY_URL_HERE
cd PASTE_YOUR_REPOSITORY_NAME_HERE
```

#### 2. Create a new or use exiting environment
Example:
```bash
conda create -n DataScience python=3.12 -y
conda activate DataScience
```

#### 3. Install dependencies 
```bash
pip install -r requirements.txt
```

#### 4. Run the full pipeline
```bash
python run_pipeline.py
```

This will:
ingest data --> preprocess data --> engineer features --> train the model --> generate the latest prediction --> generate the static report

#### 5. Open FastAPI locally
```bash
uvicorn app.api:app --reload
```

#### Then open in a browser window:
```bash
http://127.0.0.1:8000/docs
```

#### 6. Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

### Option B — Run with Docker

#### 1. Build Docker image
```bash
docker build -t electronics-price-pressure .
```

#### 2. Run FastAPI service in Docker
```bash
docker run -p 8000:8000 electronics-price-pressure
```

#### 3. Open in browser
```bash
http://127.0.0.1:8000/docs
```

#### 4. Run full pipeline in Docker
```bash
docker run --rm electronics-price-pressure python run_pipeline.py
```

---

## Github Actions
This repository includes a workflow in:
```bash
.github/workflows/pipeline.yml
```

This workflow is used for:
- Run the pipeline on a schedule or demand
- updating artifacts
- regenerate the static dashboard
- publish the result on Github Pages

---

### Author

Made by Anders Geert: **[Github profile](https://github.com/Geert98)**


