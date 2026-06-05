# Electronics Price Pressure Monitoring Pipeline

An end-to-end MLOps-style pipeline for monitoring and predicting short-term **electronics price pressure** using:

- structured market indicator data from **FRED** stored in **SQLite**
- unstructured news data from **The Guardian**, **Hacker News**, **NYT**, and
  optional Reuters/RapidAPI sources stored in **MongoDB**
- feature engineering across both sources
- Logistic Regression and XGBoost model comparisons, with and without AI news
  features
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
├── ai_lab/
│   ├── prepare_batch.py
│   ├── import_enrichment.py
│   ├── run_ai_lab.sh
│   └── enrich_articles.py
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
│   ├── db/
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
│   ├── storage.py
│   ├── train.py
│   └── utils.py
├── docker-compose.yml
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

Storage:
• src/storage.py

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

The project is split into two flows so the expensive LLM step stays explicit.

Core modeling flow:
1. Download the FRED producer price index series, WTI oil prices, and the New
   York Fed Global Supply Chain Pressure Index, then persist them in SQLite
2. Download or update news articles in MongoDB
3. Clean and preprocess the news articles
4. Import AI-LAB enrichment labels when a new AI batch has been run
5. Aggregate article-level AI labels and economic indicators to monthly features
6. Build lagged and rolling features, then construct the next-month target class
7. Train and compare Logistic Regression and XGBoost with and without AI features
8. Generate the latest prediction and static report

AI-LAB enrichment flow:
1. Export cleaned articles with `python -m ai_lab.prepare_batch`
2. Upload `ai_lab/upload_bundle/` to AI-LAB
3. Run `sbatch run_ai_lab.sh qwen25-32b`
4. Download the finished bundle
5. Import labels with `python -m ai_lab.import_enrichment --input <bundle>/results`

---
MongoDB is required for the news pipeline. The default local setup uses Docker Compose, while a production-like setup can point `MONGO_URI` to an external MongoDB service such as UCloud.
Raw news articles are upserted by URL in MongoDB, so repeated pipeline runs add new articles and update existing ones without deleting previously ingested Guardian or NewsAPI.org articles.
---

## Reproducibility Modes

### Fast Offline Review

Use this path when you do not have API keys, MongoDB Atlas access, or AI-LAB
access. The repository includes the latest processed model table, trained model,
metrics, prediction artifact, and static report.

```bash
python -m src.train
python -m src.predict
python -m src.generate_pages_report
```

This reproduces the model comparison from the committed processed data. It does
not fetch news, call APIs, connect to MongoDB, or run LLM enrichment.

The local folder `ai_lab/upload_bundle_next_ready/` can be kept as a complete
AI-LAB evidence bundle for examiners. It is intentionally ignored by Git because
its `inputs/` folder contains article text and runtime logs. If it is shared
separately, examiners can inspect `results/news_enriched_*.jsonl`,
`results/news_judged_*.jsonl`, and the AI-LAB logs without needing MongoDB.
See `ai_lab/OFFLINE_BUNDLE.md` for checksum and external sharing notes.

### Full Refresh

Use this path when you want to collect new data and rerun the full project:

```bash
python -m src.ingest_fred
python -m src.ingest_news
python -m src.preprocess
python -m ai_lab.prepare_batch
```

Then run the generated bundle on AI-LAB and import the result:

```bash
python -m ai_lab.import_enrichment --input ai_lab/upload_bundle/results
python -m src.feature_engineering
python -m src.train
python -m src.predict
python -m src.generate_pages_report
```

`run_pipeline.py` covers the local refresh after AI labels have been imported.
It does not run LLM enrichment itself.

## Build, Run, and Reproduce

### Option A — Run locally with Python

#### 1. Clone the repository (local copy)

```bash
git clone https://github.com/Geert98/Data-engineering-exam-April-2026/
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

#### 4. Configure news API keys
The default news providers are The Guardian Open Platform for historical backfill and NewsAPI.org for recent articles. GDELT and NewsData.io are configured as optional providers, but they are disabled by default: GDELT can be rate-limited during historical backfills, and NewsData historical archive access depends on the account plan. Add your API keys to the environment or to a local `.env` file:

```bash
GUARDIAN_API_KEY=your_guardian_open_platform_key
NEWSAPI_KEY=your_newsapi_org_key
NEWSDATA_API_KEY=your_newsdata_key
NYT_API_KEY=your_nyt_key
RAPIDAPI_REUTERS_KEY=your_rapidapi_reuters_key
```

The Guardian developer tier allows 1 call per second and 500 calls per day, so `configs/config.yaml` spaces Guardian requests by 1.2 seconds. The pipeline can request multiple Guardian pages per monthly window to increase article coverage while staying below that daily call limit. NewsAPI.org's free developer plan can search articles up to one month old with 100 requests per day, so this project uses it only as a recent-news source. NewsData.io free users are limited to 30 credits per 15 minutes, so NewsData requests are spaced by 31 seconds if that provider is enabled later.

The news `end_date` can be set to `today` in `configs/config.yaml`. In that case, each scheduled workflow run automatically extends the Guardian backfill through the current month.

For a historical Guardian backfill, run ingestion in date chunks. The broad query is useful when the downstream AI enrichment is responsible for filtering relevance:

```bash
python -m src.ingest_news --guardian-only --query-mode broad --start-date 1998-01-01 --end-date 2005-12-31 --guardian-max-pages 2
python -m src.ingest_news --guardian-only --query-mode broad --start-date 2006-01-01 --end-date 2013-12-31 --guardian-max-pages 2
python -m src.ingest_news --guardian-only --query-mode broad --start-date 2014-01-01 --end-date today --guardian-max-pages 2
python -m src.preprocess
```

Increase `--guardian-max-pages` for more coverage, but keep the Guardian daily API limit in mind. Raw news is upserted by URL, so rerunning a date chunk updates existing rows instead of duplicating them.

Optional providers can be refreshed separately when more coverage is needed.
They are not required for the fast offline review path.

```bash
python -m src.ingest_news --hackernews-only --query semiconductor --start-date 2022-01-01 --end-date today --max-records-per-window 100
python -m src.enrich_hackernews_links --limit 25
python -m src.ingest_news --nyt-only --query semiconductor --start-date 2022-01-01 --end-date today --max-records-per-window 10
python -m src.ingest_news --reuters-only --start-date 2022-01-01 --end-date today --max-records-per-window 20
python -m src.preprocess
```

Provider smoke tests are available through the individual provider modules if
you need to inspect raw API output before ingestion. Those tests write ignored
local files under `artifacts/hackernews/`, `artifacts/nyt/`, and
`artifacts/reuters/`.

If the Reuters RapidAPI host differs from the default, set it in `.env`:

```bash
RAPIDAPI_REUTERS_HOST=reuters-business-and-financial-news.p.rapidapi.com
```

#### 5. Optional: use UCloud MongoDB
For a production-like setup, point the pipeline at an external MongoDB instance instead of the local Docker database:

```bash
MONGO_URI=mongodb://username:password@host:port/?authSource=admin
MONGO_DB_NAME=electronics_price_pressure
```

Keep these values in `.env` locally and in GitHub Actions repository secrets. Do not commit database credentials.

#### 6. Run the local modeling pipeline
```bash
python run_pipeline.py
```

This will:
ingest data --> preprocess data --> engineer features from imported AI labels --> train the model --> generate the latest prediction --> generate the static report

AI-LAB enrichment is deliberately separate. Run `python -m ai_lab.prepare_batch`
and `python -m ai_lab.import_enrichment` when you need to refresh article labels.

#### 7. Open FastAPI locally
```bash
uvicorn app.api:app --reload
```

#### Then open in a browser window:
```bash
http://127.0.0.1:8000/docs
```

#### 8. Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

### Option B — Run with Docker Compose

#### 1. Start MongoDB and the API container
```bash
docker compose up --build
```

#### 2. Open the API docs
```bash
http://127.0.0.1:8000/docs
```

#### 3. Run the full pipeline inside the app container
```bash
docker compose exec app python run_pipeline.py
```

### Option C — Run with Docker only

```bash
docker build -t electronics-price-pressure .
docker run --rm -e MONGO_URI=mongodb://host.docker.internal:27017 electronics-price-pressure python run_pipeline.py
```

To run the API:
```bash
docker run -p 8000:8000 -e MONGO_URI=mongodb://host.docker.internal:27017 electronics-price-pressure
```

Then open:
```text
http://127.0.0.1:8000/docs
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

For UCloud MongoDB, add these repository secrets before running the workflow:

```text
MONGO_URI
MONGO_DB_NAME
GUARDIAN_API_KEY
```

---

### Author

Made by Anders Geert: **[Github profile](https://github.com/Geert98)**
