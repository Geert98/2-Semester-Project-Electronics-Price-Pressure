# AI-LAB News Enrichment Batch

This folder contains a template for running article enrichment as an offline
batch job on AAU AI-LAB.

## Local export

From the project root on your local machine:

```bash
python -m src.export_news_for_ai_lab --limit 100 --max-chars 12000
```

This writes:

```text
artifacts/ai_lab/news_to_enrich.jsonl
```

Upload that file plus this `ai_lab/` folder to AI-LAB.

## AI-LAB run

On AI-LAB, adapt the model endpoint/model command in `run_enrichment.sh`,
then submit:

```bash
sbatch run_enrichment.sh
```

The job should produce:

```text
news_enriched.jsonl
```

Slurm output is written to:

```text
logs/news-enrichment-<jobid>.out
logs/news-enrichment-<jobid>.err
```

If the job appears to do nothing, inspect those files first:

```bash
squeue -u $USER
tail -n 100 logs/news-enrichment-*.out
tail -n 100 logs/news-enrichment-*.err
```

Each line must contain at least:

```json
{
  "url": "...",
  "title": "...",
  "is_relevant": true,
  "relevance_score": 0.9,
  "sentiment_score": -0.2,
  "price_pressure_direction": "upward",
  "price_pressure_strength": 4,
  "primary_topic": "shortage",
  "reason_short": "Article discusses chip shortages affecting electronics supply."
}
```

## Local import

Download `news_enriched.jsonl` to:

```text
artifacts/ai_lab/news_enriched.jsonl
```

Then import to MongoDB:

```bash
python -m src.import_ai_lab_enrichment
```

The imported records are stored in:

```text
test_enriched_news
```

`src/feature_engineering.py` then aggregates those article-level labels into
monthly `ai_*` model features.
