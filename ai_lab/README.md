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

On AI-LAB, first check which Ollama models are available:

```bash
which ollama
ollama list
```

Pick a model from `ollama list`. For a first test, prefer a smaller instruct
model if available, for example `llama3.1:8b`, `mistral:7b`,
`qwen2.5:7b-instruct`, or similar.

Then submit:

```bash
export AI_LAB_BACKEND=ollama
export AI_LAB_MODEL=<model-from-ollama-list>
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

If Ollama is not available, you can use an OpenAI-compatible model endpoint
instead:

```bash
export AI_LAB_BACKEND=openai-compatible
export AI_LAB_BASE_URL=http://127.0.0.1:8000/v1
export AI_LAB_MODEL=<model-name>
sbatch run_enrichment.sh
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
