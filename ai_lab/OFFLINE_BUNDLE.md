# Offline AI-LAB Evidence Bundle

The finished AI-LAB run is kept outside Git because it contains article text,
model outputs, and runtime logs. Share it through an external download link
instead of committing it to the repository.

Recommended places to host it:

- AAU OneDrive or Teams files
- UCloud project storage
- a GitHub Release asset, if the article-text licensing is acceptable

Do not commit the bundle folder or archive to the normal repository history.

## Local Artifact

```text
ai_lab/upload_bundle_next_ready/
ai_lab/upload_bundle_next_ready.tar.gz
```

Current local sizes:

```text
upload_bundle_next_ready/       54M
upload_bundle_next_ready.tar.gz 16M
```

Archive checksum:

```text
SHA256 761e2d64c27e08e5dea8261ad7aafcd745f497c9188e8f4d7ec648c22481f700
```

External download URL:

```text
https://drive.google.com/drive/folders/1wk56Mwk6uxmATXxR29ff1GB3icUEj_PR?usp=sharing
```

## Inspect Without MongoDB

After downloading:

```bash
tar -xzf upload_bundle_next_ready.tar.gz
cd upload_bundle_next_ready
wc -l inputs/news_to_enrich_*.jsonl
wc -l results/news_enriched_*.jsonl
wc -l results/news_judged_*.jsonl
tail -n 50 logs/out/news-enrichment-*.out
```

Expected scale:

```text
4211 exported input articles
4206 enriched AI labels
4205 judge records
```

## Import Into MongoDB

Place the extracted folder under `ai_lab/` and run from the project root:

```bash
python -m ai_lab.import_enrichment --input ai_lab/upload_bundle_next_ready/results
python -m src.feature_engineering
python -m src.train
python -m src.predict
python -m src.generate_pages_report
```

If MongoDB is not available, use the committed processed data instead:

```bash
python -m src.train
python -m src.predict
python -m src.generate_pages_report
```
