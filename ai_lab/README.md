# AI-LAB News Enrichment Batch

This folder contains the minimal files needed to run article enrichment on AAU
AI-LAB with Slurm, Singularity, and vLLM.

## Prepare Upload Bundle

From the project root on your local machine:

```bash
python -m ai_lab.prepare_batch
```

This exports all pending relevant articles from MongoDB, splits them into
batches, and creates:

```text
ai_lab/upload_bundle/
```

The important files are:

```text
news_to_enrich.jsonl
inputs/news_to_enrich_00000.jsonl
inputs/news_to_enrich_00001.jsonl
batch_manifest.json
```

Upload the contents of that folder to AI-LAB.

## Run On AI-LAB

Inside the uploaded `upload_bundle` folder:

```bash
sbatch run_ai_lab.sh qwen25-32b
```

This runs all batch files in `inputs/` with one model load and writes one result
file per batch. By default, the same job then runs a separate Mistral judge step on
a quality-control sample of the enrichment labels.

Output lands in:

```text
logs/out/
logs/err/
results/news_enriched_00000.jsonl
results/failed_records_00000.jsonl
results/news_judged_00000.jsonl
results/failed_judgements_00000.jsonl
```

## Import Results

After downloading the finished `upload_bundle` folder back into `ai_lab/`, run:

```bash
python -m ai_lab.import_enrichment
```

This imports all `ai_lab/upload_bundle/results/news_enriched_*.jsonl` files
into MongoDB. For older single-file runs, it falls back to
`results/news_enriched.jsonl`.

## Models

Stable enrichment model:

```text
$HOME/models/Qwen2.5-32B-Instruct
```

This is downloaded from:

```text
Qwen/Qwen2.5-32B-Instruct
```

Judge model:

```text
$HOME/models/Mistral-Small-24B-Instruct-2501
```

This is downloaded from:

```text
mistralai/Mistral-Small-24B-Instruct-2501
```

Before the first run, create a private env file on AI-LAB:

```bash
nano ~/.ai_lab_env
chmod 600 ~/.ai_lab_env
```

Put this in the file:

```bash
HF_TOKEN=hf_your_token_here
```

Do not put this env file inside the project folder or upload bundle.

Then run:

```bash
sbatch run_ai_lab.sh qwen25-32b
```

You can also put optional defaults in the same file:

```bash
AI_LAB_QWEN25_32B_DIR=$HOME/models/Qwen2.5-32B-Instruct
AI_LAB_MISTRAL_JUDGE_DIR=$HOME/models/Mistral-Small-24B-Instruct-2501
AI_LAB_GPUS=4
AI_LAB_MAX_MODEL_LEN=4096
AI_LAB_JUDGE_SAMPLE_SIZE=500
```

To skip the judge step:

```bash
AI_LAB_RUN_JUDGE=false sbatch run_ai_lab.sh qwen25-32b
```

To run only the judge step on existing `results/news_enriched_*.jsonl` files:

```bash
sbatch run_ai_lab.sh judge-only
```

Check logs with:

```bash
tail -f logs/out/model-download-*.out
tail -f logs/err/model-download-*.err
tail -f logs/out/news-enrichment-*.out
tail -f logs/err/news-enrichment-*.err
```

If you only want to download the judge model without running enrichment:

```bash
sbatch run_ai_lab.sh judge-download
```

If it runs out of memory, try more GPUs if AI-LAB allows it:

```bash
AI_LAB_GPUS=8 sbatch --gres=gpu:8 run_ai_lab.sh qwen25-32b
```

To see all script commands:

```bash
./run_ai_lab.sh help
```

Ollama is not used by this bundle because `ollama` was not available on the
AI-LAB node.
