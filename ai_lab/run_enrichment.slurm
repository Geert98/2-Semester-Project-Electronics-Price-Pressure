#!/bin/bash
#SBATCH --job-name=news-enrichment
#SBATCH --output=logs/news-enrichment-%j.out
#SBATCH --error=logs/news-enrichment-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

# Adapt these lines to the model/runtime available on AAU AI-LAB.
# A common setup is to start an OpenAI-compatible vLLM server on the node,
# then let run_enrichment_job.py call http://127.0.0.1:8000/v1/chat/completions.
#
# Example placeholder:
# module load python/3.11 cuda
# python -m vllm.entrypoints.openai.api_server \
#   --model "$AI_LAB_MODEL" \
#   --host 127.0.0.1 \
#   --port 8000 &
# SERVER_PID=$!
# sleep 60

export AI_LAB_BASE_URL="${AI_LAB_BASE_URL:-http://127.0.0.1:8000/v1}"
export AI_LAB_API_KEY="${AI_LAB_API_KEY:-EMPTY}"
export AI_LAB_MODEL="${AI_LAB_MODEL:-CHANGE_ME}"

python run_enrichment_job.py \
  --input news_to_enrich.jsonl \
  --output news_enriched.jsonl \
  --model "$AI_LAB_MODEL" \
  --base-url "$AI_LAB_BASE_URL" \
  --api-key "$AI_LAB_API_KEY"

# If you started a local server above, uncomment:
# kill "$SERVER_PID"
