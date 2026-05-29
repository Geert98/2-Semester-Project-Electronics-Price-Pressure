#!/bin/bash
#SBATCH --job-name=news-enrichment
#SBATCH --output=logs/news-enrichment-%j.out
#SBATCH --error=logs/news-enrichment-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs

echo "=== News enrichment job started ==="
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(command -v python || true)"
python --version
echo "Files:"
ls -lah

if [[ ! -f "news_to_enrich.jsonl" ]]; then
  echo "ERROR: news_to_enrich.jsonl not found in $(pwd)"
  echo "Place news_to_enrich.jsonl in the same folder as run_enrichment.sh."
  exit 1
fi

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

if [[ "$AI_LAB_MODEL" == "CHANGE_ME" ]]; then
  echo "ERROR: AI_LAB_MODEL is still CHANGE_ME."
  echo "Set it before sbatch, for example:"
  echo "  export AI_LAB_MODEL=<model-available-on-ai-lab>"
  echo "  sbatch run_enrichment.sh"
  exit 1
fi

echo "AI_LAB_BASE_URL=$AI_LAB_BASE_URL"
echo "AI_LAB_MODEL=$AI_LAB_MODEL"
echo "Input lines: $(wc -l < news_to_enrich.jsonl)"

python run_enrichment_job.py \
  --input news_to_enrich.jsonl \
  --output news_enriched.jsonl \
  --model "$AI_LAB_MODEL" \
  --base-url "$AI_LAB_BASE_URL" \
  --api-key "$AI_LAB_API_KEY"

echo "Output lines: $(wc -l < news_enriched.jsonl 2>/dev/null || echo 0)"
echo "=== News enrichment job finished ==="

# If you started a local server above, uncomment:
# kill "$SERVER_PID"
