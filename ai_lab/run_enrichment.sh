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

export AI_LAB_BACKEND="${AI_LAB_BACKEND:-ollama}"
export AI_LAB_BASE_URL="${AI_LAB_BASE_URL:-http://127.0.0.1:11434}"
export AI_LAB_API_KEY="${AI_LAB_API_KEY:-EMPTY}"
export AI_LAB_MODEL="${AI_LAB_MODEL:-CHANGE_ME}"

if [[ "$AI_LAB_MODEL" == "CHANGE_ME" ]]; then
  echo "ERROR: AI_LAB_MODEL is still CHANGE_ME."
  echo "Set it before sbatch, for example:"
  echo "  export AI_LAB_MODEL=<model-available-on-ai-lab>"
  echo "  sbatch run_enrichment.sh"
  exit 1
fi

if [[ "$AI_LAB_BACKEND" == "ollama" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama command not found on this node."
    echo "Try running these on AI-LAB first:"
    echo "  module avail ollama"
    echo "  which ollama"
    echo "  ollama list"
    exit 1
  fi

  echo "Starting Ollama server..."
  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
  ollama serve > logs/ollama-${SLURM_JOB_ID:-manual}.log 2>&1 &
  SERVER_PID=$!
  sleep 10

  echo "Available Ollama models:"
  ollama list || true
fi

if [[ "$AI_LAB_BACKEND" == "openai-compatible" ]]; then
  echo "Using external/OpenAI-compatible endpoint. Make sure AI_LAB_BASE_URL is correct."
fi

echo "AI_LAB_BACKEND=$AI_LAB_BACKEND"
echo "AI_LAB_BASE_URL=$AI_LAB_BASE_URL"
echo "AI_LAB_MODEL=$AI_LAB_MODEL"
echo "Input lines: $(wc -l < news_to_enrich.jsonl)"

python run_enrichment_job.py \
  --input news_to_enrich.jsonl \
  --output news_enriched.jsonl \
  --backend "$AI_LAB_BACKEND" \
  --model "$AI_LAB_MODEL" \
  --base-url "$AI_LAB_BASE_URL" \
  --api-key "$AI_LAB_API_KEY"

echo "Output lines: $(wc -l < news_enriched.jsonl 2>/dev/null || echo 0)"
echo "=== News enrichment job finished ==="

if [[ "${SERVER_PID:-}" != "" ]]; then
  kill "$SERVER_PID" || true
fi
