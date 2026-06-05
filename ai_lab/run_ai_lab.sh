#!/bin/bash
#SBATCH --job-name=news-enrichment
#SBATCH --output=logs/out/news-enrichment-%j.out
#SBATCH --error=logs/err/news-enrichment-%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:4

set -euo pipefail

MODE="${1:-run}"
REQUESTED_MODE="$MODE"
DOWNLOAD_MODEL_IF_MISSING=false
AI_LAB_ENV_FILE="${AI_LAB_ENV_FILE:-$HOME/.ai_lab_env}"

if [[ -f "$AI_LAB_ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$AI_LAB_ENV_FILE"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  ./run_ai_lab.sh                 Run enrichment with Qwen 2.5 32B, then Mistral judge
  ./run_ai_lab.sh qwen25-32b      Run enrichment with Qwen 2.5 32B, then Mistral judge
  ./run_ai_lab.sh judge-only      Run Mistral judge on existing enrichment results
  ./run_ai_lab.sh judge-download  Only download the Mistral judge model
  ./run_ai_lab.sh help            Show this help

You can run this script directly, or submit it with sbatch.
If you use sbatch directly and a downloaded model is missing, set HF_TOKEN
first, save it in ~/.ai_lab_env, or save it in ~/.cache/huggingface/token.

Advanced overrides:
  AI_LAB_MODEL=/path/or/hf-id ./run_ai_lab.sh
  AI_LAB_RUN_JUDGE=false ./run_ai_lab.sh qwen25-32b
  AI_LAB_GPUS=8 ./run_ai_lab.sh qwen25-32b
EOF
}

QWEN25_32B_REPO="${AI_LAB_QWEN25_32B_REPO:-Qwen/Qwen2.5-32B-Instruct}"
QWEN25_32B_DIR="${AI_LAB_QWEN25_32B_DIR:-$HOME/models/Qwen2.5-32B-Instruct}"
MISTRAL_JUDGE_REPO="${AI_LAB_MISTRAL_JUDGE_REPO:-mistralai/Mistral-Small-24B-Instruct-2501}"
MISTRAL_JUDGE_DIR="${AI_LAB_MISTRAL_JUDGE_DIR:-$HOME/models/Mistral-Small-24B-Instruct-2501}"
HF_TOKEN_FILE="${AI_LAB_HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
AI_LAB_USE_CONTAINER="${AI_LAB_USE_CONTAINER:-auto}"
AI_LAB_RUNTIME=""

model_ready() {
  [[ -f "$1/config.json" ]]
}

ensure_container_runtime() {
  if [[ ! -f "$AI_LAB_VLLM_CONTAINER" ]]; then
    echo "ERROR: vLLM container not found: $AI_LAB_VLLM_CONTAINER"
    echo "Check available containers with: ls -lah /ceph/container"
    exit 1
  fi

  if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: singularity command not found on this node."
    exit 1
  fi
}

select_runtime() {
  if [[ -n "$AI_LAB_RUNTIME" ]]; then
    return
  fi

  case "$AI_LAB_USE_CONTAINER" in
    false|False|0|no|No)
      AI_LAB_RUNTIME="host"
      ;;
    true|True|1|yes|Yes)
      ensure_container_runtime
      AI_LAB_RUNTIME="container"
      ;;
    auto)
      if getent passwd "$(id -u)" >/dev/null 2>&1 && command -v singularity >/dev/null 2>&1 && [[ -f "$AI_LAB_VLLM_CONTAINER" ]]; then
        AI_LAB_RUNTIME="container"
      else
        echo "Container user lookup/runtime is not available. Falling back to host python3 -s."
        echo "To force host mode, add AI_LAB_USE_CONTAINER=false to ~/.ai_lab_env."
        AI_LAB_RUNTIME="host"
      fi
      ;;
    *)
      echo "ERROR: Invalid AI_LAB_USE_CONTAINER=$AI_LAB_USE_CONTAINER"
      echo "Use auto, true, or false."
      exit 1
      ;;
  esac

  echo "Python runtime: $AI_LAB_RUNTIME"
}

run_python() {
  select_runtime
  if [[ "$AI_LAB_RUNTIME" == "container" ]]; then
    singularity exec --nv "$AI_LAB_VLLM_CONTAINER" env PYTHONNOUSERSITE=1 PYTHONPATH= "$@"
  else
    env PYTHONNOUSERSITE=1 PYTHONPATH= "$@"
  fi
}

download_model() {
  echo "=== Model download started ==="
  echo "Repository: $AI_LAB_DOWNLOAD_REPO"
  echo "Target folder: $AI_LAB_DOWNLOAD_DIR"

  if [[ -z "${HF_TOKEN:-}" && ! -f "$HF_TOKEN_FILE" ]]; then
    echo "WARNING: No HF_TOKEN env var or token file found at $HF_TOKEN_FILE."
    echo "Public models may still download, but gated models will fail."
  fi

  mkdir -p "$AI_LAB_DOWNLOAD_DIR"
  run_python \
    HF_TOKEN="${HF_TOKEN:-}" \
    AI_LAB_HF_TOKEN_FILE="$HF_TOKEN_FILE" \
    AI_LAB_DOWNLOAD_REPO="$AI_LAB_DOWNLOAD_REPO" \
    AI_LAB_DOWNLOAD_DIR="$AI_LAB_DOWNLOAD_DIR" \
    python3 -s -c "from huggingface_hub import snapshot_download; from pathlib import Path; import os; token = os.environ.get('HF_TOKEN') or None; token_file = Path(os.environ['AI_LAB_HF_TOKEN_FILE']); token = token or (token_file.read_text().strip() if token_file.exists() else None); snapshot_download(repo_id=os.environ['AI_LAB_DOWNLOAD_REPO'], local_dir=os.environ['AI_LAB_DOWNLOAD_DIR'], token=token)"

  echo "Downloaded model files:"
  find "$AI_LAB_DOWNLOAD_DIR" -maxdepth 1 -type f | sort | sed 's#^#  #'
  echo "=== Model download finished ==="
}

download_model_if_needed() {
  local repo="$1"
  local target_dir="$2"
  local label="$3"

  if model_ready "$target_dir"; then
    echo "$label model is already available: $target_dir"
    return
  fi

  echo "$label model is missing. Downloading before continuing."
  AI_LAB_DOWNLOAD_REPO="$repo" AI_LAB_DOWNLOAD_DIR="$target_dir" download_model
}

export AI_LAB_GPU_MEMORY_UTILIZATION="${AI_LAB_GPU_MEMORY_UTILIZATION:-0.85}"
export AI_LAB_VLLM_CONTAINER="${AI_LAB_VLLM_CONTAINER:-/ceph/container/vllm-openai_latest.sif}"
export AI_LAB_TIME="${AI_LAB_TIME:-02:00:00}"
export AI_LAB_MEM="${AI_LAB_MEM:-40G}"
export AI_LAB_DOWNLOAD_TIME="${AI_LAB_DOWNLOAD_TIME:-04:00:00}"
export AI_LAB_DOWNLOAD_MEM="${AI_LAB_DOWNLOAD_MEM:-16G}"
AI_LAB_TRUST_REMOTE_CODE="${AI_LAB_TRUST_REMOTE_CODE:-}"
export AI_LAB_JUDGE_MODEL="${AI_LAB_JUDGE_MODEL:-$MISTRAL_JUDGE_DIR}"
export AI_LAB_JUDGE_REPO="${AI_LAB_JUDGE_REPO:-$MISTRAL_JUDGE_REPO}"
export AI_LAB_JUDGE_DIR="${AI_LAB_JUDGE_DIR:-$MISTRAL_JUDGE_DIR}"
export AI_LAB_JUDGE_SAMPLE_SIZE="${AI_LAB_JUDGE_SAMPLE_SIZE:-500}"
export AI_LAB_JUDGE_MAX_MODEL_LEN="${AI_LAB_JUDGE_MAX_MODEL_LEN:-4096}"
export AI_LAB_JUDGE_MAX_TOKENS="${AI_LAB_JUDGE_MAX_TOKENS:-500}"
export AI_LAB_JUDGE_TRUST_REMOTE_CODE="${AI_LAB_JUDGE_TRUST_REMOTE_CODE:-false}"
export AI_LAB_JUDGE_TOKENIZER_MODE="${AI_LAB_JUDGE_TOKENIZER_MODE:-mistral}"
export AI_LAB_JUDGE_CONFIG_FORMAT="${AI_LAB_JUDGE_CONFIG_FORMAT:-mistral}"
export AI_LAB_JUDGE_LOAD_FORMAT="${AI_LAB_JUDGE_LOAD_FORMAT:-mistral}"

case "$MODE" in
  help|-h|--help)
    usage
    exit 0
    ;;
  run|qwen25-32b|large)
    export AI_LAB_MODEL="${AI_LAB_MODEL:-$QWEN25_32B_DIR}"
    export AI_LAB_GPUS="${AI_LAB_GPUS:-4}"
    export AI_LAB_MAX_MODEL_LEN="${AI_LAB_MAX_MODEL_LEN:-4096}"
    export AI_LAB_DOWNLOAD_REPO="${AI_LAB_DOWNLOAD_REPO:-$QWEN25_32B_REPO}"
    export AI_LAB_DOWNLOAD_DIR="${AI_LAB_DOWNLOAD_DIR:-$QWEN25_32B_DIR}"
    export AI_LAB_RUN_JUDGE="${AI_LAB_RUN_JUDGE:-true}"
    DOWNLOAD_MODEL_IF_MISSING=true
    MODE="run"
    ;;
  judge-download)
    export AI_LAB_DOWNLOAD_REPO="${AI_LAB_DOWNLOAD_REPO:-$AI_LAB_JUDGE_REPO}"
    export AI_LAB_DOWNLOAD_DIR="${AI_LAB_DOWNLOAD_DIR:-$AI_LAB_JUDGE_DIR}"
    MODE="download-model"
    ;;
  judge-only)
    export AI_LAB_RUN_JUDGE=true
    export AI_LAB_GPUS="${AI_LAB_GPUS:-4}"
    export AI_LAB_MAX_MODEL_LEN="${AI_LAB_MAX_MODEL_LEN:-4096}"
    MODE="judge-only"
    ;;
  download-model)
    export AI_LAB_DOWNLOAD_REPO="${AI_LAB_DOWNLOAD_REPO:-$QWEN25_32B_REPO}"
    export AI_LAB_DOWNLOAD_DIR="${AI_LAB_DOWNLOAD_DIR:-$QWEN25_32B_DIR}"
    ;;
  *)
    echo "ERROR: Unknown command: $MODE"
    usage
    exit 1
    ;;
esac

export AI_LAB_GPUS="${AI_LAB_GPUS:-1}"
export AI_LAB_MODEL="${AI_LAB_MODEL:-$QWEN25_32B_DIR}"
export AI_LAB_MAX_MODEL_LEN="${AI_LAB_MAX_MODEL_LEN:-8192}"
export AI_LAB_TRUST_REMOTE_CODE="${AI_LAB_TRUST_REMOTE_CODE:-false}"
export AI_LAB_TENSOR_PARALLEL_SIZE="${AI_LAB_TENSOR_PARALLEL_SIZE:-$AI_LAB_GPUS}"
export AI_LAB_JUDGE_TENSOR_PARALLEL_SIZE="${AI_LAB_JUDGE_TENSOR_PARALLEL_SIZE:-$AI_LAB_TENSOR_PARALLEL_SIZE}"
export AI_LAB_RUN_JUDGE="${AI_LAB_RUN_JUDGE:-false}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p logs/out logs/err results
  if [[ "$MODE" == "download-model" ]]; then
    if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
      read -rsp "Hugging Face token: " HF_TOKEN
      echo
      export HF_TOKEN
    fi

    echo "Submitting AI-LAB model download job"
    echo "Repository: $AI_LAB_DOWNLOAD_REPO"
    echo "Target folder: $AI_LAB_DOWNLOAD_DIR"
    sbatch \
      --job-name=model-download \
      --output=logs/out/model-download-%j.out \
      --error=logs/err/model-download-%j.err \
      --time="$AI_LAB_DOWNLOAD_TIME" \
      --mem="$AI_LAB_DOWNLOAD_MEM" \
      --export=ALL \
      "$0" download-model
    exit 0
  fi

  if [[ "$MODE" == "judge-only" ]]; then
    echo "Submitting AI-LAB judge job"
  else
    echo "Submitting AI-LAB enrichment job"
  fi
  echo "Model: $AI_LAB_MODEL"
  echo "GPUs: $AI_LAB_GPUS"
  echo "Max model length: $AI_LAB_MAX_MODEL_LEN"

  DEPENDENCY_ARGS=()
  if [[ "$MODE" != "judge-only" && "$DOWNLOAD_MODEL_IF_MISSING" == true && "$AI_LAB_MODEL" == "$AI_LAB_DOWNLOAD_DIR" ]] && ! model_ready "$AI_LAB_MODEL"; then
    if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
      read -rsp "Hugging Face token: " HF_TOKEN
      echo
      export HF_TOKEN
    fi

    echo "Local model is not downloaded yet. Submitting download job first."
    echo "Repository: $AI_LAB_DOWNLOAD_REPO"
    echo "Target folder: $AI_LAB_DOWNLOAD_DIR"
    DOWNLOAD_JOB_ID="$(sbatch --parsable \
      --job-name=model-download \
      --output=logs/out/model-download-%j.out \
      --error=logs/err/model-download-%j.err \
      --time="$AI_LAB_DOWNLOAD_TIME" \
      --mem="$AI_LAB_DOWNLOAD_MEM" \
      --export=ALL \
      "$0" download-model)"
    echo "Download job id: $DOWNLOAD_JOB_ID"
    DEPENDENCY_ARGS=(--dependency="afterok:$DOWNLOAD_JOB_ID")
  fi

  sbatch \
    --job-name=news-enrichment \
    --output=logs/out/news-enrichment-%j.out \
    --error=logs/err/news-enrichment-%j.err \
    --time="$AI_LAB_TIME" \
    --mem="$AI_LAB_MEM" \
    --gres="gpu:$AI_LAB_GPUS" \
    --export=ALL \
    "${DEPENDENCY_ARGS[@]}" \
    "$0" "$REQUESTED_MODE"
  exit 0
fi

if [[ "$MODE" == "download-model" ]]; then
  echo "=== Model download job started ==="
  echo "Host: $(hostname)"
  download_model
  echo "=== Model download job finished ==="
  exit 0
fi

echo "=== News enrichment job started ==="
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Model: $AI_LAB_MODEL"
echo "GPUs requested: $AI_LAB_GPUS"
echo "Tensor parallel size: $AI_LAB_TENSOR_PARALLEL_SIZE"
echo "Max model length: $AI_LAB_MAX_MODEL_LEN"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container: $AI_LAB_VLLM_CONTAINER"

INPUT_DIR="${AI_LAB_INPUT_DIR:-inputs}"
RESULTS_DIR="${AI_LAB_RESULTS_DIR:-results}"
mkdir -p logs/out logs/err "$RESULTS_DIR"

INPUT_FILES=()
if [[ -d "$INPUT_DIR" ]]; then
  mapfile -t INPUT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -name 'news_to_enrich_*.jsonl' | sort)
fi

if [[ "${#INPUT_FILES[@]}" -eq 0 && ! -f "news_to_enrich.jsonl" ]]; then
  echo "ERROR: No input files found in $(pwd)"
  echo "Expected either:"
  echo "  $INPUT_DIR/news_to_enrich_*.jsonl"
  echo "or:"
  echo "  news_to_enrich.jsonl"
  echo "Run this script from the upload_bundle folder."
  exit 1
fi

run_judge_step() {
  echo "=== News enrichment judge step started ==="
  echo "Judge model: $AI_LAB_JUDGE_MODEL"
  echo "Judge sample size: $AI_LAB_JUDGE_SAMPLE_SIZE"
  echo "Judge max model length: $AI_LAB_JUDGE_MAX_MODEL_LEN"
  echo "Judge tokenizer mode: ${AI_LAB_JUDGE_TOKENIZER_MODE:-auto}"
  echo "Judge config format: ${AI_LAB_JUDGE_CONFIG_FORMAT:-auto}"
  echo "Judge load format: ${AI_LAB_JUDGE_LOAD_FORMAT:-auto}"

  if [[ "$AI_LAB_JUDGE_MODEL" == "$AI_LAB_JUDGE_DIR" ]] && ! model_ready "$AI_LAB_JUDGE_MODEL"; then
    download_model_if_needed "$AI_LAB_JUDGE_REPO" "$AI_LAB_JUDGE_DIR" "Judge"
  elif [[ "$AI_LAB_JUDGE_MODEL" == /* ]] && ! model_ready "$AI_LAB_JUDGE_MODEL"; then
    echo "ERROR: Local judge model folder is missing or incomplete: $AI_LAB_JUDGE_MODEL"
    echo "Expected a config.json in that folder."
    exit 1
  fi

  JUDGE_TRUST_REMOTE_CODE_ARGS=()
  if [[ "$AI_LAB_JUDGE_TRUST_REMOTE_CODE" == "true" ]]; then
    JUDGE_TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
  fi
  JUDGE_FORMAT_ARGS=()
  if [[ -n "${AI_LAB_JUDGE_TOKENIZER_MODE:-}" ]]; then
    JUDGE_FORMAT_ARGS+=(--tokenizer-mode "$AI_LAB_JUDGE_TOKENIZER_MODE")
  fi
  if [[ -n "${AI_LAB_JUDGE_CONFIG_FORMAT:-}" ]]; then
    JUDGE_FORMAT_ARGS+=(--config-format "$AI_LAB_JUDGE_CONFIG_FORMAT")
  fi
  if [[ -n "${AI_LAB_JUDGE_LOAD_FORMAT:-}" ]]; then
    JUDGE_FORMAT_ARGS+=(--load-format "$AI_LAB_JUDGE_LOAD_FORMAT")
  fi

  run_python \
    python3 -s judge_enrichment.py \
      --input-dir "$INPUT_DIR" \
      --enriched-dir "$RESULTS_DIR" \
      --output-dir "$RESULTS_DIR" \
      --model "$AI_LAB_JUDGE_MODEL" \
      --max-model-len "$AI_LAB_JUDGE_MAX_MODEL_LEN" \
      --max-tokens "$AI_LAB_JUDGE_MAX_TOKENS" \
      --tensor-parallel-size "$AI_LAB_JUDGE_TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$AI_LAB_GPU_MEMORY_UTILIZATION" \
      --sample-size "$AI_LAB_JUDGE_SAMPLE_SIZE" \
      "${JUDGE_FORMAT_ARGS[@]}" \
      "${JUDGE_TRUST_REMOTE_CODE_ARGS[@]}"

  echo "Judged lines: $(find "$RESULTS_DIR" -maxdepth 1 -name 'news_judged*.jsonl' -exec wc -l {} + 2>/dev/null | tail -n 1 || echo 0)"
  echo "Failed judgement lines: $(find "$RESULTS_DIR" -maxdepth 1 -name 'failed_judgements*.jsonl' -exec wc -l {} + 2>/dev/null | tail -n 1 || echo 0)"
  echo "=== News enrichment judge step finished ==="
}

select_runtime

echo "Host GPU visibility:"
nvidia-smi -L || true

echo "Python and CUDA visibility:"
run_python \
  python3 -s -c "import sys, torch; print(sys.executable); print('cuda_available=', torch.cuda.is_available()); print('cuda_device_count=', torch.cuda.device_count()); [print('cuda_device', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

if [[ "$MODE" == "judge-only" ]]; then
  run_judge_step
  exit 0
fi

if [[ "$DOWNLOAD_MODEL_IF_MISSING" == true && "$AI_LAB_MODEL" == "$AI_LAB_DOWNLOAD_DIR" ]] && ! model_ready "$AI_LAB_MODEL"; then
  echo "Local model is missing. Downloading it in this Slurm job before enrichment."
  download_model
elif [[ "$AI_LAB_MODEL" == /* ]] && ! model_ready "$AI_LAB_MODEL"; then
  echo "ERROR: Local model folder is missing or incomplete: $AI_LAB_MODEL"
  echo "For the stable fallback model, run: sbatch --export=ALL run_ai_lab.sh qwen25-32b"
  exit 1
fi

: > "$RESULTS_DIR/news_enriched.jsonl"
: > "$RESULTS_DIR/failed_records.jsonl"

TRUST_REMOTE_CODE_ARGS=()
if [[ "$AI_LAB_TRUST_REMOTE_CODE" == "true" ]]; then
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

if [[ "${#INPUT_FILES[@]}" -gt 0 ]]; then
  echo "Batch input files: ${#INPUT_FILES[@]}"
  echo "Total input lines: $(wc -l "${INPUT_FILES[@]}" | tail -n 1)"

  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#INPUT_FILES[@]}" ]]; then
      echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID, but only ${#INPUT_FILES[@]} batch files exist."
      exit 1
    fi
    INPUT_PATH="${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}"
    BATCH_STEM="$(basename "$INPUT_PATH" .jsonl)"
    BATCH_SUFFIX="${BATCH_STEM#news_to_enrich_}"
    echo "Running array batch $SLURM_ARRAY_TASK_ID: $INPUT_PATH"
    run_python \
      python3 -s enrich_articles.py \
        --input "$INPUT_PATH" \
        --output "$RESULTS_DIR/news_enriched_${BATCH_SUFFIX}.jsonl" \
        --failed-output "$RESULTS_DIR/failed_records_${BATCH_SUFFIX}.jsonl" \
        --model "$AI_LAB_MODEL" \
        --max-model-len "$AI_LAB_MAX_MODEL_LEN" \
        --tensor-parallel-size "$AI_LAB_TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$AI_LAB_GPU_MEMORY_UTILIZATION" \
        "${TRUST_REMOTE_CODE_ARGS[@]}"
  else
    echo "Running all batch files in one job with one model load."
    find "$RESULTS_DIR" -maxdepth 1 -name 'news_enriched_*.jsonl' -delete
    find "$RESULTS_DIR" -maxdepth 1 -name 'failed_records_*.jsonl' -delete
    run_python \
      python3 -s enrich_articles.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$RESULTS_DIR" \
        --model "$AI_LAB_MODEL" \
        --max-model-len "$AI_LAB_MAX_MODEL_LEN" \
        --tensor-parallel-size "$AI_LAB_TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$AI_LAB_GPU_MEMORY_UTILIZATION" \
        "${TRUST_REMOTE_CODE_ARGS[@]}"
  fi
else
  echo "Input lines: $(wc -l < news_to_enrich.jsonl)"
  run_python \
    python3 -s enrich_articles.py \
      --input news_to_enrich.jsonl \
      --output "$RESULTS_DIR/news_enriched.jsonl" \
      --failed-output "$RESULTS_DIR/failed_records.jsonl" \
      --model "$AI_LAB_MODEL" \
      --max-model-len "$AI_LAB_MAX_MODEL_LEN" \
      --tensor-parallel-size "$AI_LAB_TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$AI_LAB_GPU_MEMORY_UTILIZATION" \
      "${TRUST_REMOTE_CODE_ARGS[@]}"
fi

echo "Output lines: $(find "$RESULTS_DIR" -maxdepth 1 -name 'news_enriched*.jsonl' -exec wc -l {} + 2>/dev/null | tail -n 1 || echo 0)"
echo "Failed lines: $(find "$RESULTS_DIR" -maxdepth 1 -name 'failed_records*.jsonl' -exec wc -l {} + 2>/dev/null | tail -n 1 || echo 0)"

if [[ "$AI_LAB_RUN_JUDGE" == "true" ]]; then
  run_judge_step
else
  echo "Judge step skipped. Set AI_LAB_RUN_JUDGE=true to run Mistral quality audit."
fi

echo "=== News enrichment job finished ==="
