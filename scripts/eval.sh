#!/usr/bin/env bash
# Evaluate the released HF checkpoint: bash scripts/eval.sh [all|chartqa|ccocrv2|...]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/paths.sh"

# Optional override: MODEL=xiongyq/SCOPED-4B bash scripts/eval.sh all
MODEL=${MODEL:-xiongyq/SCOPED-8B}

DATASETS=${1:-all}
if [[ $# -gt 0 ]]; then shift; fi
source "$ROOT/scripts/lib/python.sh"
EVAL_PYTHON=$(scoped_python scoped "${EVAL_PYTHON:-}")
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
CMD=("$EVAL_PYTHON" "$ROOT/src/evaluate.py" --model "$MODEL" --datasets "$DATASETS" --online
    --eval-root "${EVAL_DATA_DIR:-$SCOPED_STORAGE/data/hf_eval}"
    --gpus "$CUDA_VISIBLE_DEVICES" --serve-tp "${SERVE_TP:-${TENSOR_PARALLEL_SIZE:-4}}" --serve-dp "${SERVE_DP:-1}"
    --port "${PORT:-8000}" --concurrency "${CONCURRENCY:-100}")
if [[ -n ${BATCH_SIZE:-} ]]; then CMD+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n ${MAX_NUM_SEQS:-} ]]; then CMD+=(--max-num-seqs "$MAX_NUM_SEQS"); fi
if [[ -n ${MAX_NUM_BATCHED_TOKENS:-} ]]; then CMD+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"); fi
if [[ -n ${MAX_MODEL_LEN:-} ]]; then CMD+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [[ -n ${MAX_TOKENS:-} ]]; then CMD+=(--max-tokens "$MAX_TOKENS"); fi
if [[ -n ${EVAL_OUT_DIR:-} ]]; then CMD+=(--out-dir "$EVAL_OUT_DIR"); fi
if [[ -n ${BASE_URL:-} ]]; then CMD+=(--base-url "$BASE_URL"); fi
CMD+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
exec "${CMD[@]}"
