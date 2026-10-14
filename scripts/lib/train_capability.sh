#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/paths.sh"
CAPABILITY=${1:?Usage: bash scripts/lib/train_capability.sh capability1 [key=value ...]}
shift
case "$CAPABILITY" in capability1|capability2|capability3|capability4) ;; *) echo "Unknown capability: $CAPABILITY" >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT/src/easyr1${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/scoped-ray}
source "$ROOT/scripts/lib/python.sh"
PYTHON_BIN=$(scoped_python rl "${PYTHON_BIN:-${RL_PYTHON:-}}")
CMD=("$PYTHON_BIN" -m verl.trainer.main "config=$ROOT/configs/rl/$CAPABILITY.yaml"
    "data.train_files=$DATA_DIR/$CAPABILITY/train" "data.val_files=$DATA_DIR/$CAPABILITY/test"
    "worker.actor.model.model_path=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE:-4}"
    "trainer.save_checkpoint_path=$OUTPUT_DIR/rl/$CAPABILITY" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
if [[ ${SKIP_ENV_CHECK:-0} != 1 ]]; then "$PYTHON_BIN" "$ROOT/scripts/lib/check_environment.py" --stage rl --gpus "${NGPUS_PER_NODE:-4}"; fi
PREPARE_ARGS=(--capability "$CAPABILITY" --output "$DATA_DIR")
if [[ ${AUTO_DOWNLOAD_DATA:-1} == 0 ]]; then PREPARE_ARGS+=(--offline); fi
"$PYTHON_BIN" "$ROOT/scripts/lib/prepare_data.py" "${PREPARE_ARGS[@]}"
exec "${CMD[@]}"
