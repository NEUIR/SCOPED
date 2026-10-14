#!/usr/bin/env bash
# Stage 2 only: bash scripts/extract_skills.sh [all|capability1|...]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/paths.sh"
source "$ROOT/scripts/lib/teacher_paths.sh"
load_teacher_paths

TARGET=${1:-all}
case "$TARGET" in
    all) CAPABILITIES=(capability1 capability2 capability3 capability4) ;;
    capability1|capability2|capability3|capability4) CAPABILITIES=("$TARGET") ;;
    *) echo "Usage: bash scripts/extract_skills.sh [all|capability1|capability2|capability3|capability4]" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
source "$ROOT/scripts/lib/python.sh"
SKILL_PYTHON=$(scoped_python scoped "${SKILL_PYTHON:-${EVAL_PYTHON:-}}")
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
SKILL_WORK_DIR=${SKILL_WORK_DIR:-$OUTPUT_DIR/skill_extraction}
for capability in "${CAPABILITIES[@]}"; do
    index=${capability#capability}
    key="CAPABILITY${index}_TEACHER"
    model=${!key:-}
    [[ -n "$model" ]] || { echo "Set $key in scripts/train.sh, or export the stage-1 checkpoints first." >&2; exit 2; }
done
for capability in "${CAPABILITIES[@]}"; do
    index=${capability#capability}; key="CAPABILITY${index}_TEACHER"; model=${!key}
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        PREPARE_ARGS=(--capability "$capability" --output "$DATA_DIR")
        if [[ ${AUTO_DOWNLOAD_DATA:-1} == 0 ]]; then PREPARE_ARGS+=(--offline); fi
        "$SKILL_PYTHON" "$ROOT/scripts/lib/prepare_data.py" "${PREPARE_ARGS[@]}"
    fi
    CMD=("$SKILL_PYTHON" -m scoped_skill --capability "$capability" --model "$model"
        --data-dir "$DATA_DIR" --work-dir "$SKILL_WORK_DIR"
        --config "${SKILL_CONFIG:-$ROOT/configs/skill_extraction.yaml}" --port "${SKILL_PORT:-8100}" --resume)
    if [[ -n ${SKILL_SAMPLE:-} ]]; then CMD+=(--sample "$SKILL_SAMPLE"); fi
    if [[ -n ${SKILL_SERVER_TP:-} ]]; then CMD+=(--server-tp "$SKILL_SERVER_TP"); fi
    if [[ -n ${SKILL_BASE_URL:-} ]]; then CMD+=(--base-url "$SKILL_BASE_URL"); fi
    CMD+=("$@")
    if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n';
    else "${CMD[@]}"; fi
done
