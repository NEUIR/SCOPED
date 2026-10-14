#!/usr/bin/env bash
# Run one stage: bash scripts/train.sh [rl|skill|opd|capability1|...]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/paths.sh"
source "$ROOT/scripts/lib/teacher_paths.sh"
source "$ROOT/scripts/lib/python.sh"

# ==================== EDIT MODEL PATHS HERE ====================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
STUDENT_MODEL="${STUDENT_MODEL:-$BASE_MODEL}"
# Stage-1 exports are recorded in OUTPUT_DIR/teachers.env; optional overrides:
CAPABILITY1_TEACHER="${CAPABILITY1_TEACHER:-}"
CAPABILITY2_TEACHER="${CAPABILITY2_TEACHER:-}"
CAPABILITY3_TEACHER="${CAPABILITY3_TEACHER:-}"
CAPABILITY4_TEACHER="${CAPABILITY4_TEACHER:-}"
# ===============================================================

STAGE=${1:-}
case "$STAGE" in rl|skill|opd|capability1|capability2|capability3|capability4) ;;
    *) echo "Choose a stage: bash scripts/train.sh [rl|skill|opd|capability1|capability2|capability3|capability4]" >&2; exit 2 ;; esac
shift
export MODEL_PATH="$BASE_MODEL" STUDENT_MODEL
if [[ "$STAGE" == opd ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi

choose_python() {
    if [[ "$1" == rl ]]; then
        scoped_python rl "${RL_PYTHON:-}"
    else
        scoped_python scoped "${OPD_PYTHON:-${DISTILL_PYTHON:-}}"
    fi
}

attention_backend() {
    if [[ -n ${ATTENTION_BACKEND:-} ]]; then printf '%s' "$ATTENTION_BACKEND";
    elif "$1" -c 'import flash_attn' >/dev/null 2>&1; then printf '%s' flash_attention_2;
    else printf '%s' sdpa; fi
}

run_rl() {
    PYTHON_BIN="$1" NGPUS_PER_NODE="${RL_GPUS:-4}" bash "$ROOT/scripts/lib/train_capability.sh" "$2"
}

run_opd() {
    local python=$1
    shift
    load_teacher_paths
    export TEACHER_SKILL_DIR=${TEACHER_SKILL_DIR:-${SKILL_WORK_DIR:-$OUTPUT_DIR/skill_extraction}/skills}
    PYTHON_BIN="$python" NGPUS_PER_NODE="${OPD_ACTOR_GPUS:-4}" \
        TEACHER_GPU_PER_SLOT="${TEACHER_GPU_PER_SLOT:-1.0}" \
        TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.70}" \
        ATTENTION_BACKEND="$(attention_backend "$python")" bash "$ROOT/scripts/lib/train_distillation.sh" "$@"
}

case "$STAGE" in
    capability1|capability2|capability3|capability4)
        RL_BIN=$(choose_python rl)
        run_rl "$RL_BIN" "$STAGE" ;;
    rl)
        RL_BIN=$(choose_python rl)
        for capability in capability1 capability2 capability3 capability4; do run_rl "$RL_BIN" "$capability"; done
        if [[ ${DRY_RUN:-0} != 1 ]]; then "$RL_BIN" "$ROOT/scripts/lib/export_teachers.py"; fi
        ;;
    skill)
        load_teacher_paths
        if [[ $# -eq 0 ]]; then set -- all; fi
        bash "$ROOT/scripts/extract_skills.sh" "$@"
        ;;
    opd)
        OPD_BIN=$(choose_python opd)
        run_opd "$OPD_BIN" "$@" ;;
esac
