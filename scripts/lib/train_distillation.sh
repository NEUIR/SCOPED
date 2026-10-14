#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "$ROOT/scripts/lib/paths.sh"
export PYTHONPATH="$ROOT/src/verl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source "$ROOT/scripts/lib/python.sh"
PYTHON_BIN=$(scoped_python scoped "${PYTHON_BIN:-${OPD_PYTHON:-${DISTILL_PYTHON:-}}}")
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash_attention_2}
TEACHERS=("${CAPABILITY1_TEACHER:-}" "${CAPABILITY2_TEACHER:-}" "${CAPABILITY3_TEACHER:-}" "${CAPABILITY4_TEACHER:-}")
KEYS=(capability1_perception capability2_ocr_chart capability3_math capability4_multiimage)
USE_TEACHER_SKILLS=${USE_TEACHER_SKILLS:-1}
case "$USE_TEACHER_SKILLS" in 0|1) ;; *) echo "USE_TEACHER_SKILLS must be 0 or 1" >&2; exit 2 ;; esac
TEACHER_SKILL_DIR=${TEACHER_SKILL_DIR:-${SKILL_WORK_DIR:-$OUTPUT_DIR/skill_extraction}/skills}
SKILLS=()
for index in 0 1 2 3; do
    [[ -n ${TEACHERS[$index]} ]] || { echo "Set CAPABILITY$((index + 1))_TEACHER to an exported HF-format model" >&2; exit 2; }
    if [[ "$USE_TEACHER_SKILLS" == 1 ]]; then
        capability="capability$((index + 1))"
        file="$TEACHER_SKILL_DIR/$capability/SKILL.md"
        # Existing user-provided flat text directories remain supported.
        if [[ ! -s "$file" && -s "$TEACHER_SKILL_DIR/$capability.txt" ]]; then file="$TEACHER_SKILL_DIR/$capability.txt"; fi
        if [[ ${DRY_RUN:-0} != 1 && ! -s "$file" ]]; then
            echo "Missing teacher skill: $file. Run stage 2: bash scripts/train.sh skill" >&2; exit 2
        fi
        SKILLS+=("$file")
    fi
done
if [[ ${DRY_RUN:-0} != 1 ]]; then
    if [[ "$USE_TEACHER_SKILLS" == 1 ]]; then
        "$PYTHON_BIN" "$ROOT/scripts/lib/check_teacher_skills.py" --models "${TEACHERS[@]}" --skills "${SKILLS[@]}" \
            --max-model-len "${TEACHER_MAX_MODEL_LEN:-21409}" --student-context "${SKILL_STUDENT_CONTEXT:-20384}"
    fi
    if [[ ${SKIP_ENV_CHECK:-0} != 1 ]]; then
        required=$("$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1])+4*float(sys.argv[2]))' "${NGPUS_PER_NODE:-4}" "${TEACHER_GPU_PER_SLOT:-1.0}")
        "$PYTHON_BIN" "$ROOT/scripts/lib/check_environment.py" --stage distillation --gpus "$required" --attention "$ATTENTION_BACKEND"
    fi
    PREPARE_ARGS=(--capability all --output "$DATA_DIR")
    if [[ ${AUTO_DOWNLOAD_DATA:-1} == 0 ]]; then PREPARE_ARGS+=(--offline); fi
    "$PYTHON_BIN" "$ROOT/scripts/lib/prepare_data.py" "${PREPARE_ARGS[@]}"
fi
CMD=("$PYTHON_BIN" -m verl.trainer.main_ppo --config-path "$ROOT/configs" --config-name distillation
    "actor_rollout_ref.model.path=${STUDENT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE:-4}" "trainer.default_local_dir=$OUTPUT_DIR/distillation"
    "distillation.gpu_per_slot=${TEACHER_GPU_PER_SLOT:-1.0}")
case "$ATTENTION_BACKEND" in
    flash_attention_2) ;;
    sdpa) CMD+=(actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.model.use_remove_padding=False) ;;
    *) echo "ATTENTION_BACKEND must be flash_attention_2 or sdpa" >&2; exit 2 ;;
esac
if [[ ${DRY_RUN:-0} == 1 ]]; then
    CMD+=("data.train_files=['$DATA_DIR/capability1/train/*.parquet','$DATA_DIR/capability2/train/*.parquet','$DATA_DIR/capability3/train/*.parquet','$DATA_DIR/capability4/train/*.parquet']"
        "data.val_files=['$DATA_DIR/capability1/test/*.parquet','$DATA_DIR/capability2/test/*.parquet','$DATA_DIR/capability3/test/*.parquet','$DATA_DIR/capability4/test/*.parquet']")
else
    files=$("$PYTHON_BIN" "$ROOT/scripts/lib/list_data_files.py" "$DATA_DIR")
    mapfile -t lines <<< "$files"
    CMD+=("data.train_files=${lines[0]}" "data.val_files=${lines[1]}")
fi
for index in 0 1 2 3; do
    capability="capability$((index + 1))"
    CMD+=("+distillation.teacher_models.$capability.key=${KEYS[$index]}"
        "+distillation.teacher_models.$capability.model_path=${TEACHERS[$index]}"
        "+distillation.teacher_models.$capability.num_replicas=1"
        "+distillation.teacher_models.$capability.inference.name=vllm"
        "+distillation.teacher_models.$capability.inference.tensor_model_parallel_size=1"
        "+distillation.teacher_models.$capability.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL:-0.70}"
        "+distillation.teacher_models.$capability.inference.max_model_len=${TEACHER_MAX_MODEL_LEN:-21409}"
        "+distillation.teacher_models.$capability.inference.max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-12288}"
        "+distillation.teacher_models.$capability.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS:-20}"
        "+distillation.teacher_models.$capability.inference.enforce_eager=${TEACHER_ENFORCE_EAGER:-False}")
    if [[ "$USE_TEACHER_SKILLS" == 1 ]]; then
        CMD+=("+distillation.teacher_models.$capability.skill_prompt_path=${SKILLS[$index]}")
    fi
done
CMD+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
exec "${CMD[@]}"
