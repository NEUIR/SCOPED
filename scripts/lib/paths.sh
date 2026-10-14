#!/usr/bin/env bash
if [[ -z ${SCOPED_STORAGE:-} ]]; then
    if [[ -d /data && -w /data ]]; then SCOPED_STORAGE=/data/SCOPED;
    else SCOPED_STORAGE="${XDG_CACHE_HOME:-$HOME/.cache}/scoped"; fi
fi
export SCOPED_STORAGE
export DATA_DIR=${DATA_DIR:-$SCOPED_STORAGE/data/${SCOPED_DEFAULT_DATA_SUBDIR:-hf_dataset}}
export OUTPUT_DIR=${OUTPUT_DIR:-$SCOPED_STORAGE/outputs}
