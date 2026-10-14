#!/usr/bin/env bash
# Preserve explicit teacher overrides when reading exported checkpoint paths.
load_teacher_paths() {
    local saved1=${CAPABILITY1_TEACHER:-} saved2=${CAPABILITY2_TEACHER:-}
    local saved3=${CAPABILITY3_TEACHER:-} saved4=${CAPABILITY4_TEACHER:-}
    if [[ -z "$saved1" || -z "$saved2" || -z "$saved3" || -z "$saved4" ]]; then
        if [[ -f "$OUTPUT_DIR/teachers.env" ]]; then source "$OUTPUT_DIR/teachers.env"; fi
        CAPABILITY1_TEACHER=${saved1:-${CAPABILITY1_TEACHER:-}}
        CAPABILITY2_TEACHER=${saved2:-${CAPABILITY2_TEACHER:-}}
        CAPABILITY3_TEACHER=${saved3:-${CAPABILITY3_TEACHER:-}}
        CAPABILITY4_TEACHER=${saved4:-${CAPABILITY4_TEACHER:-}}
    fi
    export CAPABILITY1_TEACHER CAPABILITY2_TEACHER CAPABILITY3_TEACHER CAPABILITY4_TEACHER
}
