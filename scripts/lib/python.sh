#!/usr/bin/env bash
# Select the stage's interpreter without falling back to another environment.
scoped_python() {
    local environment=$1 override=${2:-} candidate requirements
    case "$environment" in
        rl) candidate="$ROOT/.venv-rl/bin/python"; requirements=requirements-rl.txt ;;
        scoped) candidate="$ROOT/.venv-scoped/bin/python"; requirements=requirements.txt ;;
        *) echo "Unknown environment: $environment" >&2; return 2 ;;
    esac
    if [[ -n "$override" ]]; then
        if command -v "$override" >/dev/null 2>&1; then
            command -v "$override"
            return
        fi
        echo "Configured Python is not executable: $override" >&2
        return 2
    fi
    if [[ -x "$candidate" ]]; then
        printf '%s' "$candidate"
        return
    fi
    echo "Missing .venv-$environment. Create it and install $requirements as shown in README." >&2
    return 2
}
