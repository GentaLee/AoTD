#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/runtime_env.sh"
MODE=run
ARGS=()
while (($#)); do
    case "$1" in
        --prepare-only) MODE=prepare; shift;;
        --check) MODE=check; shift;;
        --dataset|--stage) ARGS+=("$1" "${2:?Missing value}"); shift 2;;
        --retry-errors) ARGS+=("$1"); shift;;
        --help|-h)
            echo 'Usage: run_aotd_build_data.sh [--prepare-only|--check] [--dataset NExT-QA|STAR|AGQA|ANetQA|CLEVRER|EgoQA] [--stage all|construct|transfer|filter|export] [--retry-errors]'
            echo 'Default: all six train sources, Agent execution, correct-answer gate, Llama transfer, rationale filter and export. Resumes durable JSONL journals.'
            echo 'Requires source train annotations/media and offline teacher weights. Missing inputs stop before pausing keep-alive.'
            echo 'Prepare-only normalizes available real annotations and reports every absent source/video without GPU inference.'
            exit 0;;
        *) echo "Unknown argument: $1" >&2; exit 2;;
    esac
done
case "$(findmnt -T "$ROOT" -n -o FSTYPE)" in overlay|tmpfs|ramfs|'') echo 'Persistent volume required';exit 2;; esac
export AOTD_RUN_DIR="$ROOT/results/build_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$AOTD_RUN_DIR"
exec > >(tee "$AOTD_RUN_DIR/run.log") 2>&1
echo "[build] runtime/progress=$AOTD_RUN_DIR; durable data=$ROOT/data/construction"
if [[ "$MODE" != run ]]; then
    exec "$ROOT/env/bin/python" -B "$ROOT/tools/build_sources.py" --mode "$MODE" "${ARGS[@]}"
fi
"$ROOT/env/bin/python" -B "$ROOT/tools/build_sources.py" --mode check "${ARGS[@]}"
KEEP=${AOTD_KEEPALIVE_SCRIPT:-/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py}
"$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" --keepalive-script "$KEEP" --state-dir "$ROOT/runtime/guard-state" -- \
    "$ROOT/env/bin/python" -B "$ROOT/tools/build_sources.py" --mode run "${ARGS[@]}"
