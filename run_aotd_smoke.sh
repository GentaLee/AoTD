#!/usr/bin/env bash
# Run on the Taihua GPU pod after the offline bundle has been transferred.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=all
case "${1:-}" in
    '') ;;
    --eval-only) MODE=eval-only;;
    --train-only) MODE=train-only;;
    --help|-h) echo "Usage: $0 [--eval-only|--train-only]"; exit 0;;
    *) echo 'Unknown argument; use --help' >&2; exit 2;;
esac
case "$(findmnt -T "$ROOT" -n -o FSTYPE)" in overlay|tmpfs|ramfs|'') echo 'ERROR: project must reside on persistent mounted storage' >&2; exit 1;; esac
source "$ROOT/runtime_env.sh"
export AOTD_RUN_DIR="$ROOT/results/smoke_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$AOTD_RUN_DIR"
exec > >(tee "$AOTD_RUN_DIR/run.log") 2>&1
echo "[start] mode=$MODE result=$AOTD_RUN_DIR"
if [[ ! -f "$ROOT/runtime/environment-ready" ]]; then
    bash "$ROOT/setup_offline.sh"
fi
echo '[prepare] Verify offline model assets (before pausing keep-alive)'
"$ROOT/env/bin/python" -B "$ROOT/tools/verify_assets.py"
echo '[prepare] Pause keep-alive and run GPU smoke; restore it on completion or interruption'
if "$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" \
    --keepalive-script "${AOTD_KEEPALIVE_SCRIPT:-/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py}" \
    --state-dir "$ROOT/runtime/guard-state" -- \
    "$ROOT/env/bin/python" -B "$ROOT/tools/with_spare_keepalive.py" --mode "$MODE" \
    --keepalive-script "${AOTD_KEEPALIVE_SCRIPT:-/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py}" \
    --state-dir "$AOTD_RUN_DIR" -- bash "$ROOT/tools/smoke_pipeline.sh" "$MODE"; then
    echo '[smoke] PASS: original keep-alive restored'
else
    code=$?
    echo "[smoke] failed or interrupted: exit=$code"
    exit "$code"
fi
echo "[done] Smoke passed and keep-alive restored. Results: $AOTD_RUN_DIR"
