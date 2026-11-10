#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/runtime_env.sh"
[[ ${AOTD_INDEPENDENT_SPARE_KEEPALIVE:-0} == 1 ]] || { echo 'Independent keepalive groups have not been configured'; exit 2; }
if [[ ${1:-} == -- ]]; then shift; fi
(($#)) || { echo "Usage: bash $0 -- COMMAND [ARGS...] (physical GPUs 4-7, logical GPUs 0-3)"; exit 2; }
# The borrowed task is independent of any AoTD cleanup tag.
unset AOTD_GUARD_RUN_ID
exec "$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" \
  --keepalive-script "${AOTD_SPARE_KEEPALIVE_SCRIPT:?}" \
  --state-dir "$ROOT/runtime/spare-guard-state" --gpus 4,5,6,7 -- "$@"
