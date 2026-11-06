#!/usr/bin/env bash
set -euo pipefail
ROOT=${AOTD_WORKER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aotd-nextqa}
exec bash "$ROOT/run_aotd_smoke.sh" "$@"
