#!/usr/bin/env bash
# One-shot prerequisite wait on Shanghai, ending after staging or after 24 hours.
set -euo pipefail
set +x
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /volume/yzhao04/persist/recovery-20261022/env.sh
START=$(date +%s)
while (( $(date +%s)-START < 86400 )); do
    echo "[HF permission check] $(date -u +%FT%TZ)"
    if bash "$ROOT/configure_hf_access.sh" --check; then
        exec bash "$ROOT/download_llama_and_sync.sh"
    fi
    echo '[HF wait] Still awaiting model access; next check in five minutes'
    for _ in 1 2 3 4 5; do sleep 60; done
done
echo '[HF wait] 24-hour wait ended without both model permissions; saved token unchanged'
exit 2
