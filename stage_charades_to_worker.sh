#!/usr/bin/env bash
# Run on network-enabled source host; download there, transfer only data, extract on offline worker GPU Pod.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/tools/worker_transfer_env.sh"
"${AOTD_DOWNLOAD_PYTHON:-python3}" -B -u "$ROOT/tools/download_charades.py"
rsync -a --partial --append-verify --rsync-path="$REMOTE_RSYNC" \
    "$ROOT/data/raw/Charades/" "$HOST:$DEST/data/raw/Charades/"
scp "$ROOT/tools/extract_charades_parts.py" "$ROOT/tools/prepare_star_clips.py" "$HOST:$DEST/tools/"
ssh -o BatchMode=yes "$HOST" "cd '$DEST' && source runtime_env.sh && env/bin/python -B tools/extract_charades_parts.py && env/bin/python -B tools/prepare_star_clips.py && bash run_aotd_build_data.sh --prepare-only"
