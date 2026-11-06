#!/usr/bin/env bash
# Run on Shanghai; download there, transfer only data, extract on Taihua GPU Pod.
set -euo pipefail
source /volume/yzhao04/persist/recovery-20261022/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=/volume/pt-test/users/zhaoye/aotd-nextqa
python -B -u "$ROOT/tools/download_charades.py"
rsync -a --partial --append-verify --rsync-path="env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync" \
    "$ROOT/data/raw/Charades/" "yzhao04-gpu-taihua:$DEST/data/raw/Charades/"
scp "$ROOT/tools/extract_charades_parts.py" "$ROOT/tools/prepare_star_clips.py" "yzhao04-gpu-taihua:$DEST/tools/"
ssh -o BatchMode=yes yzhao04-gpu-taihua "cd '$DEST' && source runtime_env.sh && env/bin/python -B tools/extract_charades_parts.py && env/bin/python -B tools/prepare_star_clips.py && bash run_aotd_build_data.sh --prepare-only"
