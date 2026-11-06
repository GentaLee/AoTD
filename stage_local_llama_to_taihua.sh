#!/usr/bin/env bash
set -euo pipefail
source /volume/yzhao04/persist/recovery-20261022/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=/volume/pt-test/users/zhaoye/aotd-nextqa
REMOTE="env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync"
SOURCES=(
    "${AOTD_LLAMA3_SOURCE:?Set the authorized local Llama3 snapshot directory}"
    "${AOTD_LLAMA31_SOURCE:?Set the authorized local Llama3.1 model directory}"
)
NAMES=(Meta-Llama-3-8B-Instruct Llama-3.1-8B-Instruct)
for index in 0 1;do
    source=${SOURCES[$index]};name=${NAMES[$index]}
    rsync -aL --exclude .cache/ --exclude '*.safetensors' --rsync-path="$REMOTE" "$source/" "yzhao04-gpu-taihua:$DEST/models/teachers/$name/"
done
pids=()
for index in 0 1;do
    source=${SOURCES[$index]};name=${NAMES[$index]}
    for shard in "$source"/*.safetensors;do
        echo "[local model transfer] $name/$(basename "$shard")"
        rsync -aL --partial --append-verify --rsync-path="$REMOTE" "$shard" "yzhao04-gpu-taihua:$DEST/models/teachers/$name/" \
            > "$ROOT/logs/transfer_local_${name}_$(basename "$shard").log" 2>&1 &
        pids+=("$!")
        if ((${#pids[@]}==8));then
            for pid in "${pids[@]}";do wait "$pid";done
            pids=()
        fi
    done
done
for pid in "${pids[@]}";do wait "$pid";done
echo '[local model transfer] Both local models staged on Taihua'
