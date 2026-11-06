#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/tools/worker_transfer_env.sh"
REMOTE="$REMOTE_RSYNC"
SOURCES=(
    "${AOTD_LLAMA3_SOURCE:?Set the authorized local Llama3 snapshot directory}"
    "${AOTD_LLAMA31_SOURCE:?Set the authorized local Llama3.1 model directory}"
)
NAMES=(Meta-Llama-3-8B-Instruct Llama-3.1-8B-Instruct)
for index in 0 1;do
    source=${SOURCES[$index]};name=${NAMES[$index]}
    rsync -aL --exclude .cache/ --exclude '*.safetensors' --rsync-path="$REMOTE" "$source/" "$HOST:$DEST/models/teachers/$name/"
done
pids=()
for index in 0 1;do
    source=${SOURCES[$index]};name=${NAMES[$index]}
    for shard in "$source"/*.safetensors;do
        echo "[local model transfer] $name/$(basename "$shard")"
        rsync -aL --partial --append-verify --rsync-path="$REMOTE" "$shard" "$HOST:$DEST/models/teachers/$name/" \
            > "$ROOT/logs/transfer_local_${name}_$(basename "$shard").log" 2>&1 &
        pids+=("$!")
        if ((${#pids[@]}==8));then
            for pid in "${pids[@]}";do wait "$pid";done
            pids=()
        fi
    done
done
for pid in "${pids[@]}";do wait "$pid";done
echo '[local model transfer] Both local models staged on offline worker'
