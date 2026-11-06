#!/usr/bin/env bash
# network-enabled source host entry: token stays in its private auth directory, never copied to offline worker.
set -euo pipefail
set +x
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/tools/worker_transfer_env.sh"
export AOTD_HF_AUTH_DIR=${AOTD_HF_AUTH_DIR:?Set the private source credential directory}
bash "$ROOT/configure_hf_access.sh" --check
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
"${AOTD_DOWNLOAD_PYTHON:-python3}" -B -u "$ROOT/tools/download_gated_teachers.py"

for name in Meta-Llama-3-8B-Instruct Llama-3.1-8B-Instruct; do
    rsync -a --exclude .cache/ --exclude '*.safetensors' --rsync-path="$REMOTE_RSYNC" \
        "$ROOT/models/teachers/$name/" "$HOST:$DEST/models/teachers/$name/"
done
pids=()
for shard in "$ROOT"/models/teachers/{Meta-Llama-3-8B-Instruct,Llama-3.1-8B-Instruct}/*.safetensors; do
    name=$(basename "$(dirname "$shard")")
    rsync -a --partial --append-verify --rsync-path="$REMOTE_RSYNC" "$shard" \
        "$HOST:$DEST/models/teachers/$name/" > "$ROOT/logs/transfer_${name}_$(basename "$shard").log" 2>&1 &
    pids+=("$!")
    if ((${#pids[@]}==4));then
        for pid in "${pids[@]}";do wait "$pid";done
        pids=()
    fi
done
for pid in "${pids[@]}";do wait "$pid";done
scp "$ROOT/models/teachers/gated_manifest.json" "$HOST:$DEST/models/teachers/"
ssh -o BatchMode=yes "$HOST" "cd '$DEST' && source runtime_env.sh && env/bin/python -B tools/verify_gated_teachers.py"
echo '[Llama staging] Download, transfer and target SHA256 verification complete; token retained on network-enabled source host only'
