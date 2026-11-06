#!/usr/bin/env bash
# Shanghai entry: token stays in its private auth directory, never copied to Taihua.
set -euo pipefail
set +x
source /volume/yzhao04/persist/recovery-20261022/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=/volume/pt-test/users/zhaoye/aotd-nextqa
bash "$ROOT/configure_hf_access.sh" --check
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
python -B -u "$ROOT/tools/download_gated_teachers.py"
REMOTE_RSYNC="env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync"
for name in Meta-Llama-3-8B-Instruct Llama-3.1-8B-Instruct; do
    rsync -a --exclude .cache/ --exclude '*.safetensors' --rsync-path="$REMOTE_RSYNC" \
        "$ROOT/models/teachers/$name/" "yzhao04-gpu-taihua:$DEST/models/teachers/$name/"
done
pids=()
for shard in "$ROOT"/models/teachers/{Meta-Llama-3-8B-Instruct,Llama-3.1-8B-Instruct}/*.safetensors; do
    name=$(basename "$(dirname "$shard")")
    rsync -a --partial --append-verify --rsync-path="$REMOTE_RSYNC" "$shard" \
        "yzhao04-gpu-taihua:$DEST/models/teachers/$name/" > "$ROOT/logs/transfer_${name}_$(basename "$shard").log" 2>&1 &
    pids+=("$!")
    if ((${#pids[@]}==4));then
        for pid in "${pids[@]}";do wait "$pid";done
        pids=()
    fi
done
for pid in "${pids[@]}";do wait "$pid";done
scp "$ROOT/models/teachers/gated_manifest.json" "yzhao04-gpu-taihua:$DEST/models/teachers/"
ssh -o BatchMode=yes yzhao04-gpu-taihua "cd '$DEST' && source runtime_env.sh && env/bin/python -B tools/verify_gated_teachers.py"
echo '[Llama staging] Download, transfer and target SHA256 verification complete; token retained on Shanghai only'
