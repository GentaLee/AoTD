#!/usr/bin/env bash
# Run on Shanghai only. Requires permission to use the Taihua mounted volume.
set -euo pipefail
source /volume/yzhao04/persist/recovery-20261022/env.sh
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST=${AOTD_TAIHUA_ROOT:-/volume/pt-test/users/zhaoye/aotd-nextqa}
HOST=${AOTD_TAIHUA_HOST:-yzhao04-gpu-taihua}
mkdir -p "$ROOT/logs"
[[ "$DEST" =~ ^/volume/pt-test/users/zhaoye/[A-Za-z0-9_/-]+$ ]] || { echo 'Invalid destination'; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "test \"\$(findmnt -T /volume/pt-test/users/zhaoye -n -o FSTYPE)\" = gpfs && mkdir -p '$DEST/downloads/transfer-tools'"
scp "$ROOT/downloads/transfer-tools/rsync" "$ROOT/downloads/transfer-tools/libpopt.so.0" "$HOST:$DEST/downloads/transfer-tools/"
rsync -aL --partial --info=progress2 \
    --exclude env/ --exclude runtime/ --exclude results/ --exclude tmp/ --exclude cache/ --exclude .cache/ \
    --exclude .git/ --exclude logs/ --exclude '*.safetensors' --exclude teacher-access/ \
    --rsync-path="env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync" \
    "$ROOT/" "$HOST:$DEST/"
transfer_shard() {
    local shard=$1 relative=${1#"$ROOT/"}
    local folder=${relative%/*}
    echo "[transfer] start $relative"
    rsync -a --partial --append-verify --info=progress2 \
        --rsync-path="env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync" \
        "$shard" "$HOST:$DEST/$folder/" > "$ROOT/logs/transfer_$(basename "$folder")_$(basename "$shard").log" 2>&1
    echo "[transfer] complete $relative"
}
pids=()
while IFS= read -r -d '' shard; do
    transfer_shard "$shard" &
    pids+=("$!")
    if ((${#pids[@]} == 4)); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
    fi
done < <(rg --files --follow --hidden "$ROOT/models" -g '*.safetensors' -g '!**/.cache/**' -0)
for pid in "${pids[@]}"; do wait "$pid"; done
ssh -o BatchMode=yes "$HOST" "bash '$DEST/setup_offline.sh'"
if [[ "$DEST" == /volume/pt-test/users/zhaoye/aotd-nextqa ]]; then
    scp "$ROOT/taihua_entrypoint.sh" "$HOST:/volume/pt-test/users/zhaoye/run_aotd_smoke.sh"
    scp "$ROOT/taihua_full_entrypoint.sh" "$HOST:/volume/pt-test/users/zhaoye/run_aotd_full.sh"
    scp "$ROOT/taihua_build_entrypoint.sh" "$HOST:/volume/pt-test/users/zhaoye/run_aotd_build_data.sh"
    ssh -o BatchMode=yes "$HOST" 'chmod +x /volume/pt-test/users/zhaoye/run_aotd_{smoke,full,build_data}.sh'
fi
echo "[transfer] Offline bundle + environment ready at $HOST:$DEST"
