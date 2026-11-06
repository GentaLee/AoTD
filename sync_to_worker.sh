#!/usr/bin/env bash
# Copy an offline artifact bundle from a network-enabled source to a worker.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/tools/worker_transfer_env.sh"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "
    probe='$DEST'
    while [ ! -e \"\$probe\" ]; do probe=\$(dirname \"\$probe\"); done
    storage=\$(findmnt -T \"\$probe\" -n -o FSTYPE)
    case \"\$storage\" in overlay|tmpfs|ramfs|'') echo 'Persistent worker storage required' >&2; exit 2;; esac
    mkdir -p '$DEST/downloads/transfer-tools'"
scp "$ROOT/downloads/transfer-tools/rsync" "$ROOT/downloads/transfer-tools/libpopt.so.0" "$HOST:$DEST/downloads/transfer-tools/"
rsync -aL --partial --info=progress2 \
    --exclude env/ --exclude runtime/ --exclude results/ --exclude tmp/ --exclude cache/ --exclude .cache/ \
    --exclude .git/ --exclude logs/ --exclude '*.safetensors' --exclude teacher-access/ \
    --rsync-path="$REMOTE_RSYNC" "$ROOT/" "$HOST:$DEST/"
transfer_shard() {
    local shard=$1 relative=${1#"$ROOT/"}
    local folder=${relative%/*}
    echo "[transfer] start $relative"
    rsync -aL --partial --append-verify --info=progress2 --rsync-path="$REMOTE_RSYNC" \
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
PY=${AOTD_WORKER_PLATFORM_PYTHON:-python3}
[[ "$PY" =~ ^[A-Za-z0-9_./-]+$ ]] || { echo 'Invalid worker Python executable' >&2; exit 2; }
ssh -o BatchMode=yes "$HOST" "AOTD_PLATFORM_PYTHON='$PY' bash '$DEST/setup_offline.sh' && bash '$DEST/setup_agent_offline.sh'"
echo "[transfer] Offline bundle and environment ready at $HOST:$DEST"
echo "[run on worker] bash '$DEST/run_aotd_full.sh'"
