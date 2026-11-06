#!/usr/bin/env bash
set -euo pipefail
ROOT=${AOTD_ROOT:?}
RUN=${AOTD_RUN_DIR:?}
PY="$ROOT/env/bin/python"
STAGE=${1:-orchestrate}
if [[ "$STAGE" == orchestrate ]]; then
    KEEP=${AOTD_KEEPALIVE_SCRIPT:-/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py}
    if [[ ${AOTD_FULL_MODE:-all} == all ]]; then
        mkdir -p "$RUN/train_phase"
        "$PY" -B "$ROOT/tools/with_spare_keepalive.py" --mode all --keepalive-script "$KEEP" \
            --state-dir "$RUN/train_phase" -- bash "${BASH_SOURCE[0]}" --train-stage
    fi
    mkdir -p "$RUN/eval_phase"
    "$PY" -B "$ROOT/tools/with_spare_keepalive.py" --mode eval-only --keepalive-script "$KEEP" \
        --state-dir "$RUN/eval_phase" -- bash "${BASH_SOURCE[0]}" --eval-stage
    exit 0
fi
heartbeat() { while sleep 15; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done; }
heartbeat &
HB=$!
trap 'kill "$HB" 2>/dev/null || true; wait "$HB" 2>/dev/null || true' EXIT
GPUS=${AOTD_TRAIN_GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
N=${#GPU_ARRAY[@]}
((48%N==0)) || { echo 'GPU count must divide effective batch 48'; exit 2; }
ACC=$((48/N))
if [[ "$STAGE" == --train-stage ]]; then
    for mode in instruct aotd; do
        echo "[full training] $mode: one epoch, GPUs=$GPUS microbatch=1 accumulation=$ACC effective_batch=48"
        CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N" \
            "$ROOT/distillation/train.py" \
            --model_path "$ROOT/models/LLaVA-7B-base" --data_yaml "$RUN/$mode.yaml" \
            --output_dir "$RUN/${mode}_s42" --deepspeed "$ROOT/distillation/configs/ds_config_zero2.json" \
            --optim adamw_torch --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACC" \
            --attn_implementation sdpa --dataloader_num_workers 0 --seed 42 --save_strategy epoch
    done
    exit 0
fi
[[ "$STAGE" == --eval-stage ]] || { echo "Invalid internal stage: $STAGE"; exit 2; }
SPLIT=${AOTD_EVAL_SPLIT:-val}
VIDEO=${AOTD_VIDEO_DIR:-$ROOT/data/nextqa/NExTVideo}
for tag in author base instruct aotd; do
    case "$tag" in
        author) MODEL="$ROOT/models/AoTD-7B-author";;
        base) MODEL="$ROOT/models/LLaVA-7B-base";;
        *) [[ ${AOTD_FULL_MODE:-all} == all ]] || continue; MODEL="$RUN/${tag}_s42";;
    esac
    echo "[full evaluation] $tag $SPLIT: every question"
    CUDA_VISIBLE_DEVICES=${AOTD_EVAL_GPU:-0} "$PY" -u "$ROOT/evaluation/eval_nextqa.py" \
        --model_path "$MODEL" --data_dir "$ROOT/data/nextqa/$SPLIT.csv" --video_dir "$VIDEO" \
        --split "$SPLIT" --attn_implementation sdpa --save_dir "$RUN/${tag}_${SPLIT}_full.json"
done
"$PY" -B - <<'PY'
import os,json
from pathlib import Path
p=Path(os.environ['AOTD_RUN_DIR']);split=os.environ.get('AOTD_EVAL_SPLIT','val')
tags=['author','base','instruct','aotd'] if os.environ.get('AOTD_FULL_MODE','all')=='all' else ['author','base']
r={}
for tag in tags:
 x=json.loads((p/f'{tag}_{split}_full.json').read_text())
 assert x['meta']['full_dataset'] and x['summary']['total']==x['meta']['dataset_size']
 r[tag]=x['summary']
(p/'full_summary.json').write_text(json.dumps({'status':'PASS','split':split,'results':r},indent=2))
print('[full] PASS:',p/'full_summary.json')
PY
