#!/usr/bin/env bash
set -euo pipefail
ROOT=${AOTD_ROOT:?}
RUN_DIR=${AOTD_RUN_DIR:?}
PY="$ROOT/env/bin/python"
MODE=${1:-all}
stage=starting
heartbeat() {
    while true; do
        sleep 15
        printf '[progress %s] task running; GPU ' "$(TZ=Asia/Shanghai date '+%H:%M:%S')"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';'
        printf '\n'
    done
}
heartbeat &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true; wait "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
echo '[1/5] GPU environment and BF16 preflight'
CUDA_VISIBLE_DEVICES="$AOTD_WORK_GPUS" "$PY" -B "$ROOT/tools/gpu_preflight.py"
if [[ "$MODE" != train-only ]]; then
    echo '[2/5] Author checkpoint: NExT-QA val, 2 real questions'
    CUDA_VISIBLE_DEVICES=${AOTD_EVAL_GPU:-0} "$PY" -u "$ROOT/evaluation/eval_nextqa.py" \
        --model_path "$ROOT/models/AoTD-7B-author" --data_dir "$ROOT/data/nextqa/val.csv" \
        --video_dir "$ROOT/data/nextqa/NExTVideo" --split val --limit 2 \
        --attn_implementation sdpa --save_dir "$RUN_DIR/author_val_smoke.json"
fi
if [[ "$MODE" != eval-only ]]; then
    echo '[3/5] Prepare TRAIN QA samples (plumbing smoke only)'
    "$PY" -B "$ROOT/tools/prepare_train_smoke.py"
    echo '[4/5] Base checkpoint: 4 GPUs, ZeRO-2, 2 training steps; save model + processor'
    GPUS=${AOTD_TRAIN_GPUS:-0,1,2,3}
    IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
    CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#GPU_ARRAY[@]}" \
        "$ROOT/distillation/train.py" \
        --model_path "$ROOT/models/LLaVA-7B-base" --data_yaml "$ROOT/data/train_smoke/instruct.yaml" \
        --output_dir "$RUN_DIR/train_checkpoint" \
        --deepspeed "$ROOT/distillation/configs/ds_config_zero2.json" --optim adamw_torch \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 2 --seed 42 \
        --attn_implementation sdpa --dataloader_num_workers 0 --save_strategy no
    echo '[5/5] Reload saved checkpoint + processor: NExT-QA val, 1 question'
    CUDA_VISIBLE_DEVICES=${AOTD_EVAL_GPU:-0} "$PY" -u "$ROOT/evaluation/eval_nextqa.py" \
        --model_path "$RUN_DIR/train_checkpoint" --data_dir "$ROOT/data/nextqa/val.csv" \
        --video_dir "$ROOT/data/nextqa/NExTVideo" --split val --limit 1 \
        --attn_implementation sdpa --save_dir "$RUN_DIR/trained_val_smoke.json"
fi
"$PY" -B - "$RUN_DIR" "$MODE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);mode=sys.argv[2]
result={'status':'PASS','mode':mode,'scope':'author eval 2 samples; QA-only training plumbing smoke; not six-source AoTD reproduction'}
if mode != 'train-only':
 x=json.loads((p/'author_val_smoke.json').read_text());assert x['summary']['total']==2
 result['author_eval']=x['summary']
if mode != 'eval-only':
 x=json.loads((p/'train_checkpoint/training_metrics.json').read_text())
 assert x['finite_loss'] and x['global_step']==2
 assert x['runtime_backend']['zero_stage']==2 and x['runtime_backend']['bfloat16_enabled']
 result['training']=x
 x=json.loads((p/'trained_val_smoke.json').read_text());assert x['summary']['total']==1
 result['checkpoint_reload_eval']=x['summary']
(p/'smoke_summary.json').write_text(json.dumps(result,indent=2))
print('[smoke] PASS: '+str(p/'smoke_summary.json'))
PY
