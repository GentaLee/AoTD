#!/usr/bin/env bash
set -euo pipefail
ROOT=${AOTD_ROOT:?}
RUN=${AOTD_RUN_DIR:?}
PY="$ROOT/env/bin/python"
echo '[validation only] two optimizer steps with formal batch48; this is not the delivery entrypoint'
"$PY" -B "$ROOT/tools/prepare_train_smoke.py"
"$PY" -B - <<'PY'
import json,os
from pathlib import Path
r=Path(os.environ['AOTD_ROOT']);p=Path(os.environ['AOTD_RUN_DIR'])
rows=json.loads((r/'data/train_smoke/answering.json').read_text())
# 96 controlled QA records ensure each of two optimizer steps actually consumes
# 12 microbatches x 4 ranks. A two-record dataset ends its epoch too early.
(p/'validation_qa.json').write_text(json.dumps(rows*48))
(p/'validation.yaml').write_text('datasets:\n  - json_path: '+json.dumps(str(p/'validation_qa.json'))+'\n')
PY
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$ROOT/distillation/train.py" --model_path "$ROOT/models/LLaVA-7B-base" \
    --data_yaml "$RUN/validation.yaml" --output_dir "$RUN/checkpoint" \
    --deepspeed "$ROOT/distillation/configs/ds_config_zero2.json" --optim adamw_torch \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 12 --max_steps 2 --seed 42 \
    --attn_implementation sdpa --dataloader_num_workers 0 --save_strategy no
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/evaluation/eval_nextqa.py" \
    --model_path "$RUN/checkpoint" --data_dir "$ROOT/data/nextqa/val.csv" \
    --video_dir "$ROOT/data/nextqa/NExTVideo" --split val --limit 1 --attn_implementation sdpa \
    --save_dir "$RUN/checkpoint_reload.json"
"$PY" -B - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['AOTD_RUN_DIR']);m=json.loads((p/'checkpoint/training_metrics.json').read_text())
assert m['global_step']==2 and m['finite_loss'] and m['effective_batch_size']==48
assert m['runtime_backend']['zero_stage']==2 and m['runtime_backend']['world_size']==4
assert not m['model_accepts_loss_kwargs']
assert max(m['step_losses'])<50, 'Accumulation scaling regression on the fixed QA fixture'
(p/'validation_summary.json').write_text(json.dumps({'status':'PASS','formal_batch':48,'training':m},indent=2))
print('[validation] PASS: effective batch48, ZeRO2, finite gradients/loss, checkpoint reload')
PY
