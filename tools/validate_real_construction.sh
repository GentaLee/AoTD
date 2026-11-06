#!/usr/bin/env bash
# Four real train questions in an isolated validation directory; no full-data writes.
set -euo pipefail
ROOT=${AOTD_ROOT:?}
RUN=${AOTD_RUN_DIR:?}
PY="$ROOT/env/bin/python"
QA="$ROOT/data/construction/NExT-QA/qa_normalized.jsonl"
OUT="$RUN/agent_data"
mkdir -p "$OUT"
KEEP=/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py
for stage in construct transfer filter;do
    MODE=eval-only
    GPUS=0
    if [[ "$stage" == construct ]];then MODE=all;GPUS=0,1,2,3;fi
    mkdir -p "$RUN/$stage"
    "$PY" -B "$ROOT/tools/with_spare_keepalive.py" --mode "$MODE" --keepalive-script "$KEEP" --state-dir "$RUN/$stage" -- \
        env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -B "$ROOT/tools/build_cot.py" --stage "$stage" --qa "$QA" --output-dir "$OUT" --limit 4
done
"$PY" -B "$ROOT/tools/build_cot.py" --stage export --qa "$QA" --output-dir "$OUT"
"$PY" -B - <<'PY'
import json,os
from pathlib import Path
r=Path(os.environ['AOTD_RUN_DIR']);p=r/'agent_data'
summary={name:json.loads((p/f'{name}.summary.json').read_text()) for name in ['construct','transfer','filter']}
if summary['filter']['accepted']<1:raise RuntimeError('No accepted real CoT in the validation subset; inspect traces before extending validation')
assert all(item['statuses'].get('error',0)==0 for item in summary.values())
summary.update(status='PASS',validation_only=True,real_train_questions=4)
(r/'real_construction_validation.json').write_text(json.dumps(summary,indent=2))
print('[real Agent construction] PASS:',r/'real_construction_validation.json')
PY
