#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/runtime_env.sh"
cd "$ROOT"
export AOTD_RUN_DIR="$ROOT/results/validation_local_llama_taihua_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$AOTD_RUN_DIR/phase"
exec > >(tee "$AOTD_RUN_DIR/run.log") 2>&1
echo "[local Llama validation] results=$AOTD_RUN_DIR"
"$ROOT/env/bin/python" -B - <<'PY'
import json,time
from pathlib import Path
r=Path.cwd();manifest=json.loads((r/'models/teachers/local_llama_manifest.json').read_text())
start=time.monotonic();files=[x for x in manifest['files'] if x['path'].endswith('.safetensors')]
while time.monotonic()-start<7200:
    done=sum((r/x['path']).stat().st_size for x in files if (r/x['path']).is_file())
    expected=sum(x['bytes'] for x in files)
    print(f'[local Llama transfer progress] {done/expected:.1%}',flush=True)
    if all((r/x['path']).is_file() and (r/x['path']).stat().st_size==x['bytes'] for x in files):break
    time.sleep(30)
else:raise TimeoutError('Local model transfer did not finish within two hours')
PY
"$ROOT/env/bin/python" -B "$ROOT/tools/verify_local_llama_transfer.py"
KEEP=/volume/pt-test/users/zhaoye/gpu-workspace/keep_alive/main.py
"$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" --keepalive-script "$KEEP" --state-dir "$ROOT/runtime/guard-state" -- \
    "$ROOT/env/bin/python" -B "$ROOT/tools/with_spare_keepalive.py" --mode eval-only --keepalive-script "$KEEP" --state-dir "$AOTD_RUN_DIR/phase" -- \
    env CUDA_VISIBLE_DEVICES=0 bash "$ROOT/tools/validate_gated_agent.sh"
LLAMA_RUN="$AOTD_RUN_DIR"
export AOTD_RUN_DIR="$ROOT/results/validation_real_agent_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$AOTD_RUN_DIR"
echo "[real Agent validation] results=$AOTD_RUN_DIR"
"$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" --keepalive-script "$KEEP" --state-dir "$ROOT/runtime/guard-state" -- \
    bash "$ROOT/tools/validate_real_construction.sh" > "$AOTD_RUN_DIR/run.log" 2>&1
echo "[local Llama + real Agent] complete; model tests=$LLAMA_RUN; real construction=$AOTD_RUN_DIR"
