#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/runtime_env.sh"
"$ROOT/env/bin/python" -B -m pip install --no-index --no-deps --find-links "$ROOT/downloads/agent-wheels" -r "$ROOT/requirements-agent.txt"
"$ROOT/env/bin/python" -B -m pip check
"$ROOT/env/bin/python" -B - <<'PY'
import cv2,einops,ftfy,ffmpeg,h5py,ijson,omegaconf,rich,scipy,timm,tensorboardX
print('[Agent dependencies] Offline imports PASS')
PY
