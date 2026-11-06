#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "$(findmnt -T "$ROOT" -n -o FSTYPE)" in overlay|tmpfs|ramfs|'') echo 'ERROR: project must reside on the persistent mounted volume' >&2; exit 1;; esac
source "$ROOT/runtime_env.sh"
exec 9>"$ROOT/runtime/setup.lock"
flock -n 9 || { echo 'Another environment setup is running' >&2; exit 1; }
PY=${AOTD_PLATFORM_PYTHON:-/opt/venv/bin/python}
echo '[setup 1/5] Create isolated Python environment on the mounted volume'
if [[ ! -x "$ROOT/env/bin/python" ]]; then
    "$PY" -m venv "$ROOT/env"
fi
SITE=$("$ROOT/env/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo '[setup 2/5] Reuse the platform Blackwell PyTorch binaries (read only)'
"$PY" -B "$ROOT/tools/link_platform_torch.py" "$SITE" > "$ROOT/logs/platform_torch_links.json"
echo '[setup 3/5] Install pinned dependencies from local wheelhouse'
"$ROOT/env/bin/python" -m pip install --no-index --find-links "$ROOT/downloads/wheels" -r "$ROOT/requirements-runtime.txt"
echo '[setup 4/5] Install DeepSpeed without compiled extensions (offline)'
"$ROOT/env/bin/python" -m pip install --no-index --find-links "$ROOT/downloads/wheels" deepspeed==0.15.4
echo '[setup 5/5] Check dependencies and imports'
"$ROOT/env/bin/python" -m pip check
"$ROOT/env/bin/python" -B - <<'PY'
import torch, torchvision, transformers, datasets, decord, deepspeed
assert transformers.__version__ == '4.49.0'
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
print('transformers', transformers.__version__, 'deepspeed', deepspeed.__version__)
print('Offline environment ready')
PY
"$ROOT/env/bin/python" -m pip freeze > "$ROOT/logs/pip-freeze.txt"
touch "$ROOT/runtime/environment-ready"
echo '[setup] COMPLETE'
