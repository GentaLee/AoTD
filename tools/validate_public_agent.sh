#!/usr/bin/env bash
set -euo pipefail
"$AOTD_ROOT/env/bin/python" -B "$AOTD_ROOT/tools/gpu_preflight.py"
exec "$AOTD_ROOT/env/bin/python" -B "$AOTD_ROOT/tools/validate_public_agent.py" "$@"
