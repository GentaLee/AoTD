#!/usr/bin/env bash
# Source this file; all writable runtime state stays under this project.
AOTD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AOTD_ROOT
if [[ -f "$AOTD_ROOT/runtime/site.env" ]]; then source "$AOTD_ROOT/runtime/site.env"; fi
export AOTD_PROGRESS_INTERVAL_SECONDS=${AOTD_PROGRESS_INTERVAL_SECONDS:-30}
[[ "$AOTD_PROGRESS_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] && ((AOTD_PROGRESS_INTERVAL_SECONDS>=10 && AOTD_PROGRESS_INTERVAL_SECONDS<=60)) || {
    echo 'AOTD_PROGRESS_INTERVAL_SECONDS must be an integer from 10 to 60' >&2; return 2;
}
export AOTD_KEEPALIVE_SCRIPT=${AOTD_KEEPALIVE_SCRIPT:-$AOTD_ROOT/../gpu-workspace/keep_alive/main.py}
export HF_HUB_DISABLE_PROGRESS_BARS=1 DATASETS_DISABLE_PROGRESS_BARS=1
export HOME="$AOTD_ROOT/runtime/home"
export TMPDIR="$AOTD_ROOT/runtime/tmp" TMP="$AOTD_ROOT/runtime/tmp" TEMP="$AOTD_ROOT/runtime/tmp"
export XDG_CACHE_HOME="$AOTD_ROOT/runtime/cache" XDG_CONFIG_HOME="$AOTD_ROOT/runtime/config" XDG_DATA_HOME="$AOTD_ROOT/runtime/data"
export HF_HOME="$AOTD_ROOT/runtime/cache/huggingface" HF_DATASETS_CACHE="$AOTD_ROOT/runtime/cache/datasets"
export TORCH_HOME="$AOTD_ROOT/runtime/cache/torch" TORCH_EXTENSIONS_DIR="$AOTD_ROOT/runtime/cache/torch_extensions"
export TRITON_CACHE_DIR="$AOTD_ROOT/runtime/cache/triton" TORCHINDUCTOR_CACHE_DIR="$AOTD_ROOT/runtime/cache/inductor"
export CUDA_CACHE_PATH="$AOTD_ROOT/runtime/cache/cuda" PYTHONPYCACHEPREFIX="$AOTD_ROOT/runtime/cache/pycache"
export PIP_CACHE_DIR="$AOTD_ROOT/runtime/cache/pip" UV_CACHE_DIR="$AOTD_ROOT/runtime/cache/uv"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DS_BUILD_OPS=0 DS_ACCELERATOR=cuda
unset PIP_CONSTRAINT PYTHONPATH
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$AOTD_ROOT/logs" "$AOTD_ROOT/results"
