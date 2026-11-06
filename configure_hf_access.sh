#!/usr/bin/env bash
# Run interactively on network-enabled source host. Never copy this credential to offline worker.
set +x
set -euo pipefail
[[ -z ${AOTD_ENV_FILE:-} ]] || source "$AOTD_ENV_FILE"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AOTD_HF_DOWNLOAD_ROOT="$ROOT/downloads/teacher-access"
umask 077
MODE=${1:-setup}
case "$MODE" in
    --help|-h)
        echo "Usage: bash $0 [--check]"
        echo 'Default: read a hidden token, authenticate, save with mode 600, and check both original Llama teacher models.'
        echo '--check: recheck the saved credential after model access is granted.'
        exit 0;;
    setup|--check) ;;
    *) echo 'Unknown option; use --help' >&2; exit 2;;
esac
AUTH_DIR=${AOTD_HF_AUTH_DIR:?Set AOTD_HF_AUTH_DIR to a private persistent directory outside the checkout}
case "$AUTH_DIR" in
    /*) ;;
    *) echo 'The credential directory must be an absolute path' >&2; exit 2;;
esac
AUTH_DIR=$(realpath -m -- "$AUTH_DIR")
case "$AUTH_DIR" in
    "$ROOT"|"$ROOT"/*) echo 'Keep credentials outside the checkout and transfer bundle' >&2; exit 2;;
esac
PROBE="$AUTH_DIR"
while [[ ! -e "$PROBE" ]]; do PROBE=$(dirname "$PROBE"); done
case "$(findmnt -T "$PROBE" -n -o FSTYPE)" in
    overlay|tmpfs|ramfs|'') echo 'Persistent credential storage required' >&2; exit 2;;
esac
mkdir -p "$AUTH_DIR"
TOKEN_FILE="$AUTH_DIR/token"
chmod 700 "$AUTH_DIR"
TEMP_TOKEN=''
cleanup() { [[ -z "$TEMP_TOKEN" ]] || rm -f -- "$TEMP_TOKEN"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if [[ "$MODE" == --check ]]; then
    [[ -s "$TOKEN_FILE" ]] || { echo 'No saved token. Run this script without --check first.' >&2; exit 2; }
    INPUT="$TOKEN_FILE"
else
    [[ -t 0 ]] || { echo 'Run in an interactive network-enabled source host terminal; token input is hidden.' >&2; exit 2; }
    echo 'Enter a Hugging Face read token for an account authorized for both Llama models.'
    IFS= read -r -s -p 'HF token (hidden): ' HF_INPUT
    printf '\n'
    [[ "$HF_INPUT" == hf_* && ${#HF_INPUT} -gt 10 ]] || { unset HF_INPUT; echo 'Invalid token format.' >&2; exit 2; }
    TEMP_TOKEN=$(mktemp "$AUTH_DIR/.incoming.XXXXXXXX")
    printf '%s' "$HF_INPUT" > "$TEMP_TOKEN"
    unset HF_INPUT
    INPUT="$TEMP_TOKEN"
fi
export AOTD_HF_SETUP_INPUT="$INPUT" AOTD_HF_AUTH_DIR="$AUTH_DIR"
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
"${AOTD_DOWNLOAD_PYTHON:-python3}" -B <<'PY'
import json
import os
from pathlib import Path
import sys
from huggingface_hub import HfApi,hf_hub_download

source=Path(os.environ['AOTD_HF_SETUP_INPUT'])
directory=Path(os.environ['AOTD_HF_AUTH_DIR'])
destination=directory/'token'
token=source.read_text().strip()
api=HfApi(token=token)
try:
    api.whoami(token=token)
except Exception as exc:
    status=getattr(getattr(exc,'response',None),'status_code',None)
    print(f'Authentication failed: {type(exc).__name__}, HTTP={status}. Token was not printed or saved.',flush=True)
    sys.exit(1)
if source!=destination:
    source.chmod(0o600)
    source.replace(destination)
destination.chmod(0o600)
print(f'Account authentication passed. Credential saved privately: {destination}',flush=True)
results=[]
for repo in ['meta-llama/Meta-Llama-3-8B-Instruct','meta-llama/Llama-3.1-8B-Instruct']:
    row={'repo':repo}
    try:
        info=api.model_info(repo,token=token)
        hf_hub_download(repo,'config.json',revision=info.sha,token=token,
            local_dir=Path(os.environ['AOTD_HF_DOWNLOAD_ROOT'])/repo.replace('/','--'))
        row.update(access=True,revision=info.sha)
        print(f'ACCESS OK: {repo}@{info.sha}',flush=True)
    except Exception as exc:
        status=getattr(getattr(exc,'response',None),'status_code',None)
        row.update(access=False,error_type=type(exc).__name__,http_status=status)
        print(f'ACCESS FAILED: {repo}: {type(exc).__name__}, HTTP={status}',flush=True)
        print(f'Model access page: https://huggingface.co/{repo}',flush=True)
    results.append(row)
(directory/'access_check.json').write_text(json.dumps({'models':results,'all_accessible':all(r['access'] for r in results)},indent=2))
(directory/'access_check.json').chmod(0o600)
if not all(r['access'] for r in results):
    print('Token saved, but model access is incomplete. After access is granted, rerun with --check.',flush=True)
    sys.exit(3)
print('READY: both original Llama teacher models are accessible. No credential was copied to offline worker.',flush=True)
PY
