#!/usr/bin/env bash
# Source after defining ROOT. Site-specific configuration stays outside Git.
[[ -z ${AOTD_ENV_FILE:-} ]] || source "$AOTD_ENV_FILE"
HOST=${AOTD_WORKER_HOST:?Set AOTD_WORKER_HOST to the SSH destination}
DEST=${AOTD_WORKER_ROOT:?Set AOTD_WORKER_ROOT to the absolute persistent project directory}
[[ "$DEST" =~ ^/[A-Za-z0-9_./-]+$ && "$DEST" != / && "$DEST" != *'/../'* && "$DEST" != */.. ]] || {
    echo 'Invalid absolute worker project directory' >&2; exit 2;
}
REMOTE_RSYNC=${AOTD_REMOTE_RSYNC:-env LD_LIBRARY_PATH=$DEST/downloads/transfer-tools $DEST/downloads/transfer-tools/rsync}
mkdir -p "$ROOT/logs"
