#!/usr/bin/env bash
# Attach an existing mounted-volume artifact bundle without moving or modifying it.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE=${1:-}
REUSE=${2:-}
if [[ -z "$SOURCE" || "$SOURCE" == --help || "$SOURCE" == -h ]]; then
    echo "Usage: bash $0 /absolute/path/to/existing/aotd-nextqa [--reuse-env]"
    echo 'Links models/data and downloaded artifacts into this checkout; never overwrites existing paths.'
    exit 0
fi
[[ "$SOURCE" == /* && -d "$SOURCE" && "$SOURCE" != "$ROOT" ]] || { echo 'Provide a different existing absolute asset directory'; exit 2; }
[[ -z "$REUSE" || "$REUSE" == --reuse-env ]] || { echo 'Unknown option'; exit 2; }
for path in "$ROOT" "$SOURCE"; do
    case "$(findmnt -T "$path" -n -o FSTYPE)" in overlay|tmpfs|ramfs|'') echo "Persistent mounted storage required: $path";exit 2;; esac
done
attach() {
    local source=$1 target=$2
    [[ -e "$source" ]] || { echo "[assets] absent: $source";return; }
    if [[ -e "$target" || -L "$target" ]];then
        [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || { echo "Refusing to replace existing path: $target";exit 2; }
    else
        ln -s "$source" "$target"
    fi
}
for directory in models data;do attach "$SOURCE/$directory" "$ROOT/$directory";done
mkdir -p "$ROOT/downloads"
shopt -s nullglob
for path in "$SOURCE"/downloads/*;do
    name=$(basename "$path")
    [[ "$name" == README.md || "$name" == inventory.json ]] && continue
    attach "$path" "$ROOT/downloads/$name"
done
if [[ "$REUSE" == --reuse-env ]];then attach "$SOURCE/env" "$ROOT/env";fi
echo '[assets] Existing artifacts attached; runtime state remains local to this checkout'
