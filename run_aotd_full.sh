#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=all
CHECK=0
SELECT_ARGS=()
while (($#)); do
    case "$1" in
        --check) CHECK=1; shift;;
        --eval-only) MODE=eval-only; shift;;
        --resume) SELECT_ARGS+=(--resume "${2:?Missing run directory}"); shift 2;;
        --new-run) SELECT_ARGS+=(--new-run); shift;;
        --help|-h)
            echo "Usage: $0 [--check] [--eval-only] [--resume RUN_DIR|--new-run]"
            echo 'Default: reuse the latest managed run and resume compatible construction, training checkpoints and per-question evaluation.'
            echo 'AOTD_SAVE_STEPS defaults to 100; keep the last two complete resumable training checkpoints.'
            echo 'Default: independently train Instruct and AoTD from Base for one full epoch, then evaluate author/base/instruct/aotd on the complete NExT-QA split.'
            echo 'Four GPUs, ZeRO-2, BF16, microbatch 1, accumulation 12, effective batch 48. No smoke step/sample limits.'
            echo 'Training: four work GPUs + four keep-alive GPUs. Evaluation: one work GPU + seven keep-alive GPUs.'
            echo '--check: validate inputs only; do not launch GPU workloads.'
            echo '--eval-only: full author and base evaluation, without training.'
            echo 'Inputs: AOTD_TRAIN_DATA (instruct.yaml/aotd.yaml and complete preprocessed QA/CoT/media), AOTD_VIDEO_DIR (ID.mp4 videos).'
            echo 'If normalized AOTD_QA_SOURCE/AOTD_COT_SOURCE files exist, training records and YAML are prepared automatically.'
            echo 'AOTD_EVAL_SPLIT defaults to val (4996 questions); test has 8564 questions.'
            echo "Outputs: $ROOT/results/full_<UTC timestamp>_<PID>/; run.log, checkpoints, per-question JSONL, metrics, full_summary.json."
            echo 'If training records are absent, the default run first invokes the offline five-source data construction entrypoint.'
            echo 'Offline runtime; no Markdown progress writes. --check never launches Agent inference.'
            exit 0;;
        *) echo "Unknown option $1" >&2; exit 2;;
    esac
done
source "$ROOT/runtime_env.sh"
case "$(findmnt -T "$ROOT" -n -o FSTYPE)" in overlay|tmpfs|ramfs|'') echo 'ERROR: persistent mounted storage required'; exit 2;; esac
[[ -x "$ROOT/env/bin/python" ]] || { echo "Offline environment missing: run $ROOT/setup_offline.sh"; exit 2; }
export AOTD_FULL_MODE="$MODE"
export AOTD_EVAL_SPLIT=${AOTD_EVAL_SPLIT:-val}
[[ "$AOTD_EVAL_SPLIT" == val || "$AOTD_EVAL_SPLIT" == test ]] || exit 2
exec 9>"$ROOT/runtime/full-run.lock"
flock -n 9 || { echo 'Another full invocation already holds this project run lock'; exit 2; }
((CHECK==0)) || SELECT_ARGS+=(--check)
AOTD_RUN_DIR=$("$ROOT/env/bin/python" -B "$ROOT/tools/full_run_state.py" select --root "$ROOT" --mode "$MODE" "${SELECT_ARGS[@]}")
export AOTD_RUN_DIR
mkdir -p "$AOTD_RUN_DIR"
exec > >(tee -a "$AOTD_RUN_DIR/run.log") 2>&1
finish() {
    local code=$?
    trap - EXIT
    local status=failed
    if ((code==129 || code==130 || code==143)); then status=interrupted; fi
    if ((code==0)); then status=complete; ((CHECK==0)) || status=checked; fi
    "$ROOT/env/bin/python" -B "$ROOT/tools/full_run_state.py" record --run "$AOTD_RUN_DIR" --status "$status" --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
"$ROOT/env/bin/python" -B "$ROOT/tools/full_run_state.py" record --run "$AOTD_RUN_DIR" --status running --exit-code 0
echo "[full] results=$AOTD_RUN_DIR mode=$MODE"
TRAIN_DATA=${AOTD_TRAIN_DATA:-$ROOT/data/five_source_train}
QA_SOURCE=${AOTD_QA_SOURCE:-$ROOT/data/five_source_sources/qa_normalized.json}
COT_SOURCE=${AOTD_COT_SOURCE:-$ROOT/data/five_source_sources/cot_filtered.json}
if [[ ! -f "$QA_SOURCE" && -f "${QA_SOURCE}l" ]]; then QA_SOURCE="${QA_SOURCE}l"; fi
if [[ ! -f "$COT_SOURCE" && -f "${COT_SOURCE}l" ]]; then COT_SOURCE="${COT_SOURCE}l"; fi
if [[ "$MODE" == all && "$CHECK" == 0 && ( ! -f "$TRAIN_DATA/instruct.yaml" || ! -f "$TRAIN_DATA/aotd.yaml" ) && ( ! -f "$QA_SOURCE" || ! -f "$COT_SOURCE" ) ]]; then
    echo '[full] Training inputs absent; preparing five-source QA and Agent CoT before student training'
    bash "$ROOT/run_aotd_build_data.sh"
    QA_SOURCE=${AOTD_QA_SOURCE:-$ROOT/data/five_source_sources/qa_normalized.jsonl}
    COT_SOURCE=${AOTD_COT_SOURCE:-$ROOT/data/five_source_sources/cot_filtered.jsonl}
fi
if [[ "$MODE" == all && "$CHECK" == 0 && -f "$QA_SOURCE" && -f "$COT_SOURCE" ]]; then
    "$ROOT/env/bin/python" -B "$ROOT/tools/prepare_full_train.py" --qa "$QA_SOURCE" --cot "$COT_SOURCE" --output-dir "$TRAIN_DATA" --full-pool
fi
echo '[full] Checking all training records and all evaluation media before pausing keep-alive'
"$ROOT/env/bin/python" -B "$ROOT/tools/full_preflight.py"
((CHECK==0)) || exit 0
"$ROOT/env/bin/python" -B "$ROOT/tools/verify_assets.py"
KEEP=${AOTD_KEEPALIVE_SCRIPT:-$ROOT/runtime/keep_alive/main.py}
"$ROOT/env/bin/python" -B "$ROOT/tools/keepalive_guard.py" --keepalive-script "$KEEP" --state-dir "$ROOT/runtime/guard-state" -- \
    bash "$ROOT/tools/full_pipeline.sh"
echo "[full] complete; original keep-alive restored; results=$AOTD_RUN_DIR"
