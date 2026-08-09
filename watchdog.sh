#!/bin/bash
# =============================================================================
# DenseTNT training watchdog — restarts a crashed train_v4.py session.
#
# Logic:
#   - "Finish." in the training log (printed once by train_v4.main) + exit 0
#     => training completed normally, watchdog exits.
#   - non-zero exit or exit 0 without "Finish." => crash, clean up and retry.
#   - Up to $MAX_RETRIES restarts with exponential backoff.
#
# Usage:
#   bash watchdog.sh "<train command>" [log_file] [max_retries]
#
# Example:
#   bash watchdog.sh "python src/train_v4.py --do_train --data_dir train/data \
#                     --data_dir_for_val val/data --output_dir model_save_full_chunked \
#                     --train_batch_size 64 --num_train_epochs 16 --patience 5 \
#                     --distributed_training 1 --use_map --use_centerline --argoverse \
#                     --other_params semantic_lane direction l1_loss goals_2D \
#                     enhance_global_graph subdivide goal_scoring laneGCN \
#                     point_sub_graph lane_scoring complete_traj complete_traj-3" \
#                     model_save_full_chunked/training.log 10
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CMD="${1:?Usage: watchdog.sh \"<train command>\" [log_file] [max_retries]}"
LOG_FILE="${2:-model_save_full_chunked/training.log}"
MAX_RETRIES="${3:-10}"

OUTPUT_DIR="model_save_full_chunked"
WD_LOG="$OUTPUT_DIR/watchdog.log"
PID_FILE="$OUTPUT_DIR/watchdog.pid"
BACKOFF_BASE=30

mkdir -p "$OUTPUT_DIR"

log_wd() { echo "[$(date '+%m-%d %H:%M:%S')] WD: $1" | tee -a "$WD_LOG"; }

# Single instance
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log_wd "Another watchdog is running (PID $(cat "$PID_FILE")); exiting."
    exit 0
fi
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

cleanup() {
    # Release the DDP master port (only kill processes whose command line looks
    # like a torch.distributed training process — never blind-kill port users)
    lsof -ti :12355 2>/dev/null | while read -r pid; do
        if ps -p "$pid" -o args= 2>/dev/null | grep -qE "train_v4|torch.distributed|distributed"; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$OUTPUT_DIR"/model_save/checkpoint_intra_latest.pt 2>/dev/null || true
    rm -f "$OUTPUT_DIR"/model_save/checkpoint_intra_e*_s*.pt 2>/dev/null || true
    sleep 1
}

retry=0
while [ "$retry" -lt "$MAX_RETRIES" ]; do
    retry=$((retry + 1))
    log_wd "Attempt $retry/$MAX_RETRIES"
    cleanup

    set +e
    # Truncate (not append) so a stale "Finish." from a previous run can never
    # be mistaken for completion of this attempt; dashboard reads the same file.
    eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if grep -q "Finish." "$LOG_FILE" 2>/dev/null; then
        log_wd "Training finished (Finish. found), exit=$EXIT_CODE"
        exit 0
    fi

    if [ "$EXIT_CODE" -eq 0 ]; then
        log_wd "Train process exited 0 but no Finish. marker — treating as abnormal."
    else
        log_wd "Training crashed (exit=$EXIT_CODE)"
    fi

    if [ "$retry" -ge "$MAX_RETRIES" ]; then
        log_wd "Max retries reached; giving up."
        exit "$EXIT_CODE"
    fi

    BACKOFF=$((BACKOFF_BASE * 2 ** (retry - 1)))
    log_wd "Restarting in ${BACKOFF}s..."
    sleep "$BACKOFF"
done
