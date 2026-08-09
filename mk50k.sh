#!/bin/bash
# Create a subset of the training data via symlinks (useful for quick experiments
# on machines with limited resources, e.g. 50k of 205k samples).
#
# Usage:
#   bash mk50k.sh <source_train_dir> <subset_dir> [count]
#
# Example:
#   bash mk50k.sh train/data train/data_50k 50000
#
# Requires a Linux environment with symlink support (WSL recommended).
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_train_dir> <subset_dir> [count]" >&2
    echo "Example: $0 train/data train/data_50k 50000" >&2
    exit 1
fi

TRAIN="$1"
SUBSET="$2"
COUNT="${3:-50000}"

if [ ! -d "$TRAIN" ]; then
    echo "Error: source dir not found: $TRAIN" >&2
    exit 1
fi

case "$COUNT" in
    ''|*[!0-9]*)
        echo "Error: count must be a non-negative integer: $COUNT" >&2
        exit 1
        ;;
esac

# --- Safety checks before any deletion ---
if [ -z "$SUBSET" ] || [ "$SUBSET" = "/" ] || [ "$SUBSET" = "." ] || [ "$SUBSET" = ".." ]; then
    echo "Error: refusing to use unsafe subset dir: $SUBSET" >&2
    exit 1
fi

# Resolve both paths to absolute form for containment checks
# 父目录必须存在，否则 cd 失败会导致 SUBSET_ABS 为空串，进而误报
# "subset contains source"（见 review L8）
SUBSET_PARENT="$(dirname "$SUBSET")"
if [ ! -d "$SUBSET_PARENT" ]; then
    echo "Error: parent directory of subset does not exist: $SUBSET_PARENT" >&2
    exit 1
fi
TRAIN_ABS="$(cd "$(dirname "$TRAIN")" && pwd)/$(basename "$TRAIN")"
SUBSET_ABS="$(cd "$SUBSET_PARENT" && pwd)/$(basename "$SUBSET")"
if [ "$SUBSET_ABS" = "$TRAIN_ABS" ] || [ "${TRAIN_ABS#$SUBSET_ABS/}" != "$TRAIN_ABS" ]; then
    echo "Error: subset dir '$SUBSET' is the source dir or contains it (would delete source data)" >&2
    exit 1
fi
if [ "${SUBSET_ABS#$TRAIN_ABS/}" != "$SUBSET_ABS" ]; then
    echo "Error: subset dir '$SUBSET' is inside the source dir (rm -rf would delete it before use)" >&2
    exit 1
fi

rm -rf "$SUBSET"
mkdir -p "$SUBSET"

# Symlink targets must be absolute: a relative target is resolved from the
# link's own directory, so it would point at $SUBSET/train/data/... (broken).
shopt -s nullglob
count=0
for f in "$TRAIN_ABS"/*.csv; do
    if [ $count -ge "$COUNT" ]; then break; fi
    ln -s "$f" "$SUBSET/$(basename "$f")"
    count=$((count + 1))
done
if [ "$count" -eq 0 ]; then
    echo "Warning: no .csv files found in $TRAIN; created no symlinks" >&2
else
    echo "Created $count symlinks in $SUBSET"
fi
