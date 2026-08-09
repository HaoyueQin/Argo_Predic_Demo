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

rm -rf "$SUBSET"
mkdir -p "$SUBSET"

count=0
for f in "$TRAIN"/*.csv; do
    if [ $count -ge "$COUNT" ]; then break; fi
    ln -s "$f" "$SUBSET/$(basename "$f")"
    count=$((count + 1))
done
echo "Created $count symlinks in $SUBSET"
