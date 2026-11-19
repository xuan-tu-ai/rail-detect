#!/usr/bin/env bash
# Long training runs: rail seg (s, m) and obstacle det (s, m), sequential.
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-150}
IMGSZ=${IMGSZ:-1024}
BATCH=${BATCH:-8}
for spec in "seg yolo11s-seg.pt" "seg yolo11m-seg.pt" "det yolo11s.pt" "det yolo11m.pt"; do
  set -- $spec
  name="$1_${2%.pt}_${IMGSZ}"
  start=$(date +%s)
  .venv/bin/python scripts/05_train.py --task "$1" --model "$2" --epochs "$EPOCHS" --imgsz "$IMGSZ" \
    --batch "$BATCH" --patience 40 --name "$name" > "runs/$name.log" 2>&1
  echo "$name done in $(( $(date +%s) - start ))s"
done
