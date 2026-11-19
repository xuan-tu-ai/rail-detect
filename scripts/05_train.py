"""Train a YOLO11 model on data/yolo_seg or data/yolo_det.

Usage:
  .venv/bin/python scripts/05_train.py --task seg --model yolo11s-seg.pt --epochs 100 --imgsz 1024 --batch 8
  .venv/bin/python scripts/05_train.py --task det --model yolo11m.pt --epochs 3 --name smoke_det_m
Pretrained weights are downloaded into weights/ on first use.
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, ROOT, RUNS  # noqa: E402

DATASETS = {"seg": DATA / "yolo_seg" / "data.yaml", "det": DATA / "yolo_det" / "data.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg", "det"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    weights_dir = ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    os.chdir(weights_dir)  # Ultralytics downloads pretrained weights into cwd
    model = YOLO(args.model)
    name = args.name or f"{args.task}_{Path(args.model).stem}_{args.imgsz}"
    model.train(
        data=str(DATASETS[args.task]),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(RUNS / args.task),
        name=name,
        exist_ok=True,
        seed=0,
        deterministic=False,
        fraction=args.fraction,
        cos_lr=True,
        close_mosaic=10,
        # top-down aerial views: vertical flips are as valid as horizontal ones
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
