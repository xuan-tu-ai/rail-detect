"""Build Ultralytics datasets from the splits.

data/yolo_seg: V1 frames, classes rail_inside / rail_lines, polygons from external contours
               (holes left by obstacles are filled: the target is the full track area).
data/yolo_det: the same V1 frames, 6 obstacle classes, boxes consolidated from V2 (data/splits/v1_boxes.csv).
Images are symlinked; both datasets share one split. Usage: .venv/bin/python scripts/04_build_yolo.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OBST_CLASSES, RAIL_CLASSES, SPLITS, v1_image, v1_mask  # noqa: E402

SEG = DATA / "yolo_seg"
DET = DATA / "yolo_det"
MIN_AREA = 400  # px^2 at 1920x1080


def filled(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c for c in cs if cv2.contourArea(c) >= MIN_AREA], -1, 1, -1)
    return out


def polygons(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        p = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        if len(p) >= 3:
            out.append(p)
    return out


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_seg(v1):
    if SEG.exists():
        shutil.rmtree(SEG)
    gt = SEG / "gt_masks"
    for _, r in v1.iterrows():
        i, s = int(r.v1_id), r.split
        link(v1_image(i), SEG / "images" / s / f"{i}.jpg")
        lines = []
        for k, c in enumerate(RAIL_CLASSES):
            m = v1_mask(i, c)
            # evaluation targets: hole-filled masks, stored as PNG
            (gt / c).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gt / c / f"{i}.png"), filled(m) * 255)
            for p in polygons(m):
                xy = (p / [1920, 1080]).clip(0, 1).ravel()
                lines.append(f"{k} " + " ".join(f"{v:.5f}" for v in xy))
        lp = SEG / "labels" / s / f"{i}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines) + "\n")
    yaml.safe_dump({"path": str(SEG), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": dict(enumerate(RAIL_CLASSES))}, open(SEG / "data.yaml", "w"), sort_keys=False)
    print("seg:", v1.split.value_counts().to_dict())


def build_det(v1, boxes):
    if DET.exists():
        shutil.rmtree(DET)
    cls_idx = {c: k for k, c in enumerate(OBST_CLASSES)}
    for _, r in v1.iterrows():
        i, s = int(r.v1_id), r.split
        link(v1_image(i), DET / "images" / s / f"{i}.jpg")
        lines = []
        for b in boxes[boxes.v1_id == i].itertuples():
            cx, cy, bw, bh = (b.x1 + b.x2) / 2 / 1920, (b.y1 + b.y2) / 2 / 1080, (b.x2 - b.x1) / 1920, (b.y2 - b.y1) / 1080
            lines.append(f"{cls_idx[b.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lp = DET / "labels" / s / f"{i}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml.safe_dump({"path": str(DET), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": dict(enumerate(OBST_CLASSES))}, open(DET / "data.yaml", "w"), sort_keys=False)
    b = boxes.merge(v1[["v1_id", "split"]], on="v1_id")
    print("det boxes:", b.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict(orient="index"))


def main():
    v1 = pd.read_csv(SPLITS / "v1.csv")
    boxes = pd.read_csv(SPLITS / "v1_boxes.csv")
    build_seg(v1)
    build_det(v1, boxes)


if __name__ == "__main__":
    main()
