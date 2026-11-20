"""Prepare an OpenDroneMap project from the 315 V1 frames.

- images/: hardlinks to the frames (Docker bind mounts do not follow symlinks out of the volume)
- images/<name>_mask.jpg: black over labelled obstacles/people (moved between passes), white elsewhere
- geo.txt: approximate camera positions in metres from the SIFT frame-position solve (02_split.py),
  scaled by the rail gauge, placed at a nominal UTM 44N origin near Tiruchirappalli. Only relative
  positions are meaningful; the frames carry no GPS.

Usage: .venv/bin/python scripts/08_map_prepare.py [--altitude 10.5] [--pad 20]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, REPORTS, SPLITS, v1_image  # noqa: E402

PROJECT = DATA / "map" / "odm" / "rail"
EPSG = "EPSG:32644"
ORIGIN_E, ORIGIN_N = 247100.0, 1194000.0  # nominal, not surveyed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--altitude", type=float, default=4.5,
                    help="height above rail; footprint ~5.8 m wide with an 84 deg lens gives ~3-4.5 m (paper: 10.5)")
    ap.add_argument("--pad", type=int, default=20, help="mask dilation around obstacle boxes, px")
    ap.add_argument("--horiz-acc", type=float, default=0.3)
    ap.add_argument("--vert-acc", type=float, default=3.0)
    args = ap.parse_args()

    ppm = json.load(open(REPORTS / "split_summary.json"))["px_per_m_matching_scale"]
    v1 = pd.read_csv(SPLITS / "v1.csv")
    boxes = pd.read_csv(SPLITS / "v1_boxes.csv")

    img_dir = PROJECT / "images"
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    img_dir.mkdir(parents=True)

    lines = [EPSG]
    for r in v1.itertuples():
        name = f"{r.v1_id:03d}.jpg"
        os.link(v1_image(r.v1_id), img_dir / name)
        mask = np.full((1080, 1920), 255, np.uint8)
        for b in boxes[boxes.v1_id == r.v1_id].itertuples():
            x1, y1 = max(0, int(b.x1) - args.pad), max(0, int(b.y1) - args.pad)
            x2, y2 = min(1920, int(b.x2) + args.pad), min(1080, int(b.y2) + args.pad)
            mask[y1:y2, x1:x2] = 0
        cv2.imwrite(str(img_dir / f"{r.v1_id:03d}_mask.jpg"), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # image axes: x right, y down -> east, north = -y
        e = ORIGIN_E + r.x_px / ppm
        n = ORIGIN_N - r.y_px / ppm
        lines.append(f"{name} {e:.3f} {n:.3f} {args.altitude:.2f} 0 0 0 {args.horiz_acc} {args.vert_acc}")
    (PROJECT / "geo.txt").write_text("\n".join(lines) + "\n")
    span = (v1.y_px.max() - v1.y_px.min()) / ppm
    print(f"{len(v1)} frames, {int((boxes.groupby('v1_id').size() > 0).sum())} masked; "
          f"camera span along track {span:.1f} m -> {PROJECT}")


if __name__ == "__main__":
    main()
