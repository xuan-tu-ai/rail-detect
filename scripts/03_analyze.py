"""Dataset statistics and labelled samples.

Outputs: reports/data_stats.json, reports/samples/*.jpg
Requires data/splits/ from 02_split.py.
Usage: .venv/bin/python scripts/03_analyze.py
"""
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import (OBST_CLASSES, RAIL_CLASSES, REPORTS, SPLITS, V1, v1_ids, v1_image, v1_mask,  # noqa: E402
                    v1_mask_provided, v2_records)
from viz import draw_boxes, overlay_masks  # noqa: E402

SAMPLES = REPORTS / "samples"


def v1_stats(ids):
    sizes = collections.Counter(Image.open(v1_image(i)).size for i in ids)
    low_res = {c: sum(Image.open(V1 / "2 Annotations" / "2.2 Masking" / d / f"{i}.jpg").size != (1920, 1080)
                      for i in ids) for c, d in [("rail_inside", "Rail Inside"), ("rail_lines", "Rail Lines")]}
    cover = {c: [] for c in RAIL_CLASSES}
    iou = {c: [] for c in RAIL_CLASSES}
    for i in ids:
        for c in RAIL_CLASSES:
            m = v1_mask(i, c)
            cover[c].append(m.mean())
            p = V1 / "2 Annotations" / "2.2 Masking" / ("Rail Inside" if c == "rail_inside" else "Rail Lines") / f"{i}.jpg"
            if Image.open(p).size == (1920, 1080):
                g = v1_mask_provided(i, c).astype(bool)
                mb = m.astype(bool)
                iou[c].append((mb & g).sum() / max((mb | g).sum(), 1))
    return {
        "images": len(ids),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "provided_masks_low_res_334x188": low_res,
        "mask_area_pct_mean": {c: round(100 * float(np.mean(v)), 1) for c, v in cover.items()},
        "derived_vs_provided_fullres_iou": {c: {"n": len(v), "mean": round(float(np.mean(v)), 4),
                                                "min": round(float(np.min(v)), 4)} for c, v in iou.items()},
    }


def v2_stats(recs, split_df):
    boxes = [(r["author_split"], b) for r in recs for b in r["boxes"]]
    wh = np.array([(b[3] - b[1], b[4] - b[2]) for _, b in boxes])
    out = {
        "images": len(recs),
        "author_split_images": dict(collections.Counter(r["author_split"] for r in recs)),
        "image_sizes": {f"{w}x{h}": n for (w, h), n in collections.Counter((r["w"], r["h"]) for r in recs).items()},
        "boxes_per_class": {c: sum(b[0] == c for _, b in boxes) for c in OBST_CLASSES},
        "boxes_per_image": dict(collections.Counter(len(r["boxes"]) for r in recs)),
        "box_w_px_p5_p50_p95": np.percentile(wh[:, 0], [5, 50, 95]).round().tolist(),
        "box_h_px_p5_p50_p95": np.percentile(wh[:, 1], [5, 50, 95]).round().tolist(),
    }
    if split_df is not None:
        d = split_df
        out["matched_to_v1_frame"] = int(d.matched.sum())
        out["unaugmented_copies"] = int(d.identity.sum())
        out["distinct_source_frames"] = int(d[d.matched].v1_id.nunique())
        out["copies_per_source_frame"] = d[d.matched].groupby("v1_id").size().describe()[["mean", "min", "max"]].round(2).to_dict()
        out["flipped"] = int(d[d.matched].flip.sum())
        out["abs_rotation_deg_p50_p95"] = d[d.matched].rot_deg.abs().quantile([0.5, 0.95]).round(1).tolist()
        # how often the authors' train/test split puts copies of one frame on both sides
        g = d[d.matched].groupby("v1_id").author_split.nunique()
        out["author_split_frames_on_both_sides"] = f"{int((g > 1).sum())}/{len(g)}"
    return out


def consolidated_stats(boxes, v1):
    b = boxes.merge(v1[["v1_id", "split", "test_kind"]], on="v1_id")
    per_frame = boxes.groupby("v1_id").size().reindex(v1.v1_id, fill_value=0)
    return {
        "frames": len(v1),
        "boxes_per_class": boxes.cls.value_counts().to_dict(),
        "objects_per_frame": {int(k): int(v) for k, v in per_frame.value_counts().sort_index().items()},
        "boxes_from_unaugmented_copy": int(boxes.from_identity.sum()),
        "frames_per_split": v1.split.value_counts().to_dict(),
        "boxes_per_split_class": b.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "test_frames_by_kind": v1[v1.split == "test"].test_kind.fillna("").value_counts().to_dict(),
    }


def samples(ids, recs, split_df, boxes):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for f in SAMPLES.glob("*.jpg"):
        f.unlink()
    rng = np.random.default_rng(0)
    for c in OBST_CLASSES:
        cand = boxes[boxes.cls == c].v1_id.unique()
        i = int(rng.choice(cand))
        img = cv2.imread(str(v1_image(i)))
        vis = overlay_masks(img, {k: v1_mask(i, k) for k in RAIL_CLASSES})
        vis = draw_boxes(vis, [(r.cls, (r.x1, r.y1, r.x2, r.y2), None) for r in boxes[boxes.v1_id == i].itertuples()])
        cv2.imwrite(str(SAMPLES / f"v1_{c}_frame{i}.jpg"), cv2.resize(vis, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    by_cls = collections.defaultdict(list)
    for r in recs:
        by_cls[r["boxes"][0][0]].append(r)
    for c in OBST_CLASSES:
        r = by_cls[c][rng.integers(len(by_cls[c]))]
        vis = draw_boxes(cv2.imread(str(r["path"])), [(b[0], b[1:], None) for b in r["boxes"]])
        s = 960 / max(vis.shape[:2])
        cv2.imwrite(str(SAMPLES / f"v2_{c}_{r['id']}.jpg"), cv2.resize(vis, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if split_df is not None:
        # one source frame with all of its augmented copies
        d = split_df[split_df.matched]
        v1_id = d.groupby("v1_id").size().sort_values().index[-1]
        rid = {r["id"]: r for r in recs}
        tiles = [cv2.imread(str(v1_image(int(v1_id))))]
        for v2_id in d[d.v1_id == v1_id].v2_id.head(7):
            r = rid[int(v2_id)]
            tiles.append(draw_boxes(cv2.imread(str(r["path"])), [(b[0], b[1:], None) for b in r["boxes"]]))
        W, H = 480, 270
        sheet = np.full((H * 2, W * 4, 3), 255, np.uint8)
        for k, t in enumerate(tiles[:8]):
            s = min(W / t.shape[1], H / t.shape[0])
            t = cv2.resize(t, None, fx=s, fy=s)
            y0, x0 = (k // 4) * H, (k % 4) * W
            sheet[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        cv2.imwrite(str(SAMPLES / f"aug_group_v1_{int(v1_id)}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    REPORTS.mkdir(exist_ok=True)
    ids = v1_ids()
    recs = v2_records()
    split_df = pd.read_csv(SPLITS / "v2.csv")
    v1 = pd.read_csv(SPLITS / "v1.csv")
    boxes = pd.read_csv(SPLITS / "v1_boxes.csv")
    stats = {"v1_segmentation": v1_stats(ids), "v2_obstacle": v2_stats(recs, split_df),
             "v1_obstacle_labels_consolidated": consolidated_stats(boxes, v1)}
    json.dump(stats, open(REPORTS / "data_stats.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    samples(ids, recs, split_df, boxes)
    print("samples ->", SAMPLES)


if __name__ == "__main__":
    main()
