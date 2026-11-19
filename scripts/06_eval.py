"""Evaluate trained models on the test split.

Rail (seg): Ultralytics mask/box mAP + per-class and track-union IoU on hole-filled GT masks.
Obstacles (det): Ultralytics mAP50 / mAP50-95 / per-class P,R / confusion matrix on the test frames
(unaugmented originals), plus P/R at fixed confidence thresholds (recall-first operating point).
Speed: ms/image at batch 1 on this machine's GPU.
Errors: worst cases rendered into reports/errors/.

Usage: .venv/bin/python scripts/06_eval.py --seg runs/seg/<run>/weights/best.pt [--seg ...] --det runs/det/<run>/weights/best.pt [--det ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OBST_CLASSES, RAIL_CLASSES, REPORTS  # noqa: E402
from viz import draw_boxes, overlay_masks, put_text  # noqa: E402

SEG = DATA / "yolo_seg"
DET = DATA / "yolo_det"
CONF_GRID = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5]


def fill_holes(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def seg_class_masks(result, shape):
    out = {c: np.zeros(shape, np.uint8) for c in RAIL_CLASSES}
    if result.masks is None:
        return out
    data = result.masks.data.cpu().numpy() > 0.5
    for m, c in zip(data, result.boxes.cls.cpu().numpy().astype(int)):
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        out[RAIL_CLASSES[c]] |= m
    return out


def eval_seg(weights, imgsz, n_err):
    model = YOLO(weights)
    m = model.val(data=str(SEG / "data.yaml"), split="test", imgsz=imgsz, batch=1, plots=True,
                  project=str(Path(weights).parents[1]), name="test_eval", exist_ok=True, verbose=False)
    out = {"mask_mAP50": m.seg.map50, "mask_mAP50_95": m.seg.map, "box_mAP50": m.box.map50,
           "per_class_mask_mAP50_95": dict(zip(RAIL_CLASSES, m.seg.maps.tolist()))}
    inter = {c: 0 for c in RAIL_CLASSES + ["track"]}
    union = dict(inter)
    per_img = []
    imgs = sorted((SEG / "images" / "test").glob("*.jpg"))
    for p in imgs:
        r = model.predict(str(p), imgsz=imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
        pm = seg_class_masks(r, r.orig_shape)
        gm = {c: cv2.imread(str(SEG / "gt_masks" / c / f"{p.stem}.png"), 0) > 0 for c in RAIL_CLASSES}
        pm["track"] = fill_holes(pm["rail_inside"] | pm["rail_lines"]).astype(bool)
        gm["track"] = fill_holes(gm["rail_inside"] | gm["rail_lines"]).astype(bool)
        ious = {}
        for c in inter:
            i, u = (pm[c].astype(bool) & gm[c]).sum(), (pm[c].astype(bool) | gm[c]).sum()
            inter[c] += i
            union[c] += u
            ious[c] = i / max(u, 1)
        per_img.append((p, ious, pm, gm))
    out["iou_dataset"] = {c: round(inter[c] / max(union[c], 1), 4) for c in inter}
    out["iou_per_image_mean"] = {c: round(float(np.mean([x[1][c] for x in per_img])), 4) for c in inter}
    out["n_test_images"] = len(imgs)
    # worst images by track IoU
    errs = []
    for p, ious, pm, gm in sorted(per_img, key=lambda x: x[1]["track"])[:n_err]:
        img = cv2.imread(str(p))
        gt_vis = overlay_masks(img, {"rail_inside": gm["rail_inside"], "rail_lines": gm["rail_lines"]})
        pr_vis = overlay_masks(img, {"rail_inside": pm["rail_inside"], "rail_lines": pm["rail_lines"]})
        gt_vis = put_text(gt_vis, (20, 20), "Nhãn (GT)", 44)
        pr_vis = put_text(pr_vis, (20, 20), f"Dự đoán - IoU vùng ray {ious['track']:.2f}", 44)
        errs.append((f"seg_{p.stem}", np.hstack([gt_vis, pr_vis]), ious["track"]))
    return out, errs


def box_iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def load_gt(label_path, w, h):
    rows = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    cls = np.array([int(r[0]) for r in rows])
    xywh = np.array([[float(v) for v in r[1:]] for r in rows]).reshape(-1, 4)
    xyxy = np.stack([(xywh[:, 0] - xywh[:, 2] / 2) * w, (xywh[:, 1] - xywh[:, 3] / 2) * h,
                     (xywh[:, 0] + xywh[:, 2] / 2) * w, (xywh[:, 1] + xywh[:, 3] / 2) * h], 1)
    return cls, xyxy


def match(gt_cls, gt_box, pr_cls, pr_box, pr_conf, thr):
    """Greedy same-class matching at IoU>=0.5. Returns per-pred TP flags, matched GT flags, and
    class-agnostic best match for each unmatched GT (for confusion cases)."""
    keep = pr_conf >= thr
    pc, pb, pf = pr_cls[keep], pr_box[keep], pr_conf[keep]
    order = np.argsort(-pf)
    pc, pb, pf = pc[order], pb[order], pf[order]
    tp = np.zeros(len(pc), bool)
    used = np.zeros(len(gt_cls), bool)
    if len(gt_cls) and len(pc):
        ious = box_iou(pb, gt_box)
        for k in range(len(pc)):
            cand = np.where((~used) & (gt_cls == pc[k]) & (ious[k] >= 0.5))[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                used[j], tp[k] = True, True
    return pc, pb, pf, tp, used


def eval_det(weights, imgsz, n_err):
    model = YOLO(weights)
    out = {}
    for subset, yml in [("test", "data.yaml")]:
        m = model.val(data=str(DET / yml), split="test", imgsz=imgsz, batch=1,
                      plots=True, project=str(Path(weights).parents[1]), name=f"{subset}_eval", exist_ok=True,
                      verbose=False)
        res = {"mAP50": m.box.map50, "mAP50_95": m.box.map, "precision": m.box.mp, "recall": m.box.mr,
               "per_class": {}}
        for k, c in enumerate(OBST_CLASSES):
            if k in m.box.ap_class_index:
                j = list(m.box.ap_class_index).index(k)
                p, r, ap50, ap = m.box.class_result(j)
                res["per_class"][c] = {"P": round(p, 4), "R": round(r, 4), "AP50": round(ap50, 4), "AP50_95": round(ap, 4)}
        out[subset] = res

    # own P/R at fixed thresholds + error mining on 'test'
    imgs = sorted((DET / "images" / "test").glob("*.jpg"))
    preds = []
    for p in imgs:
        r = model.predict(str(p), imgsz=imgsz, conf=min(CONF_GRID), verbose=False)[0]
        h, w = r.orig_shape
        gc, gb = load_gt(DET / "labels" / "test" / f"{p.stem}.txt", w, h)
        preds.append((p, gc, gb, r.boxes.cls.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy(),
                      r.boxes.conf.cpu().numpy()))
    grid = {}
    for thr in CONF_GRID:
        tp_c, fp_c, fn_c = (np.zeros(len(OBST_CLASSES), int) for _ in range(3))
        for _, gc, gb, pc, pb, pf in preds:
            c, _, _, tp, used = match(gc, gb, pc, pb, pf, thr)
            np.add.at(tp_c, c[tp], 1)
            np.add.at(fp_c, c[~tp], 1)
            np.add.at(fn_c, gc[~used], 1)
        grid[thr] = {"P": round(tp_c.sum() / max(tp_c.sum() + fp_c.sum(), 1), 4),
                     "R": round(tp_c.sum() / max(tp_c.sum() + fn_c.sum(), 1), 4),
                     "per_class_R": {c: round(tp_c[k] / max(tp_c[k] + fn_c[k], 1), 4) for k, c in enumerate(OBST_CLASSES)},
                     "per_class_P": {c: round(tp_c[k] / max(tp_c[k] + fp_c[k], 1), 4) for k, c in enumerate(OBST_CLASSES)}}
    out["pr_at_conf"] = grid

    # error cases at conf 0.25: missed GT first, then confident false positives
    errs = []
    for p, gc, gb, pc, pb, pf in preds:
        c, b, f, tp, used = match(gc, gb, pc, pb, pf, 0.25)
        n_fn, fp_conf = int((~used).sum()), float(f[~tp].max()) if (~tp).any() else 0.0
        if n_fn or fp_conf > 0:
            img = cv2.imread(str(p))
            vis = draw_boxes(img, [(OBST_CLASSES[k], bx, None) for k, bx in zip(gc, gb)], color=(0, 200, 0))
            vis = draw_boxes(vis, [(OBST_CLASSES[k], bx, s) for k, bx, s in zip(c, b, f)], color=(0, 0, 255))
            tag = "Bỏ sót" if n_fn else "Báo nhầm"
            vis = put_text(vis, (20, 20), f"{tag} - xanh: nhãn, đỏ: dự đoán", 40)
            score = n_fn * 2 + fp_conf
            errs.append((f"det_{'fn' if n_fn else 'fp'}_{p.stem}", vis, -score))
    errs.sort(key=lambda e: e[2])
    # keep variety: alternate FN / FP
    fn = [e for e in errs if "_fn_" in e[0]]
    fp = [e for e in errs if "_fp_" in e[0]]
    picked = (fn[: (n_err + 1) // 2] + fp[: n_err // 2])
    picked += [e for e in errs if e not in picked][: n_err - len(picked)]
    out["n_error_candidates"] = {"fn_images": len(fn), "fp_images": len(fp)}
    return out, picked


def speed(weights, imgsz, images, n=100):
    model = YOLO(weights)
    for p in images[:10]:
        model.predict(str(p), imgsz=imgsz, verbose=False)
    t = []
    for k in range(n):
        p = images[k % len(images)]
        img = cv2.imread(str(p))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False, retina_masks=False)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return {"ms_per_image_median": round(float(np.median(t)), 1), "ms_p90": round(float(np.percentile(t, 90)), 1),
            "imgsz": imgsz, "batch": 1, "device": torch.cuda.get_device_name(0), "note": "end-to-end predict(), image already decoded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--det", action="append", default=[])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default=str(REPORTS / "metrics.json"))
    ap.add_argument("--errors", type=int, default=10)
    args = ap.parse_args()
    (REPORTS / "errors").mkdir(parents=True, exist_ok=True)
    results = {"seg": {}, "det": {}, "gpu_busy_note": "GPU shared with other jobs during timing"}
    seg_imgs = sorted((SEG / "images" / "test").glob("*.jpg"))
    det_imgs = sorted((DET / "images" / "test").glob("*.jpg"))
    all_errs = {}
    for w in args.seg:
        name = Path(w).parents[1].name
        r, errs = eval_seg(w, args.imgsz, args.errors)
        r["speed"] = speed(w, args.imgsz, seg_imgs)
        results["seg"][name] = r
        all_errs[name] = errs
    for w in args.det:
        name = Path(w).parents[1].name
        r, errs = eval_det(w, args.imgsz, args.errors)
        r["speed"] = speed(w, args.imgsz, det_imgs)
        results["det"][name] = r
        all_errs[name] = errs
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False, default=float)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    # error gallery: 4 seg + 6 det from the last (largest) model of each task
    seg_e = all_errs.get(Path(args.seg[-1]).parents[1].name, [])[:4] if args.seg else []
    det_e = all_errs.get(Path(args.det[-1]).parents[1].name, [])[: args.errors - len(seg_e)] if args.det else []
    for name, vis, _ in seg_e + det_e:
        s = 1600 / max(vis.shape[:2])
        cv2.imwrite(str(REPORTS / "errors" / f"{name}.jpg"), cv2.resize(vis, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
