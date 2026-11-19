"""Rail region + on-track obstacle demo.

For each drone image: track region (rail_inside + rail_lines, holes filled) is shaded green; obstacles
whose box intersects the track region dilated by --buffer-m metres are boxed in red with a Vietnamese
label and confidence; the top-left banner reads "Phát hiện N vật cản trên ray".
Metres are converted to pixels per image from the predicted rail spacing (gauge 1.676 m).

Usage:
  .venv/bin/python scripts/demo.py <image|dir> [--out reports/demo_out] [--seg weights/rail_seg.pt]
                                   [--det weights/obstacle_det.pt] [--conf 0.25] [--buffer-m 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import GAUGE_M, OBST_CLASSES, OBST_VI, RAIL_CLASSES, REPORTS, ROOT  # noqa: E402
from viz import put_text  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RAIL_HEAD_M = 0.07
DEFAULT_PX_PER_M = 300.0  # at 1920x1080, dataset median; used when rails cannot be measured
GREEN = (60, 200, 60)
RED = (0, 0, 230)


def fill_holes(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def class_masks(result, shape):
    out = {c: np.zeros(shape, np.uint8) for c in RAIL_CLASSES}
    if result.masks is None:
        return out
    for m, c in zip(result.masks.data.cpu().numpy() > 0.5, result.boxes.cls.cpu().numpy().astype(int)):
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
        out[RAIL_CLASSES[c]] |= m.astype(np.uint8)
    return out


def px_per_metre(track, rails):
    """Rail centre spacing in px, measured perpendicular to the track direction."""
    ys, xs = np.nonzero(track[::4, ::4])
    if len(xs) < 50 or rails.sum() < 500:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32) * 4
    mean, vecs = cv2.PCACompute(pts, mean=None)
    angle = np.degrees(np.arctan2(vecs[0, 1], vecs[0, 0]))  # principal axis = along track
    h, w = rails.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle - 90, 1.0)  # make track vertical
    rot = cv2.warpAffine(rails, R, (w, h), flags=cv2.INTER_NEAREST)
    gaps = []
    for y in range(0, h, 16):
        xs = np.nonzero(rot[y])[0]
        if len(xs) < 4:
            continue
        runs = np.split(xs, np.where(np.diff(xs) > 3)[0] + 1)
        runs = [r for r in runs if len(r) >= 3]
        if len(runs) == 2:
            gaps.append(runs[1].mean() - runs[0].mean())
    if len(gaps) < 5:
        return None
    return float(np.median(gaps)) / (GAUGE_M + RAIL_HEAD_M)


class RailObstacleDemo:
    def __init__(self, seg_w, det_w, imgsz=1024, conf=0.25, seg_conf=0.25, buffer_m=0.5, device=None):
        self.seg, self.det = YOLO(seg_w), YOLO(det_w)
        self.imgsz, self.conf, self.seg_conf, self.buffer_m, self.device = imgsz, conf, seg_conf, buffer_m, device

    def analyse(self, img):
        h, w = img.shape[:2]
        rs = self.seg.predict(img, imgsz=self.imgsz, conf=self.seg_conf, retina_masks=True, verbose=False, device=self.device)[0]
        cm = class_masks(rs, (h, w))
        track = fill_holes(cm["rail_inside"] | cm["rail_lines"])
        ppm = px_per_metre(track, cm["rail_lines"])
        ppm_used = ppm if ppm and 0.3 * DEFAULT_PX_PER_M * w / 1920 < ppm < 3 * DEFAULT_PX_PER_M * w / 1920 \
            else DEFAULT_PX_PER_M * w / 1920
        r = max(1, int(round(self.buffer_m * ppm_used)))
        zone = cv2.dilate(track, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) if track.any() else track
        rd = self.det.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
        dets = []
        for c, b, s in zip(rd.boxes.cls.cpu().numpy().astype(int), rd.boxes.xyxy.cpu().numpy(), rd.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            on = bool(zone[y1:y2, x1:x2].any()) if x2 > x1 and y2 > y1 else False
            inside = bool(track[y1:y2, x1:x2].any()) if x2 > x1 and y2 > y1 else False
            dets.append({"cls": OBST_CLASSES[c], "conf": float(s), "box": [x1, y1, x2, y2], "on_track": on,
                         "touches_track": inside})
        return {"track": track, "zone": zone, "px_per_m": ppm, "px_per_m_used": ppm_used, "buffer_px": r, "dets": dets}

    def render(self, img, a, show_off_track=False):
        h, w = img.shape[:2]
        out = img.copy()
        t = a["track"].astype(bool)
        out[t] = (out[t] * 0.55 + np.array(GREEN) * 0.45).astype(np.uint8)
        lw = max(2, round(w / 640))
        zc, _ = cv2.findContours(a["zone"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, zc, -1, (150, 255, 150), max(1, lw // 2), cv2.LINE_AA)
        size = max(20, round(w / 60))
        on = [d for d in a["dets"] if d["on_track"]]
        if show_off_track:
            for d in a["dets"]:
                if not d["on_track"]:
                    x1, y1, x2, y2 = d["box"]
                    cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), lw)
        for d in on:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), RED, 2 * lw)
        for d in on:
            x1, y1, x2, y2 = d["box"]
            ty = y1 - size - 16 if y1 > size + 20 else y2 + 10
            out = put_text(out, (x1 + 6, min(ty, h - size - 12)), f"{OBST_VI[d['cls']]} {d['conf']:.0%}", size, bg=RED)
        n = len(on)
        banner = f"Phát hiện {n} vật cản trên ray"
        out = put_text(out, (round(w / 64), round(w / 64)), banner, round(size * 1.35), bg=RED if n else (40, 120, 40), pad=12)
        out = put_text(out, (round(w / 64), h - round(size * 0.8) - round(w / 64)),
                       f"Vùng ray (xanh) · ngưỡng ≤ {self.buffer_m:g} m", round(size * 0.7), bg=(30, 30, 30), pad=8)
        return out


def iter_images(src: Path):
    if src.is_file():
        yield src
    else:
        yield from sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=str(REPORTS / "demo_out"))
    ap.add_argument("--seg", default=str(ROOT / "weights" / "rail_seg.pt"))
    ap.add_argument("--det", default=str(ROOT / "weights" / "obstacle_det.pt"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25, help="obstacle confidence threshold")
    ap.add_argument("--buffer-m", type=float, default=0.5, help="max distance from track region, metres")
    ap.add_argument("--show-off-track", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    demo = RailObstacleDemo(args.seg, args.det, args.imgsz, args.conf, buffer_m=args.buffer_m, device=args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for p in iter_images(Path(args.src)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        a = demo.analyse(img)
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), demo.render(img, a, args.show_off_track), [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.append({"image": str(p), "px_per_m": a["px_per_m"], "buffer_px": a["buffer_px"],
                    "track_area_pct": round(100 * float(a["track"].mean()), 2), "detections": a["dets"]})
        n = sum(d["on_track"] for d in a["dets"])
        print(f"{p.name}: {n} on track / {len(a['dets'])} detected")
    json.dump(log, open(out_dir / "detections.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
