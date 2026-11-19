"""Demo media for the meeting: reports/demo.mp4 and reports/demo_picks/.

Frames: V1 test-block originals (never seen in training, nor any augmented copy of them).
Ground truth for picking: consolidated boxes (data/splits/v1_boxes.csv); an obstacle counts as
on-track when its GT box meets the GT track region dilated by the same buffer as the demo.
Picks: 6 correct frames (all on-track obstacles found with the right class, no false alarm),
2 failures (miss, wrong class or false alarm).

Usage: .venv/bin/python scripts/07_demo_media.py [--seconds-per-image 2.5] [--n-video 10]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OBST_VI, REPORTS, ROOT, SPLITS, v1_image  # noqa: E402
from demo import RailObstacleDemo  # noqa: E402
from viz import put_text  # noqa: E402

FPS = 25
W, H = 1920, 1080


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def gt_track_zone(v1_id, buffer_px):
    seg = DATA / "yolo_seg" / "gt_masks"
    m = (cv2.imread(str(seg / "rail_inside" / f"{v1_id}.png"), 0) > 0) | (cv2.imread(str(seg / "rail_lines" / f"{v1_id}.png"), 0) > 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_px + 1, 2 * buffer_px + 1))
    return cv2.dilate(m.astype(np.uint8), k)


def judge(a, gt_boxes, zone):
    gt_on = [g for g in gt_boxes if zone[int(g[2]):int(g[4]), int(g[1]):int(g[3])].any()]
    pred_on = [d for d in a["dets"] if d["on_track"]]
    used, hits, wrong_cls = set(), 0, 0
    for g in gt_on:
        best = max(((iou(g[1:], d["box"]), k) for k, d in enumerate(pred_on) if k not in used), default=(0, -1))
        if best[0] >= 0.5:
            used.add(best[1])
            if pred_on[best[1]]["cls"] == g[0]:
                hits += 1
            else:
                wrong_cls += 1
    false_alarms = len(pred_on) - len(used)
    missed = len(gt_on) - hits - wrong_cls
    ok = missed == 0 and wrong_cls == 0 and false_alarms == 0 and len(gt_on) > 0
    conf = min((d["conf"] for d in pred_on), default=0)
    return {"gt_on": len(gt_on), "hits": hits, "missed": missed, "wrong_cls": wrong_cls,
            "false_alarms": false_alarms, "ok": ok, "min_conf": conf,
            "classes": sorted({g[0] for g in gt_on})}


def title_card(text, sub):
    img = np.full((H, W, 3), 25, np.uint8)
    img = put_text(img, (120, 420), text, 72, bg=None)
    img = put_text(img, (120, 540), sub, 40, fg=(200, 200, 200), bg=None)
    return img


def write_video(frames, path, seconds, fade=0.4):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    hold, nf = int(seconds * FPS), int(fade * FPS)
    prev = None
    for f in frames:
        f = cv2.resize(f, (W, H))
        if prev is not None:
            for k in range(1, nf + 1):
                p.stdin.write(cv2.addWeighted(prev, 1 - k / nf, f, k / nf, 0).tobytes())
        for _ in range(hold):
            p.stdin.write(f.tobytes())
        prev = f
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(ROOT / "weights" / "rail_seg.pt"))
    ap.add_argument("--det", default=str(ROOT / "weights" / "obstacle_det.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--buffer-m", type=float, default=0.5)
    ap.add_argument("--seconds-per-image", type=float, default=2.5)
    ap.add_argument("--n-video", type=int, default=10)
    args = ap.parse_args()

    v1 = pd.read_csv(SPLITS / "v1.csv")
    gt = {i: [(b.cls, b.x1, b.y1, b.x2, b.y2) for b in g.itertuples()]
          for i, g in pd.read_csv(SPLITS / "v1_boxes.csv").groupby("v1_id")}
    test_ids = sorted(v1[v1.split == "test"].v1_id)

    demo = RailObstacleDemo(args.seg, args.det, conf=args.conf, buffer_m=args.buffer_m)
    out_dir = REPORTS / "demo_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in test_ids:
        img = cv2.imread(str(v1_image(i)))
        a = demo.analyse(img)
        vis = demo.render(img, a)
        cv2.imwrite(str(out_dir / f"{i}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        row = {"v1_id": i, "n_on_track": sum(d["on_track"] for d in a["dets"]), "px_per_m": a["px_per_m"]}
        row.update(judge(a, gt.get(i, []), gt_track_zone(i, a["buffer_px"])))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "judged.csv", index=False)
    judged = df[df.gt_on > 0]
    print(f"{len(df)} test frames, {len(judged)} with on-track GT obstacles; correct {int(judged.ok.sum())}")

    # picks: 6 correct with distinct classes first, then 2 failures
    picks_dir = REPORTS / "demo_picks"
    if picks_dir.exists():
        shutil.rmtree(picks_dir)
    picks_dir.mkdir(parents=True)
    good = judged[judged.ok].sort_values("min_conf", ascending=False)
    chosen, seen = [], set()
    for _, r in good.iterrows():
        cls = tuple(r.classes)
        if cls not in seen:
            chosen.append(r.v1_id)
            seen.add(cls)
    chosen += [i for i in good.v1_id if i not in chosen]
    chosen = chosen[:6]
    bad = df[~df.ok & ((df.gt_on > 0) | (df.false_alarms > 0))]
    bad = bad.assign(sev=bad.missed * 3 + bad.wrong_cls * 2 + bad.false_alarms).sort_values("sev", ascending=False)
    failures = list(bad.v1_id[:2])
    for k, i in enumerate(chosen, 1):
        shutil.copy(out_dir / f"{i}.jpg", picks_dir / f"good_{k}_frame{i}.jpg")
    for k, i in enumerate(failures, 1):
        r = df[df.v1_id == i].iloc[0]
        why = "bo_sot" if r.missed else ("sai_lop" if r.wrong_cls else "bao_nham")
        shutil.copy(out_dir / f"{i}.jpg", picks_dir / f"fail_{k}_{why}_frame{i}.jpg")
        (picks_dir / f"fail_{k}_{why}_frame{i}.txt").write_text(
            "Nhãn đúng: " + ", ".join(OBST_VI[g[0]] for g in gt.get(i, [])) + f"\nmissed={r.missed} wrong_cls={r.wrong_cls} false_alarms={r.false_alarms}\n")
    json.dump({"good": [int(i) for i in chosen], "fail": [int(i) for i in failures]}, open(picks_dir / "picks.json", "w"))

    # video: title + correct frames with variety, then 1 failure for honesty is left to the slides
    order = chosen + [i for i in df.sort_values("n_on_track", ascending=False).v1_id if i not in chosen]
    frames = [title_card("AI phát hiện vật cản trên đường ray", "Ảnh drone, dữ liệu công khai UAV-RSOD (Ấn Độ) · ảnh test, model chưa từng thấy")]
    frames += [cv2.imread(str(out_dir / f"{i}.jpg")) for i in order[: args.n_video]]
    write_video(frames, REPORTS / "demo.mp4", args.seconds_per_image)
    dur = len(frames) * args.seconds_per_image + (len(frames) - 1) * 0.4
    print(f"video: {len(frames)} frames, ~{dur:.0f}s -> {REPORTS / 'demo.mp4'}")


if __name__ == "__main__":
    main()
