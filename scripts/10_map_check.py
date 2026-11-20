"""Check the ODM map and export previews.

- Rail seg model on the orthophoto (resampled to the frames' ~331 px/m, tiled) -> rail centre spacing,
  compared with gauge + rail head (1.676 + 0.07 m) as a scale check.
- Sleeper period along the track from the intensity autocorrelation inside the track region.
- Levelling: without GPS the model is tilted by a few degrees. A plane fitted to the rail tops (assumed
  level, straight track) is subtracted; heights in dsm_levelled.tif are relative to the rail top.
- Previews: orthophoto with rail region, coloured hillshaded levelled DSM.

Outputs: reports/map/{map_stats.json, ortho.jpg, ortho_rail.jpg, dsm.jpg},
         data/map/odm/rail/odm_dem/dsm_levelled.tif
Usage: .venv/bin/python scripts/10_map_check.py [--seg weights/rail_seg.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, GAUGE_M, REPORTS, ROOT  # noqa: E402

PROJECT = DATA / "map" / "odm" / "rail"
OUT = REPORTS / "map"
FRAME_PX_PER_M = 331.3
RAIL_HEAD_M = 0.07


def segment(bgr, model, f):
    big = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    H, W = big.shape[:2]
    masks = np.zeros((2, H, W), np.uint8)
    ys = sorted(set(list(range(0, max(1, H - 1080), 700)) + [max(0, H - 1080)]))
    xs = sorted({0, max(0, W - 1920)})
    for y0 in ys:
        for x0 in xs:
            tile = big[y0:y0 + 1080, x0:x0 + 1920]
            r = model.predict(tile, imgsz=1024, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                continue
            th, tw = tile.shape[:2]
            for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int)):
                masks[c, y0:y0 + th, x0:x0 + tw] |= m[:th, :tw].astype(np.uint8)
    return masks


def rail_spacing_px(lines):
    gaps = []
    for y in range(0, lines.shape[0], 20):
        xs = np.nonzero(lines[y])[0]
        if len(xs) < 4:
            continue
        runs = [r for r in np.split(xs, np.where(np.diff(xs) > 3)[0] + 1) if len(r) >= 3]
        if len(runs) == 2:
            gaps.append(runs[1].mean() - runs[0].mean())
    return float(np.median(gaps)), len(gaps)


def sleeper_period_px(gray, track, valid, lo, hi):
    rows = np.array([gray[y][track[y] & valid[y]].mean() if (track[y] & valid[y]).sum() > 50 else np.nan
                     for y in range(gray.shape[0])])
    p = rows[~np.isnan(rows)]
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac /= ac[0]
    lag = int(np.argmax(ac[lo:hi]) + lo)
    return lag, float(ac[lag])


def rail_top_plane(lines, ortho_tf, dsm, dsm_tf):
    """Fit z = a*X + b*Y + c to DSM maxima on the rail centre lines (world coordinates)."""
    pts = []
    inv = ~dsm_tf
    for y in range(0, lines.shape[0], 10):
        xs = np.nonzero(lines[y])[0]
        if len(xs) < 4:
            continue
        runs = [r for r in np.split(xs, np.where(np.diff(xs) > 3)[0] + 1) if len(r) >= 3]
        if len(runs) != 2:
            continue
        for k, r in enumerate(runs):
            X, Y = ortho_tf * (r.mean(), y)
            c, rw = (int(v) for v in inv * (X, Y))
            z = np.nanmax(dsm[max(0, rw - 1):rw + 2, max(0, c - 1):c + 2])
            if np.isfinite(z):
                pts.append((k, X, Y, z))
    pts = np.array(pts)
    A = np.c_[pts[:, 1], pts[:, 2], np.ones(len(pts))]
    coef = np.linalg.lstsq(A, pts[:, 3], rcond=None)[0]
    resid = pts[:, 3] - A @ coef
    keep = np.abs(resid) < 3 * np.median(np.abs(resid)) + 1e-6
    coef = np.linalg.lstsq(A[keep], pts[keep, 3], rcond=None)[0]
    resid = pts[keep, 3] - A[keep] @ coef
    left, right = pts[pts[:, 0] == 0], pts[pts[:, 0] == 1]
    return coef, {
        "rail_points": int(keep.sum()),
        "tilt_deg": round(float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))), 2),
        "cross_level_before_m": round(float(np.median(right[:, 3]) - np.median(left[:, 3])), 3),
        "rail_top_residual_std_after_m": round(float(resid.std()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(ROOT / "weights" / "rail_seg.pt"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with rasterio.open(PROJECT / "odm_orthophoto" / "odm_orthophoto.tif") as src:
        rgba = src.read().transpose(1, 2, 0)
        cm_px = src.transform.a * 100
        ortho_tf = src.transform
    with rasterio.open(PROJECT / "odm_dem" / "dsm.tif") as src:
        dsm = src.read(1).astype(np.float64)
        dsm_nodata = src.nodata
        dsm_cm_px = src.transform.a * 100
        dsm_tf = src.transform
        profile = src.profile
    if dsm_nodata is not None:
        dsm[dsm == dsm_nodata] = np.nan
    bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    valid = rgba[..., 3] > 0

    f = cm_px / (100 / FRAME_PX_PER_M)
    masks = segment(bgr, YOLO(args.seg), f)
    spacing_px, n = rail_spacing_px(masks[1])
    spacing_m = spacing_px / f * cm_px / 100
    expected = GAUGE_M + RAIL_HEAD_M
    full = [cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
    track = full[0] | full[1]
    ys = np.nonzero(full[1].any(1))[0]
    lag, ac = sleeper_period_px(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), full[0], valid,
                                int(40 / cm_px), int(100 / cm_px))
    coef, level = rail_top_plane(full[1], ortho_tf, dsm, dsm_tf)
    cols, rows = np.meshgrid(np.arange(dsm.shape[1]) + 0.5, np.arange(dsm.shape[0]) + 0.5)
    X = dsm_tf.c + cols * dsm_tf.a
    Y = dsm_tf.f + rows * dsm_tf.e
    lev = dsm - (coef[0] * X + coef[1] * Y + coef[2])
    profile.update(dtype="float32", nodata=-9999.0)
    with rasterio.open(PROJECT / "odm_dem" / "dsm_levelled.tif", "w", **profile) as dst:
        dst.write(np.nan_to_num(lev, nan=-9999.0).astype(np.float32), 1)
    v = lev[np.isfinite(lev)]
    stats = {
        "orthophoto_px": [int(bgr.shape[1]), int(bgr.shape[0])],
        "orthophoto_cm_per_px": round(cm_px, 3),
        "dsm_cm_per_px": round(dsm_cm_px, 3),
        "covered_area_m2": round(float(valid.sum()) * (cm_px / 100) ** 2, 1),
        "rail_visible_length_m": round(float(ys.max() - ys.min()) * cm_px / 100, 1),
        "rail_centre_spacing_m": round(spacing_m, 3),
        "rail_centre_spacing_expected_m": expected,
        "scale_error_pct": round(100 * (spacing_m / expected - 1), 1),
        "rail_spacing_samples": n,
        "sleeper_period_m": round(lag * cm_px / 100, 3),
        "sleeper_period_autocorr": round(ac, 2),
        "levelling": level,
        "height_rel_rail_top_p1_p50_p99_m": np.percentile(v, [1, 50, 99]).round(2).tolist(),
        "note": "positions from SIFT frame solve + nominal UTM origin; no GPS, relative geometry only",
    }
    json.dump(stats, open(OUT / "map_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

    bg = np.where(valid[..., None], bgr, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "ortho.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ov = bg.copy()
    ov[track] = (ov[track] * 0.55 + np.array([60, 200, 60]) * 0.45).astype(np.uint8)
    cv2.imwrite(str(OUT / "ortho_rail.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    d = lev
    lo, hi = np.nanpercentile(d, [1, 99])
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    color = cv2.applyColorMap((np.nan_to_num(norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    gy, gx = np.gradient(np.nan_to_num(d, nan=float(np.nanmedian(d))))
    shade = np.clip(0.75 + (gx - gy) * 8, 0.35, 1.25)[..., None]
    out = np.clip(color * shade, 0, 255).astype(np.uint8)
    out[np.isnan(d)] = 255
    cv2.imwrite(str(OUT / "dsm.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
