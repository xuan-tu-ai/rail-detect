"""Match augmented images to source frames, consolidate obstacle labels, write leak-free splits.

1. V1 x V1 SIFT matching -> pairwise similarity transforms -> least-squares frame positions (track coords).
2. V2 -> V1 matching (FLANN voting + RANSAC affine, flip-aware) -> source frame and transform of each V2 image.
3. V2 boxes projected into their V1 frame, kept when the box region matches the frame content (NCC),
   merged per frame. V2 images carry one box each although frames often hold several obstacles.
4. Frames from one moment of the recording (same obstacle set at the same track position, or near-identical)
   are grouped into placements. The recording has ~10 placements; Boulder, Branch and Jerrycan have one each.
   Multi-placement classes: a whole placement goes to test ('new_placement'). Single-placement classes:
   the far end of the drone path goes to test, the adjacent block to val as a guard ('new_view').

Outputs: data/splits/{v1.csv, v1_boxes.csv, v2.csv}, reports/split_summary.json, reports/split_positions.png
Usage: .venv/bin/python scripts/02_split.py [--test-frac 0.2] [--val-frac 0.15] [--ncc-min 0.5]
"""
import argparse
import json
import pickle
import sys
import multiprocessing as mp
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import CACHE, OBST_CLASSES, REPORTS, SPLITS, v1_ids, v1_image, v1_mask, v2_records  # noqa: E402

SCALE_W = 640  # matching resolution (long side)
N_FEAT_V1, N_FEAT_V2 = 800, 1000
PAIR_MIN = 40  # inliers for a V1 pair to count as overlapping ground
TOPK = 5


def load_small(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    s = SCALE_W / max(h, w)
    return cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def sift(g, n):
    k, d = cv2.SIFT_create(n).detectAndCompute(g, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([p.pt for p in k]), d


def ratio_matches(da, db, matcher):
    if len(da) < 2 or len(db) < 2:
        return []
    return [x for x, y in matcher.knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]


# ---------- V1 features and pairwise transforms ----------

def v1_features():
    f = CACHE / "v1_sift.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    feats = {i: sift(load_small(v1_image(i)), N_FEAT_V1) for i in v1_ids()}
    pickle.dump(feats, open(f, "wb"))
    return feats


def _v1_row(a):
    bf = cv2.BFMatcher()
    ka, da = F[a]
    out = []
    for b in IDS:
        if b <= a:
            continue
        kb, db = F[b]
        m = ratio_matches(da, db, bf)
        if len(m) < 8:
            continue
        M, inl = cv2.estimateAffinePartial2D(ka[[x.queryIdx for x in m]], kb[[x.trainIdx for x in m]],
                                             method=cv2.RANSAC, ransacReprojThreshold=4)
        if M is not None and inl.sum() >= 8:
            out.append((a, b, int(inl.sum()), M))
    return out


def _init_v1(feats):
    global F, IDS
    F, IDS = feats, sorted(feats)
    cv2.setNumThreads(1)


def v1_pairs(feats, workers):
    f = CACHE / "v1_pairs.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    pairs = []
    # spawn: forking after OpenCV has started its thread pool deadlocks the workers
    with mp.get_context("spawn").Pool(workers, initializer=_init_v1, initargs=(feats,)) as p:
        for r in p.imap_unordered(_v1_row, sorted(feats)):
            pairs += r
    pickle.dump(pairs, open(f, "wb"))
    return pairs


def solve_positions(ids, pairs):
    """Robust least squares for frame-centre positions (px at matching scale) from pairwise shifts."""
    idx = {i: k for k, i in enumerate(ids)}
    strong = [p for p in pairs if p[2] >= PAIR_MIN]
    c = np.array([SCALE_W / 2, SCALE_W * 9 / 32, 1.0])
    A = np.zeros((len(strong) + 1, len(ids)))
    rhs = np.zeros((len(strong) + 1, 2))
    for r, (a, b, n, M) in enumerate(strong):
        A[r, idx[b]], A[r, idx[a]] = 1, -1
        rhs[r] = -(M @ c - c[:2])
    A[-1, 0] = 1
    w0 = np.array([p[2] for p in strong], float)
    w = w0.copy()
    for _ in range(5):
        W = np.sqrt(np.append(w, 1e3))[:, None]
        X = np.linalg.lstsq(A * W, rhs * W, rcond=None)[0]
        res = np.linalg.norm(A[:-1] @ X - rhs[:-1], axis=1)
        w = w0 / (1 + (res / 20) ** 2)
    X -= np.median(X, 0)
    return X, res


def px_per_m(ids, n=40):
    """Rail-centre spacing from rail_lines masks, at matching scale."""
    rng = np.random.default_rng(0)
    vals = []
    for i in rng.choice(ids, min(n, len(ids)), replace=False):
        m = v1_mask(int(i), "rail_lines")
        for y in range(100, 1080, 120):
            xs = np.where(m[y] > 0)[0]
            if len(xs) < 4:
                continue
            gaps = np.where(np.diff(xs) > 30)[0]
            if len(gaps) != 1:
                continue
            left, right = xs[: gaps[0] + 1], xs[gaps[0] + 1:]
            vals.append(right.mean() - left.mean())
    head_m = 0.07  # rail head width; centre spacing = gauge + head width
    return float(np.median(vals)) * SCALE_W / 1920 / (1.676 + head_m)


# ---------- V2 -> V1 matching ----------

def _init_v2(feats):
    global F, IDS, FLANN, OWNER
    cv2.setNumThreads(1)
    F, IDS = feats, sorted(feats)
    descs = np.vstack([feats[i][1] for i in IDS]).astype(np.float32)
    OWNER = np.concatenate([np.full(len(feats[i][1]), i) for i in IDS])
    FLANN = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    FLANN.add([descs])
    FLANN.train()


def _match_v2(rec):
    g = load_small(rec["path"])
    bf = cv2.BFMatcher()
    best = {"v2_id": rec["id"], "v1_id": -1, "inliers": 0, "flip": False, "M": None, "second": 0}
    cands = []
    for flip in (False, True):
        gi = cv2.flip(g, 1) if flip else g
        kq, dq = sift(gi, N_FEAT_V2)
        if len(dq) < 8:
            continue
        votes = np.zeros(max(IDS) + 1)
        for x, y in FLANN.knnMatch(dq.astype(np.float32), k=2):
            if x.distance < 0.8 * y.distance:
                votes[OWNER[x.trainIdx]] += 1
        for v in np.argsort(votes)[::-1][:TOPK]:
            kv, dv = F[int(v)]
            m = ratio_matches(dq, dv, bf)
            if len(m) < 8:
                continue
            M, inl = cv2.estimateAffine2D(kq[[x.queryIdx for x in m]], kv[[x.trainIdx for x in m]],
                                          method=cv2.RANSAC, ransacReprojThreshold=5)
            if M is not None:
                cands.append((int(inl.sum()), int(v), flip, M))
    cands.sort(key=lambda c: -c[0])
    if cands:
        n, v, flip, M = cands[0]
        second = next((c[0] for c in cands[1:] if c[1] != v), 0)
        best.update(v1_id=v, inliers=n, flip=flip, M=M, second=second)
    # identity check: same frame without geometric augmentation
    M = best["M"]
    best["identity"] = bool(M is not None and not best["flip"] and g.shape[1] > g.shape[0]
                            and np.abs(M[:, :2] - np.eye(2)).max() < 0.02 and np.abs(M[:, 2]).max() < 3)
    if best["identity"]:
        ref = load_small(v1_image(best["v1_id"]))
        best["brightness"] = float(g.mean() / max(ref.mean(), 1))
    else:
        best["brightness"] = np.nan
    best["rot_deg"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))) if M is not None else np.nan
    best["scale"] = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) if M is not None else np.nan
    for k, v in zip(["m00", "m01", "m02", "m10", "m11", "m12"], (M.ravel() if M is not None else [np.nan] * 6)):
        best[k] = float(v)
    best["w_small"], best["h_small"] = g.shape[1], g.shape[0]
    best.pop("M")
    return best


def match_v2(feats, recs, workers):
    f = CACHE / "v2_match_affine.csv"
    if f.exists():
        return pd.read_csv(f)
    out = []
    with mp.get_context("spawn").Pool(workers, initializer=_init_v2, initargs=(feats,)) as p:
        for k, r in enumerate(p.imap_unordered(_match_v2, recs, chunksize=8)):
            out.append(r)
            if k % 200 == 0:
                print(f"  v2 match {k}/{len(recs)}", flush=True)
    df = pd.DataFrame(out).sort_values("v2_id")
    df.to_csv(f, index=False)
    return df


# ---------- label consolidation ----------

def project_boxes(row, rec):
    """V2 boxes -> full-res coordinates of the matched V1 frame."""
    M = np.array([[row.m00, row.m01, row.m02], [row.m10, row.m11, row.m12]])
    s2, s1 = SCALE_W / max(rec["w"], rec["h"]), SCALE_W / 1920
    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    t = np.radians((theta + 45) % 90 - 45)  # rotation left after the nearest multiple of 90 deg
    c, s = abs(np.cos(t)), abs(np.sin(t))
    out = []
    for cls, x1, y1, x2, y2 in rec["boxes"]:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float) * s2
        if row.flip:
            pts[:, 0] = row.w_small - pts[:, 0]
        q = (pts @ M[:, :2].T + M[:, 2]) / s1
        (bx1, by1), (bx2, by2) = q.min(0), q.max(0)
        cx, cy, bw, bh = (bx1 + bx2) / 2, (by1 + by2) / 2, bx2 - bx1, by2 - by1
        if s > 0.03:  # undo bbox inflation from a residual rotation
            d = c * c - s * s
            bw, bh = max((bw * c - bh * s) / d, 0.6 * bw), max((bh * c - bw * s) / d, 0.6 * bh)
        out.append((cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _init_verify():
    global V1_SMALL
    cv2.setNumThreads(1)
    V1_SMALL = {}


def _verify(args):
    """Check each projected box against the frame content: NCC between the frame and the warped V2 image."""
    row, rec = args
    f = int(row.v1_id)
    if f not in V1_SMALL:
        V1_SMALL[f] = load_small(v1_image(f)).astype(np.float32)
    g1 = V1_SMALL[f]
    g2 = load_small(rec["path"])
    if row.flip:
        g2 = cv2.flip(g2, 1)
    M = np.array([[row.m00, row.m01, row.m02], [row.m10, row.m11, row.m12]])
    warped = cv2.warpAffine(g2, M, (g1.shape[1], g1.shape[0])).astype(np.float32)
    valid = cv2.warpAffine(np.ones_like(g2), M, (g1.shape[1], g1.shape[0]), flags=cv2.INTER_NEAREST) > 0
    s1 = SCALE_W / 1920
    out = []
    for cls, x1, y1, x2, y2 in project_boxes(row, rec):
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(1920, x2), min(1080, y2)
        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1):
            continue
        a, b, c, d = int(cx1 * s1), int(cy1 * s1), int(np.ceil(cx2 * s1)), int(np.ceil(cy2 * s1))
        v = valid[b:d, a:c]
        if v.size < 16 or v.mean() < 0.6:
            continue
        score = ncc(g1[b:d, a:c][v], warped[b:d, a:c][v])
        out.append({"v1_id": f, "v2_id": int(row.v2_id), "cls": cls, "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                    "ncc": score, "identity": bool(row.identity)})
    return out


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    area = lambda z: (z[2] - z[0]) * (z[3] - z[1])  # noqa: E731
    return i / (area(a) + area(b) - i), i / min(area(a), area(b))


def same_object(a, b, iou_min=0.3, contain_min=0.6):
    iou, contain = box_iou(a, b)
    return iou >= iou_min or contain >= contain_min


def consolidate(cands, ncc_min, ncc_extra):
    """Per frame: boxes from geometrically unaugmented copies are taken as drawn (exact frame pixels);
    boxes projected from augmented copies only add objects not already covered (they are looser).
    A box inside an object of another class with less support is dropped."""
    rows = []
    for f, g in cands[cands.ncc >= ncc_min].groupby("v1_id"):
        objs = []

        def add(b, exact):
            box = (b.x1, b.y1, b.x2, b.y2)
            for o in objs:
                if o["cls"] == b.cls and same_object(o["seed"], box):
                    o["members"].append(box)
                    o["exact"].append(exact)
                    return
            objs.append({"cls": b.cls, "seed": box, "members": [box], "exact": [exact]})

        exact = g[g.identity]
        for b in exact.itertuples():
            add(b, True)
        for b in g[~g.identity & (g.ncc >= ncc_extra)].sort_values("ncc", ascending=False).itertuples():
            add(b, False)
        for o in objs:
            pool = [m for m, e in zip(o["members"], o["exact"]) if e] or o["members"]
            o["box"] = tuple(np.median(np.array(pool), 0))
            o["support"] = len(o["members"])
        keep = []
        for o in sorted(objs, key=lambda o: -o["support"]):
            if any(k["cls"] != o["cls"] and box_iou(k["box"], o["box"])[1] >= 0.7 for k in keep):
                continue
            keep.append(o)
        for o in keep:
            rows.append({"v1_id": f, "cls": o["cls"], "x1": o["box"][0], "y1": o["box"][1], "x2": o["box"][2],
                         "y2": o["box"][3], "support": o["support"], "from_identity": any(o["exact"])})
    return pd.DataFrame(rows)


# ---------- grouping and split ----------

def union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)
    return find, union


def group_frames(ids, pos, pairs, boxes, ppm, link_m, max_shift_m, dup_inliers, dup_shift_m):
    """Union frames that show one moment of the recording:
    (a) overlapping frames <= max_shift_m apart whose obstacle sets match both ways (same class within link_m), or
    (b) near-identical frames (>= dup_inliers RANSAC inliers, < dup_shift_m apart), whatever their labels.
    Linking on any single shared object instead chains almost all frames together (static branches)."""
    idx = {i: k for k, i in enumerate(ids)}
    s1 = SCALE_W / 1920
    centre = np.array([SCALE_W / 2, SCALE_W * 9 / 32])
    objs = {i: [] for i in ids}
    for b in boxes.itertuples():
        objs[b.v1_id].append((b.cls, pos[idx[b.v1_id]] + np.array([(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]) * s1 - centre))

    def covered(X, Y):
        return all(any(ca == cb and np.linalg.norm(ga - gb) / ppm < link_m for cb, gb in Y) for ca, ga in X)

    find, union = union_find(len(ids))
    for a, b, n, _ in pairs:
        if n < PAIR_MIN:
            continue
        shift = np.linalg.norm(pos[idx[a]] - pos[idx[b]]) / ppm
        same_set = objs[a] and objs[b] and covered(objs[a], objs[b]) and covered(objs[b], objs[a])
        if (same_set and shift <= max_shift_m) or (n >= dup_inliers and shift < dup_shift_m):
            union(idx[a], idx[b])
    return np.array([find(k) for k in range(len(ids))])


def placement_split(v1, frame_cls, test_frac, val_frac, min_size, seed):
    """Classes recorded in >= 2 placements: one whole placement goes to test (closest to test_frac of the
    class), one to val when there are >= 3, the rest to train  -> test_kind 'new_placement'.
    Classes recorded in a single placement: frames are ordered along the track; the block at the far end
    goes to test, the adjacent block to val (guard band), the rest to train -> test_kind 'new_view'.
    Small leftover groups are assigned at random with the same fractions."""
    rng = np.random.default_rng(seed)
    v1 = v1.copy()
    v1["dom"] = [max(c, key=lambda k: 0) if len(c) == 1 else (sorted(c)[0] if c else "none") for c in frame_cls]
    v1["split"], v1["test_kind"] = "train", ""
    sizes = v1.groupby("group").size()
    big = sizes[sizes >= min_size].index
    dom = v1[v1.group.isin(big)].groupby("group").dom.agg(lambda s: s.value_counts().index[0])
    for c, gs in dom.groupby(dom).groups.items():
        gs = sorted(gs, key=lambda g: -sizes[g])
        if len(gs) >= 2:
            n_c = sizes[gs].sum()
            test_g = min(gs[1:], key=lambda g: abs(sizes[g] - test_frac * n_c))
            v1.loc[v1.group == test_g, ["split", "test_kind"]] = ["test", "new_placement"]
            rest = [g for g in gs[1:] if g != test_g]
            if rest:
                v1.loc[v1.group == rest[-1], "split"] = "val"
        else:
            g = gs[0]
            idx = v1[v1.group == g].sort_values("y_m").index
            n_t, n_v = round(len(idx) * test_frac), round(len(idx) * val_frac)
            v1.loc[idx[len(idx) - n_t:], ["split", "test_kind"]] = ["test", "new_view"]
            v1.loc[idx[len(idx) - n_t - n_v: len(idx) - n_t], "split"] = "val"
    for g in sizes[sizes < min_size].index:
        r = rng.random()
        s = "test" if r < test_frac else ("val" if r < test_frac + val_frac else "train")
        v1.loc[v1.group == g, ["split", "test_kind"]] = [s, "new_placement" if s == "test" else ""]
    return v1.drop(columns="dom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-inliers", type=int, default=25, help="V2 match accepted at or above this")
    ap.add_argument("--ncc-min", type=float, default=0.7, help="box content must match the frame at this NCC")
    ap.add_argument("--ncc-extra", type=float, default=0.8, help="NCC needed for boxes from augmented copies")
    ap.add_argument("--link-m", type=float, default=0.5, help="same-class objects closer than this are the same object")
    ap.add_argument("--max-shift-m", type=float, default=1.0)
    ap.add_argument("--dup-inliers", type=int, default=400)
    ap.add_argument("--dup-shift-m", type=float, default=0.2)
    ap.add_argument("--min-group", type=int, default=5, help="groups at least this big count as a placement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for d in (CACHE, SPLITS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    ids = v1_ids()
    print("[1/5] V1 features + pairwise matching")
    feats = v1_features()
    pairs = v1_pairs(feats, args.workers)
    pos, res = solve_positions(ids, pairs)
    ppm = px_per_m(ids)
    print(f"  {len(pairs)} matched pairs, {sum(p[2] >= PAIR_MIN for p in pairs)} strong; "
          f"residual p50 {np.median(res):.1f}px; {ppm:.1f} px/m at matching scale")

    print("[2/5] V2 -> V1 matching")
    recs = v2_records()
    rid = {r["id"]: r for r in recs}
    m = match_v2(feats, recs, args.workers)
    m = m.merge(pd.DataFrame([{"v2_id": r["id"], "author_split": r["author_split"],
                               "cls": "|".join(sorted({b[0] for b in r["boxes"]}))} for r in recs]), on="v2_id")
    m["matched"] = m["inliers"] >= args.min_inliers
    print(f"  matched {m.matched.sum()}/{len(m)}; geometric identity {m.identity.sum()}")

    print("[3/5] consolidate obstacle boxes per frame")
    f = CACHE / "box_candidates.csv"
    if f.exists():
        cands = pd.read_csv(f)
    else:
        jobs = [(SimpleNamespace(**r), rid[int(r["v2_id"])]) for r in m[m.matched].to_dict("records")]
        with ctx.Pool(args.workers, initializer=_init_verify) as p:
            cands = pd.DataFrame([b for out in p.imap_unordered(_verify, jobs, chunksize=16) for b in out])
        cands.to_csv(f, index=False)
    boxes = consolidate(cands, args.ncc_min, args.ncc_extra)
    boxes.to_csv(SPLITS / "v1_boxes.csv", index=False)
    n_obj = boxes.groupby("v1_id").size().reindex(ids, fill_value=0)
    print(f"  {len(cands)} projected boxes, {(cands.ncc >= args.ncc_min).mean():.0%} pass NCC; "
          f"objects/frame {n_obj.value_counts().sort_index().to_dict()}")

    print("[4/5] group frames into obstacle placements, stratified split")
    groups = group_frames(ids, pos, pairs, boxes, ppm, args.link_m, args.max_shift_m, args.dup_inliers, args.dup_shift_m)
    frame_cls = [set(boxes[boxes.v1_id == i].cls) for i in ids]
    v1 = pd.DataFrame({"v1_id": ids, "x_px": pos[:, 0], "y_px": pos[:, 1], "y_m": pos[:, 1] / ppm,
                       "group": groups, "n_obj": n_obj.values})
    v1 = placement_split(v1, frame_cls, args.test_frac, args.val_frac, args.min_group, args.seed)
    sp = dict(zip(v1.v1_id, v1.split))
    m["split"] = np.where(m.matched, m.v1_id.map(sp), "unmatched")
    cross = sum(1 for a, b, n, _ in pairs if n >= PAIR_MIN and {sp[a], sp[b]} == {"train", "test"})

    print("[5/5] write")
    v1.to_csv(SPLITS / "v1.csv", index=False)
    m.to_csv(SPLITS / "v2.csv", index=False)
    bx = boxes.merge(v1[["v1_id", "split"]], on="v1_id")
    gsize = pd.Series(groups).value_counts()
    summary = {
        "px_per_m_matching_scale": round(ppm, 2),
        "px_per_m_full_res": round(ppm * 1920 / SCALE_W, 1),
        "track_span_m": round(float((pos[:, 1].max() - pos[:, 1].min()) / ppm), 1),
        "v2_matched": int(m.matched.sum()), "v2_unmatched": int((~m.matched).sum()),
        "v2_geometric_identity": int(m.identity.sum()),
        "projected_boxes": len(cands), "projected_boxes_pass_ncc": int((cands.ncc >= args.ncc_min).sum()),
        "objects_per_frame": {int(k): int(v) for k, v in n_obj.value_counts().sort_index().items()},
        "v2_labels_per_image": "1 (2001/2002 images)",
        "groups": int(len(gsize)), "group_size_max": int(gsize.max()), "group_size_median": float(gsize.median()),
        "frames_per_split": v1.split.value_counts().to_dict(),
        "test_frames_by_kind": v1[v1.split == "test"].test_kind.value_counts().to_dict(),
        "placements": {str(k): v for k, v in v1.merge(boxes.groupby("v1_id").cls.first().rename("c"), on="v1_id", how="left")
                       .groupby("group").agg(n=("v1_id", "size"), cls=("c", lambda s: s.value_counts().index[0] if s.notna().any() else "none"),
                                             split=("split", lambda s: "/".join(sorted(set(s))))).query("n >= 5").to_dict(orient="index").items()},
        "objects_per_split_class": bx.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "strong_pairs_crossing_train_test": cross,
        "note": "all frames come from the same ~10 m of track; splits separate obstacle placements, not track locations",
    }
    json.dump(summary, open(REPORTS / "split_summary.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 8))
    for s, c in {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}.items():
        d = v1[v1.split == s]
        ax.scatter(d.x_px / ppm, d.y_m, s=10, c=c, label=f"{s} ({len(d)})")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("across track (m)")
    ax.set_ylabel("along track (m)")
    ax.set_title("V1 frame centres by split")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "split_positions.png", dpi=110)


if __name__ == "__main__":
    main()
