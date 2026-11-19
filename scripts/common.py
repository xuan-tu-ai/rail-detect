"""Shared paths, class names and dataset loaders."""
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
V1 = RAW / "V1 UAV-RSOD_Dataset for Segmentation"
V2 = RAW / "V2 UAV-RSOD_Dataset for Obstacle Detection" / "images"
CACHE = DATA / "cache"
SPLITS = DATA / "splits"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
FONT = ROOT / "assets" / "NotoSans-Bold.ttf"

RAIL_CLASSES = ["rail_inside", "rail_lines"]
RAIL_DIRS = {"rail_inside": "Rail Inside", "rail_lines": "Rail Lines"}
OBST_CLASSES = ["Branch", "IronRod", "Barrel", "Boulder", "Jerrycan", "Person"]
OBST_VI = {
    "Branch": "Cành cây",
    "IronRod": "Thanh sắt",
    "Barrel": "Thùng phuy",
    "Boulder": "Tảng đá",
    "Jerrycan": "Can nhựa",
    "Person": "Người",
}
GAUGE_M = 1.676  # Indian broad gauge


def v1_ids():
    return sorted(int(p.stem) for p in (V1 / "1 Images").glob("*.jpg"))


def v1_image(i: int) -> Path:
    return V1 / "1 Images" / f"{i}.jpg"


def v1_mask(i: int, cls: str) -> np.ndarray:
    """Full-resolution binary mask extracted from the red overlay in '2.1 Labelling'.

    The provided '2.2 Masking' files are partly stored at 334x188; the overlay is always 1920x1080
    and agrees with the full-res masks (IoU 0.998 rail_inside, 0.983 rail_lines).
    """
    img = cv2.imread(str(v1_image(i))).astype(np.int16)
    lab = cv2.imread(str(V1 / "2 Annotations" / "2.1 Labelling" / RAIL_DIRS[cls] / f"{i}.jpg")).astype(np.int16)
    b, g, r = lab[..., 0], lab[..., 1], lab[..., 2]
    diff = np.abs(lab - img).sum(-1)
    m = ((r > 180) & (g < 110) & (b < 110) & (diff > 60)).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def v1_mask_provided(i: int, cls: str) -> np.ndarray:
    m = cv2.imread(str(V1 / "2 Annotations" / "2.2 Masking" / RAIL_DIRS[cls] / f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
    if m.shape != (1080, 1920):
        m = cv2.resize(m, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return (m > 127).astype(np.uint8)


def v2_records():
    """One dict per V2 image: id, author split, path, size, boxes [(cls, x1, y1, x2, y2)]."""
    out = []
    for split in ["train", "test"]:
        for x in sorted((V2 / split).glob("*.xml"), key=lambda p: int(p.stem)):
            t = ET.parse(x).getroot()
            boxes = []
            for o in t.findall("object"):
                b = o.find("bndbox")
                boxes.append((o.find("name").text, *[float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]))
            out.append({
                "id": int(x.stem),
                "author_split": split,
                "path": x.with_suffix(".jpg"),
                "w": int(t.find("size/width").text),
                "h": int(t.find("size/height").text),
                "boxes": boxes,
            })
    return sorted(out, key=lambda r: r["id"])
