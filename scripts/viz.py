"""Drawing helpers (Vietnamese text via PIL)."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import FONT

MASK_COLORS = {"rail_inside": (60, 200, 60), "rail_lines": (0, 220, 255), "track": (60, 200, 60)}  # BGR
BOX_COLORS = {"Branch": (40, 160, 40), "IronRod": (200, 120, 0), "Barrel": (220, 60, 20),
              "Boulder": (120, 120, 120), "Jerrycan": (0, 140, 255), "Person": (180, 0, 200)}


@lru_cache(maxsize=8)
def font(size):
    return ImageFont.truetype(str(FONT), size)


def overlay_masks(img, masks, alpha=0.45):
    out = img.copy()
    for name, m in masks.items():
        m = m.astype(bool)
        color = np.array(MASK_COLORS.get(name, (60, 200, 60)), np.float32)
        out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def put_text(img, xy, text, size, fg=(255, 255, 255), bg=(0, 0, 0), pad=6):
    """Draw text with a filled background box; colors are BGR."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font(size)
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=f)
    if bg is not None:
        d.rectangle([l - pad, t - pad, r + pad, b + pad], fill=bg[::-1])
    d.text((x, y), text, font=f, fill=fg[::-1])
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_boxes(img, boxes, color=None, thickness=None, size=None):
    """boxes: [(label, (x1, y1, x2, y2), score or None)]"""
    h, w = img.shape[:2]
    thickness = thickness or max(2, round(max(h, w) / 400))
    size = size or max(18, round(max(h, w) / 55))
    out = img.copy()
    for label, (x1, y1, x2, y2), score in boxes:
        c = color or BOX_COLORS.get(label, (0, 0, 255))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, thickness)
    for label, (x1, y1, x2, y2), score in boxes:
        c = color or BOX_COLORS.get(label, (0, 0, 255))
        text = label if score is None else f"{label} {score:.2f}"
        ty = int(y1) - size - 10 if y1 > size + 14 else int(y1) + 8
        out = put_text(out, (int(x1) + 4, ty), text, size, bg=c)
    return out
