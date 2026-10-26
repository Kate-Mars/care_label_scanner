"""Shared test helpers: deterministic synthetic labels."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care_label_recognition.taxonomy import load  # noqa: E402
from synth.renderer import render_symbol  # noqa: E402

TAX = load()


def clean_label(class_ids, sym_px=90, gap=40, text=None, seed=0, variant=0, size=None):
    """White label with the given symbols in a row (dark ink), optional text line.
    Returns (bgr image, list of gt boxes)."""
    rng = np.random.default_rng(seed)
    n = len(class_ids)
    W = size[0] if size else int(60 + n * (sym_px + gap) + 60)
    H = size[1] if size else int(sym_px * 2.6)
    img = np.full((H, W), 240, np.float32)
    boxes = []
    x = 60
    for cid in class_ids:
        specs = TAX.render_specs(cid)
        spec = specs[min(variant, len(specs) - 1)]
        m = render_symbol(spec, rng, size=int(sym_px * 2.2), split="train", stroke=0.06)
        ys, xs = np.where(m > 0.3)
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sc = sym_px / max(m.shape)
        m = cv2.resize(m, (max(4, int(m.shape[1] * sc)), max(4, int(m.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        mh, mw = m.shape
        y = int(sym_px * 0.5 + (sym_px - mh) / 2)
        img[y:y + mh, x:x + mw] = img[y:y + mh, x:x + mw] * (1 - m) + 20 * m
        boxes.append([x, y, mw, mh])
        x += mw + gap
    img = np.clip(img, 0, 255).astype(np.uint8)
    if text:
        cv2.putText(img, text, (60, int(sym_px * 2.3)), cv2.FONT_HERSHEY_SIMPLEX, sym_px / 90, 20, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), boxes


def on_background(label_bgr, angle=0.0, persp=0.0, bg=90, seed=0):
    """Put a label on a darker garment with rotation / perspective."""
    rng = np.random.default_rng(seed)
    h, w = label_bgr.shape[:2]
    PW, PH = int(w * 1.6), int(h * 2.2)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c = np.float32([PW / 2, PH / 2])
    th = np.radians(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    dst = (src - [w / 2, h / 2]) @ R.T + c
    dst += rng.uniform(-persp, persp, (4, 2)).astype(np.float32) * [w, h]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    photo = np.full((PH, PW, 3), bg, np.uint8)
    warped = cv2.warpPerspective(label_bgr, M, (PW, PH))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (PW, PH))
    photo[mask > 128] = warped[mask > 128]
    return photo, M
