"""Normalisation of a single symbol crop and feature extraction.

The exact same functions are used when training the classifier and at
recognition time, so the model always sees identically prepared inputs.
"""
from __future__ import annotations

import cv2
import numpy as np

NORM_SIZE = 64



def hog(img: np.ndarray, cell: int, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients (unsigned, 2x2-cell blocks, L2-Hys).

    Small self-contained implementation (OpenCV 5 dropped HOGDescriptor).
    """
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins  # [0, bins)
    b0 = np.floor(ang).astype(np.int32) % bins
    w1 = ang - np.floor(ang)
    b1 = (b0 + 1) % bins
    h, w = img.shape
    ch, cw = h // cell, w // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    cy = (np.arange(h) // cell).clip(max=ch - 1)
    cx = (np.arange(w) // cell).clip(max=cw - 1)
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    np.add.at(hist, (CY, CX, b0), mag * (1 - w1))
    np.add.at(hist, (CY, CX, b1), mag * w1)
    blocks = np.concatenate([hist[:-1, :-1], hist[1:, :-1], hist[:-1, 1:], hist[1:, 1:]], axis=-1)
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    return blocks.ravel()



def ink_map(gray: np.ndarray) -> np.ndarray:
    """Dark-ink-on-light gray crop -> ink intensity in [0, 1] (1 = ink).

    * local term: darkness relative to a morphological background estimate
      (closing with a kernel larger than any stroke) – invariant to uneven light
      and to soft or hard shadows (shadows are multiplicative);
    * solid term: large filled shapes (filled circle, hand) are wider than the
      kernel and vanish from the local term, so pixels that are as dark as the
      strokes themselves are added back.
    Contrast is stretched between background and stroke level; no smoothing of
    the strokes – thin dots, bars and letters must survive.
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 4 or w < 4:
        return np.zeros_like(g)
    k = max(5, int(0.25 * min(h, w)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)
    bg = np.maximum(cv2.GaussianBlur(bg, (0, 0), k / 6.0), g)
    local = np.clip((bg - g) / np.maximum(bg, 1.0), 0, 1)
    s99 = float(np.percentile(local, 99.5))
    if s99 < 0.06:  # no ink at all
        return np.zeros_like(g)
    strokes = local > 0.5 * s99
    lo = float(np.percentile(local[~strokes], 90)) if (~strokes).any() else 0.0
    hi = float(np.percentile(local[strokes], 70))
    loc_n = np.clip((local - lo) / max(hi - lo, 1e-3), 0, 1)
    ref = float(np.percentile(g, 92))
    dark = np.clip((ref - g) / max(ref, 1.0), 0, 1)
    s_ink = float(np.median(dark[strokes]))
    solid = np.clip((dark - (s_ink - 0.12)) / 0.12, 0, 1) if s_ink > 0.2 else np.zeros_like(g)
    return np.maximum(loc_n, solid)


def normalize_crop(gray: np.ndarray, size: int = NORM_SIZE) -> np.ndarray:
    """Crop (uint8, dark ink) -> float32 (size, size) ink map, aspect preserved, centred."""
    lim = int(size * 2)
    if max(gray.shape) > lim:  # work at ~2x the output resolution (speed; detail is kept)
        f = lim / max(gray.shape)
        gray = cv2.resize(gray, (max(4, int(round(gray.shape[1] * f))), max(4, int(round(gray.shape[0] * f)))), interpolation=cv2.INTER_AREA)
    ink = ink_map(gray)
    h, w = ink.shape
    s = max(h, w)
    canvas = np.zeros((s, s), np.float32)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = ink
    interp = cv2.INTER_AREA if s > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interp)


def features(norm: np.ndarray) -> np.ndarray:
    """Feature vector for a normalised crop: fine HOG + coarse HOG + 16x16 intensity."""
    u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    f1 = hog(u8, 8)
    f2 = hog(u8, 16)
    small = cv2.resize(norm, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([f1, f2, small]).astype(np.float32)


def features_batch(norms) -> np.ndarray:
    return np.stack([features(n) for n in norms]) if len(norms) else np.zeros((0, 2344), np.float32)
