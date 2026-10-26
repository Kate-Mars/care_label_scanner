"""Camera-like degradations used for synthetic data.

Everything here imitates a phone photo of a printed/woven label: fabric
texture, uneven light, shadows, perspective, defocus/motion blur, sensor noise
and JPEG compression. None of the operations change the meaning of a symbol
(no erasing of dots/bars, no removal of crossings).
"""
from __future__ import annotations

import cv2
import numpy as np


def fabric_texture(h, w, rng, base=None, amp=None):
    base = rng.uniform(170, 250) if base is None else base
    amp = rng.uniform(2, 14) if amp is None else amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = rng.uniform(2.5, 7.0)
    ang = rng.uniform(0, np.pi)
    t = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * 2 * np.pi / period)
    t2 = np.sin((-xx * np.sin(ang) + yy * np.cos(ang)) * 2 * np.pi / (period * rng.uniform(0.8, 1.3)))
    tex = base + amp * 0.5 * (t * t2) + rng.normal(0, amp * 0.35, (h, w))
    return tex.astype(np.float32)


def illumination(h, w, rng, strength=None):
    strength = rng.uniform(0.0, 0.45) if strength is None else strength
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    g = (xx / max(w, 1)) * np.cos(ang) + (yy / max(h, 1)) * np.sin(ang)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return 1.0 - strength * g


def shadow(h, w, rng):
    m = np.ones((h, w), np.float32)
    n = rng.integers(3, 6)
    pts = np.stack([rng.uniform(-0.2, 1.2, n) * w, rng.uniform(-0.2, 1.2, n) * h], 1).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(m, hull, float(rng.uniform(0.45, 0.8)))
    k = int(max(3, min(h, w) * rng.uniform(0.05, 0.2))) | 1
    return cv2.GaussianBlur(m, (k, k), 0)


def blur(img, rng, max_sigma=1.6):
    r = rng.random()
    if r < 0.25:
        return img
    if r < 0.85:
        s = rng.uniform(0.3, max_sigma)
        return cv2.GaussianBlur(img, (0, 0), s)
    # motion blur
    L = int(rng.integers(3, 7))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0 / L
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, M, (L, L))
    k /= k.sum() + 1e-6
    return cv2.filter2D(img, -1, k)


def noise_jpeg(img, rng, max_noise=7.0):
    img = img + rng.normal(0, rng.uniform(0, max_noise), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        q = int(rng.integers(35, 95))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return img


def random_affine_mask(m, rng, rot=10, shear=0.12, aspect=0.15, persp=0.06):
    """Mild geometric distortion of an ink mask (keeps meaning)."""
    h, w = m.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    a = rng.uniform(1 - aspect, 1 + aspect)
    dst = src.copy()
    dst[:, 0] = (dst[:, 0] - w / 2) * a + w / 2
    dst[:, 0] += (dst[:, 1] - h / 2) * rng.uniform(-shear, shear)
    dst += rng.uniform(-persp, persp, dst.shape).astype(np.float32) * w
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot, rot), 1.0), [0, 0, 1]])
    M = R @ M
    return cv2.warpPerspective(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def stroke_jitter(m, rng):
    """Thicken or thin strokes slightly (woven labels bleed, prints are thin)."""
    r = rng.random()
    k = np.ones((3, 3), np.uint8)
    if r < 0.2:
        return cv2.dilate(m, k)
    if r < 0.35:
        e = cv2.erode(m, k)
        return e if e.sum() > 0.4 * m.sum() else m
    return m


def isolated_sample(mask, rng, out_px=None):
    """Ink mask -> realistic gray crop (uint8, dark ink) around the symbol.

    Returns gray crop with random padding jitter that imitates the detector's
    bounding boxes.
    """
    m = random_affine_mask(mask, rng)
    m = stroke_jitter(m, rng)
    ys, xs = np.where(m > 0.35)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    # resolution of the symbol on the photo
    target = out_px or rng.uniform(22, 110)
    scale = target / max(bh, bw)
    # crop window with jitter (detector bbox + padding)
    pad = max(bh, bw) * 0.08
    jit = lambda: rng.uniform(-0.06, 0.06) * max(bh, bw)
    cy0 = int(max(0, y0 - pad + jit()))
    cy1 = int(min(m.shape[0], y1 + pad + jit()))
    cx0 = int(max(0, x0 - pad + jit()))
    cx1 = int(min(m.shape[1], x1 + pad + jit()))
    m = m[cy0:cy1, cx0:cx1]
    nh, nw = max(8, int(m.shape[0] * scale)), max(8, int(m.shape[1] * scale))
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    bg = fabric_texture(nh, nw, rng) * illumination(nh, nw, rng)
    ink_val = rng.uniform(0, 110)
    alpha = rng.uniform(0.75, 1.0)
    img = bg * (1 - alpha * m) + ink_val * alpha * m
    if rng.random() < 0.12:
        img = img * shadow(nh, nw, rng)
    img = blur(img, rng, max_sigma=max(0.4, min(1.6, target / 40)))
    return noise_jpeg(img, rng)
