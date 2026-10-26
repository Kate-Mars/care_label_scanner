"""Find the care label in the photo and rectify its perspective.

Candidates are quadrilaterals from (a) edge contours and (b) bright/dark
homogeneous regions. Each candidate is scored by area, rectangularity and by
whether its border is a *step* edge (label vs. garment) rather than a drawn dark
line – this keeps the square frame of a drying symbol from being taken for the
label in close-up shots. If no convincing quad is found, the full frame is used
(and the recognizer lowers its confidence slightly).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RECT_MAX_SIDE = 1600


@dataclass
class LabelRegion:
    found: bool
    quad: np.ndarray | None          # 4x2 float32, working-image coordinates (tl, tr, br, bl)
    score: float
    rectified: np.ndarray            # gray image the symbols are searched in
    M: np.ndarray                    # 3x3: working -> rectified
    Minv: np.ndarray                 # 3x3: rectified -> working
    method: str


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    q = np.stack([tl, tr, br, bl])
    if len({tuple(p) for p in q}) < 4:  # degenerate ordering -> angle sort
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        q = pts[np.argsort(ang)]
        k = np.argmin(q.sum(1))
        q = np.roll(q, -k, axis=0)
    return q


def _quad_candidates(gray: np.ndarray):
    h, w = gray.shape
    area = h * w
    out = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # (a) edges
    for lo, hi in ((30, 90), (60, 160)):
        e = cv2.Canny(blur, lo, hi)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out += [(c, "edges") for c in cnts if cv2.contourArea(c) > 0.02 * area]
    # (b) homogeneous bright or dark regions (label fabric differs from garment)
    thr, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 40),) * 2)
    for m in (b, 255 - b):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out += [(c, "region") for c in cnts if cv2.contourArea(c) > 0.02 * area]
    quads = []
    for c, how in out:
        peri = cv2.arcLength(c, True)
        q = None
        for eps in (0.02, 0.035, 0.05):
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                q = ap.reshape(4, 2).astype(np.float32)
                break
        if q is None:  # fall back to the rotated bounding rectangle
            q = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        rect = cv2.contourArea(c) / max(cv2.contourArea(q), 1.0)
        quads.append((order_quad(q), rect, how))
    return quads


def _border_profile(gray, q):
    """Mean intensity inside / on / outside the quad border."""
    h, w = gray.shape
    c = q.mean(0)
    vals = {}
    for name, f in (("in", 0.84), ("on", 1.0), ("out", 1.14)):
        qq = (q - c) * f + c
        m = np.zeros((h, w), np.uint8)
        cv2.polylines(m, [qq.astype(np.int32)], True, 255, max(2, int(0.006 * max(h, w))))
        sel = m > 0
        vals[name] = float(gray[sel].mean()) if sel.any() else np.nan
    return vals


def detect_label(gray: np.ndarray) -> LabelRegion:
    h, w = gray.shape
    s = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    sh, sw = small.shape
    best, best_score, best_how = None, 0.0, ""
    for q, rect, how in _quad_candidates(small):
        a = cv2.contourArea(q) / (sh * sw)
        if a < 0.02 or a > 0.97:
            continue
        sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        if sides.min() < 0.08 * max(sh, sw):
            continue
        # interior angles should be roughly rectangular (perspective allowed)
        v1 = np.roll(q, 1, axis=0) - q
        v2 = np.roll(q, -1, axis=0) - q
        cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        if np.abs(cosang).max() > 0.55:
            continue
        prof = _border_profile(small, q)
        step = abs(prof["in"] - prof["out"])
        line_like = (prof["on"] < prof["in"] - 25) and (prof["on"] < prof["out"] - 25) and step < 30
        if line_like:  # a drawn frame (e.g. the square of a drying symbol) – not a label edge
            continue
        score = min(1.0, rect) * (0.5 + 0.5 * min(1.0, step / 40.0)) * (0.6 + 0.4 * min(1.0, a / 0.4))
        # a quad whose sides run along the photo border is usually a shadow / garment area
        tol = 0.015 * max(sh, sw)
        on_border = 0
        for p0, p1 in zip(q, np.roll(q, -1, axis=0)):
            for axis, lim in ((0, 0), (0, sw - 1), (1, 0), (1, sh - 1)):
                if abs(p0[axis] - lim) < tol and abs(p1[axis] - lim) < tol:
                    on_border += 1
        score *= 0.7 ** on_border
        if score > best_score:
            best, best_score, best_how = q, score, how
    if best is None or best_score < 0.45:
        I = np.eye(3, dtype=np.float64)
        return LabelRegion(False, None, 0.0, gray, I, I, "full_frame")
    q = best / s
    wt = np.linalg.norm(q[1] - q[0])
    wb = np.linalg.norm(q[2] - q[3])
    hl = np.linalg.norm(q[3] - q[0])
    hr = np.linalg.norm(q[2] - q[1])
    W, H = max(wt, wb), max(hl, hr)
    k = min(1.0, RECT_MAX_SIDE / max(W, H))
    W, H = int(max(32, W * k)), int(max(32, H * k))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    rect_img = cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return LabelRegion(True, q.astype(np.float32), float(best_score), rect_img, M.astype(np.float64),
                       np.linalg.inv(M).astype(np.float64), best_how)


def map_box(box, Minv: np.ndarray) -> list[int]:
    """[x, y, w, h] in rectified coords -> axis-aligned [x, y, w, h] in working coords."""
    x, y, w, h = box
    pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def estimate_skew(gray: np.ndarray) -> float:
    """Dominant in-plane rotation (degrees, in (-45, 45]) of a label photo.

    Text lines, symbol outlines and label edges are mostly horizontal/vertical in
    the label frame, so the magnitude-weighted histogram of gradient orientations
    (taken modulo 90°) peaks at the skew angle. Returns 0 when there is no clear peak.
    """
    s = 480.0 / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90)
    m = mag > max(thr, 1e-3)
    if m.sum() < 200:
        return 0.0
    ang = (np.degrees(np.arctan2(gy[m], gx[m])) % 90.0)
    hist, _ = np.histogram(ang, bins=90, range=(0, 90), weights=mag[m])
    k = np.array([1, 2, 3, 2, 1], np.float32)
    hs = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), k / k.sum(), mode="valid")
    peak = int(np.argmax(hs))
    if hs[peak] < 1.6 * np.median(hs):
        return 0.0
    # sub-bin refinement
    l, c, r = hs[(peak - 1) % 90], hs[peak], hs[(peak + 1) % 90]
    off = 0.5 * (l - r) / (l - 2 * c + r) if (l - 2 * c + r) != 0 else 0.0
    a = (peak + 0.5 + off) % 90.0
    return float(a - 90.0 if a > 45 else a)


def rotate_for_search(gray: np.ndarray, angle: float):
    """Rotate by ``angle`` degrees (canvas expanded, border = edge median) and return
    (rotated, Minv) where Minv maps rotated -> original coordinates (3x3)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    rot = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    M3 = np.vstack([M, [0, 0, 1]])
    return rot, np.linalg.inv(M3)
