"""Independent structural checks of a classified symbol.

The CNN is good at the overall shape; safety-critical details – a crossing
line, the number of dots, the number of bars – are re-measured here with plain
geometry. A disagreement does not change the class; it lowers the confidence
(and usually turns the symbol into an "uncertain" one). See docs/algorithm.md.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .normalize import normalize_crop

DISAGREE_FACTOR = {"cross": 0.35, "dots": 0.4, "bars": 0.55, "outline": 0.3}
HOLE_MIN = 0.05


def _best_line_coverage(b: np.ndarray, box, anti: bool) -> float:
    """Max ink coverage of a straight line running corner-to-corner of `box`
    (searched over small offsets of both end points, tolerating shear/rotation)."""
    x, y, w, h = box
    offs = np.linspace(0.0, 0.28, 5)
    best = 0.0
    ts = np.linspace(0.1, 0.9, 48)
    H, W = b.shape
    for a in offs:
        for side_a in (0, 1):
            # start point near the top-left (or top-right) corner
            sx, sy = (x + a * w, y) if side_a == 0 else (x, y + a * h)
            for c in offs:
                for side_c in (0, 1):
                    ex, ey = (x + w - c * w, y + h) if side_c == 0 else (x + w, y + h - c * h)
                    if anti:
                        sx2, ex2 = 2 * x + w - sx, 2 * x + w - ex
                    else:
                        sx2, ex2 = sx, ex
                    px = np.clip(np.round(sx2 + (ex2 - sx2) * ts).astype(int), 0, W - 1)
                    py = np.clip(np.round(sy + (ey - sy) * ts).astype(int), 0, H - 1)
                    cov = float(b[py, px].mean())
                    if cov > best:
                        best = cov
    return best


def border_solid(norm64: np.ndarray) -> float:
    """Share of the 64x64 ink map covered by SOLID ink regions that touch the crop border.

    Symbols are centred in their crop with a margin and drawn with strokes; a solid dark area
    entering from the border is a hard shadow, a finger, the garment or a neighbouring object
    that the ink map could not separate from the symbol. On the validation crops the classifier
    is right only ~50-80 % of the time on such crops (vs ~94-97 % otherwise), so the result must
    not be reported as confident (see classifier.OCCLUSION_*)."""
    b = (norm64 > 0.5).astype(np.uint8)
    e = cv2.erode(b, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    H, W = b.shape
    tot = 0
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            tot += a
    return tot / float(H * W)


def measure(gray_crop: np.ndarray) -> dict:
    """Geometric measurements on a symbol crop (dark ink)."""
    S = 128
    norm = normalize_crop(gray_crop, S)
    u8 = (norm * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(thr / 255.0, 0.25, 0.6))
    b = (norm > thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = {"src_px": int(max(gray_crop.shape))}
    if n <= 1:
        out["empty"] = True
        return out
    main = 1 + int(np.argmax(st[1:, 2] * st[1:, 3]))
    mx, my, mw, mh, ma = st[main]
    out["main_box"] = [int(mx), int(my), int(mw), int(mh)]
    mb = (lab == main).astype(np.uint8)
    # ---- enclosed region: every care symbol outline (tub+water line, triangle, square,
    #      iron, circle) encloses background. Letters such as L, X, 1, 7 do not.
    cnts, hier = cv2.findContours(mb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole = 0.0
    if hier is not None:
        for ci, hh in enumerate(hier.reshape(-1, 4)):
            if hh[3] >= 0:  # has a parent -> it is a hole
                hole += cv2.contourArea(cnts[ci])
    out["hole_frac"] = round(float(hole) / float(max(1, mw * mh)), 3)
    # ---- cross: two straight corner-to-corner strokes through thin ink only
    k = max(3, int(0.16 * min(mw, mh)))
    solid = cv2.morphologyEx(mb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thin = cv2.dilate(mb & (1 - solid), np.ones((3, 3), np.uint8))
    c1 = _best_line_coverage(thin, (mx, my, mw, mh), anti=False)
    c2 = _best_line_coverage(thin, (mx, my, mw, mh), anti=True)
    out["diag_cov"] = [round(c1, 2), round(c2, 2)]
    has_solid = solid.sum() > 0.04 * mw * mh
    out["cross"] = bool(min(c1, c2) >= 0.9)
    out["cross_conf"] = bool(min(c1, c2) >= 0.95 or (max(c1, c2) <= 0.8 and not has_solid))
    # ---- dots: small, round, solid blobs inside the main outline (stricter threshold
    #      so that two blurred neighbouring dots do not merge)
    bd = (norm > max(thr, 0.6)).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(bd, connectivity=4)
    dots, glyphs = [], 0
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        cx, cy = x + w / 2, y + h / 2
        if not (mx + 0.04 * mw <= cx <= mx + 0.96 * mw and my + 0.04 * mh <= cy <= my + 0.96 * mh):
            continue
        if (lab2 == i).__and__(mb.astype(bool)).sum() > 0.5 * a and max(w, h) > 0.2 * max(mw, mh):
            continue  # part of the outline
        fill = a / float(w * h)
        ar = w / float(h)
        rel = max(w, h) / float(max(mw, mh))
        if 0.015 <= rel <= 0.14 and 0.45 <= ar <= 2.2 and fill >= 0.4:
            dots.append((i, a, ar))
        elif rel > 0.12 and h >= 0.15 * mh:
            glyphs += 1
    cnt, merged = 0, False
    if dots:
        med = np.median([a for _, a, _ in dots])
        for _, a, ar in dots:
            if a < med / 2.5:
                continue  # speck
            if a > 1.8 * med or ar > 1.6 or ar < 0.62:
                merged = True  # blurred neighbouring dots fused: count unknown
            cnt += 1
    out["dots"] = int(cnt)
    out["dots_merged"] = merged
    out["glyphs"] = glyphs
    # ---- bars: wide flat components below the main shape
    bars = []
    for i in range(1, n):
        if i == main:
            continue
        x, y, w, h, a = st[i]
        if w >= 0.45 * mw and h <= 0.3 * max(mh, 1) and w >= 3 * h and y >= my + 0.6 * mh:
            bars.append(y)
    out["bars"] = len(bars)
    return out


import os as _os
ENABLED_CHECKS = set(_os.environ.get("CLR_VCHECKS", "cross,bars").split(","))


def check(meas: dict, exp: dict) -> tuple[float, dict]:
    """Compare measurements with the class expectations -> (factor, details)."""
    if not exp or meas.get("empty"):
        return 1.0, {}
    det = {}
    factor = 1.0
    # enclosed outline (filled shapes have no hole but are not letters either)
    if "outline" in ENABLED_CHECKS and meas.get("hole_frac", 1.0) < HOLE_MIN and True not in exp.get("filled", set()):
        det["outline"] = f"open shape (hole {meas.get('hole_frac')})"
        factor *= DISAGREE_FACTOR["outline"] if meas.get("src_px", 0) >= 34 else 0.6
    reliable = meas.get("src_px", 0) >= 34   # below this, dots/bars are not resolvable
    # crossing
    if "cross" in ENABLED_CHECKS and meas.get("cross_conf") and exp.get("steam_cross") != {True}:
        ok = meas["cross"] in exp["cross"]
        det["cross"] = "agree" if ok else "disagree"
        if not ok:
            factor *= DISAGREE_FACTOR["cross"]
    # dots
    dexp = exp.get("dots", set())
    if "dots" in ENABLED_CHECKS and reliable and dexp and exp.get("cross") != {True}:
        d = meas.get("dots", 0)
        g = meas.get("glyphs", 0)
        if "numeral" in dexp and g >= 1:
            det["dots"] = "agree(numeral)"
        elif dexp == {0} and "tub" in exp.get("shape", set()) and g >= 1:
            det["dots"] = "disagree(temperature digits present)"
            factor *= DISAGREE_FACTOR["dots"]
        elif d in dexp:
            det["dots"] = "agree"
        elif meas.get("dots_merged"):
            det["dots"] = f"unknown(merged, measured {d})"
        elif "numeral" in dexp and d == 0:
            det["dots"] = "agree(numeral)" if g >= 1 else "unknown"
        else:
            det["dots"] = f"disagree(measured {d})"
            factor *= DISAGREE_FACTOR["dots"]
    # bars
    if "bars" in ENABLED_CHECKS and reliable and exp.get("bars") and exp.get("shape", set()) & {"tub", "square", "circle"}:
        bm = min(meas.get("bars", 0), 2)
        if bm in exp["bars"]:
            det["bars"] = "agree"
        elif bm < min(exp["bars"]):
            det["bars"] = f"unknown(measured {bm})"   # a bar merged with the outline is common
        else:
            det["bars"] = f"disagree(measured {bm})"
            factor *= DISAGREE_FACTOR["bars"]
    return factor, det
