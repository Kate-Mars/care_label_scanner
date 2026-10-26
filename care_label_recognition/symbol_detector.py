"""Symbol candidate detection and segmentation.

Graphic care symbols are found as *groups* of ink components: a main outline
(tub, triangle, square, iron, circle – with its crossing lines, which are
connected to it) plus the small parts that belong to it (dots, temperature
digits, letters, bars under the shape, a separate steam cross ...).

Nothing here decides *what* a candidate is; the goal is a high-recall list of
plausible symbol regions. Text, logos and noise are rejected later by the
classifier (``not_symbol`` class) and by the post-checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np



@dataclass
class Candidate:
    box: tuple[int, int, int, int]                 # x, y, w, h (in the searched image)
    main: int                                      # label of the main component
    parts: list[int] = field(default_factory=list)  # absorbed component labels
    main_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    split: bool = False                            # piece of a split merged component (1.3.0)

    @property
    def h(self):
        return self.box[3]


def binarize(gray: np.ndarray, floor: float = 0.10) -> np.ndarray:
    """Ink mask (uint8 0/255) for dark ink on light background.

    Local black-hat ratio: darkness relative to a morphological closing with a
    kernel a few times wider than a stroke. The background estimate follows
    illumination gradients *and* sharp shadow edges, so shadows do not turn
    into ink. Filled interiors wider than the kernel only keep their outline –
    enough for localisation (classification works on the gray crop).
    """
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    k = int(np.clip(max(H, W) / 45, 9, 45)) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    rel = (bg - g) / np.maximum(bg, 8.0)
    cand = rel[rel > 0.04]
    if cand.size < 50:
        return np.zeros((H, W), np.uint8)
    r8 = np.clip(cand * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t / 255.0 * 0.75, floor, 0.30))
    m = (rel > thr).astype(np.uint8) * 255
    # remove speckle (fabric texture) without eating dots; close hairline gaps
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m


def enhance_for_detection(gray: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Texture-suppressed, contrast-equalised copy of a gray image (detection only).

    Woven labels and faded prints break thin outlines into speckle in the normal
    mask. A light Gaussian (sigma ~ 1 px at the working scale, too small to erase
    an ironing dot) removes the weave, CLAHE lifts faint ink locally. The
    classifier still sees crops of the ORIGINAL image.
    """
    H, W = gray.shape
    sigma = float(np.clip(max(H, W) / 1600.0, 0.8, 1.6)) * strength
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def _remove_long_lines(sub: np.ndarray, h_ref: int, w_ref: int) -> np.ndarray:
    """Erase ruled lines (table grids, label frames) from a component mask.

    Horizontal: segments found by an opening with a moderate kernel on a
    vertically dilated mask (tolerates tilt and a bent label) that chain into a
    run longer than 1.5x the strip height - no single symbol has such a line
    (bars under a tub are at most one symbol wide).
    Vertical: thin lines spanning (almost) the whole remaining strip height are
    cell separators; the sides of a drying square are shorter than the cell.
    """
    out = sub.copy()
    L = max(15, int(0.6 * h_ref))
    hor = cv2.morphologyEx(cv2.dilate(sub, np.ones((7, 1), np.uint8)), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hor, connectivity=8)
    ruled = False
    for j in range(1, n):
        # a table rule / frame spans (almost) the whole strip of at least ~3 cells; the joined
        # base lines of a few touching symbols (tub bars + triangle base + square bottom) do not
        if st[j, 2] >= max(1.5 * h_ref, 0.8 * w_ref) and w_ref >= 2.4 * h_ref:
            out[cv2.dilate((lab == j).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
            ruled = True
    if not ruled:
        return out  # no table: never remove vertical strokes (they are sides of drying squares, irons ...)
    ys = np.where(out.any(1))[0]
    h2 = int(ys.max() - ys.min() + 1) if ys.size else h_ref
    kv = max(15, int(0.85 * h2))
    if out.shape[0] >= kv:
        ver = cv2.morphologyEx(cv2.dilate(out, np.ones((1, 5), np.uint8)), cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(ver, connectivity=8)
        for j in range(1, n):
            if st[j, 2] <= max(8, 0.16 * h2) and st[j, 3] >= kv:
                out[cv2.dilate((lab == j).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
    return out


def _cut_valleys(sub: np.ndarray, h_sym: float) -> np.ndarray:
    """Cut a horizontal strip of touching symbols at the ink valleys between them.

    Dynamic programming over cut positions: every piece must have a plausible
    symbol width (0.45 - 1.6 x strip height); the cost of a cut is the ink in
    that column (relative to the strip's median column), plus a penalty for
    pieces whose width differs from the height (care symbols are roughly
    square). Columns with a lot of ink cannot be cut. Only the 2-px cut column
    is erased; classification still uses the gray crop.
    """
    H, W = sub.shape
    if W < 1.6 * h_sym:
        return sub
    prof = (sub > 0).sum(0).astype(np.float32)
    ps = np.convolve(prof, np.ones(3, np.float32) / 3, mode="same")
    nz = ps[ps > 0]
    if nz.size == 0:
        return sub
    med = float(np.median(nz))
    cost_cut = ps / max(med, 1e-6)
    allowed = cost_cut <= 0.8
    lo, hi = max(2, int(0.45 * h_sym)), int(1.6 * h_sym)
    INF = 1e18
    best = np.full(W + 1, INF)
    prev = np.full(W + 1, -1, np.int64)
    best[0] = 0.0
    for x in range(lo, W + 1):
        if x < W and not allowed[x]:
            continue
        a0, a1 = max(0, x - hi), x - lo
        if a1 < 0:
            continue
        cand = best[a0:a1 + 1]
        widths = x - np.arange(a0, a1 + 1)
        tot = cand + (widths / h_sym - 1.0) ** 2 + (cost_cut[x] if x < W else 0.0)
        k = int(np.argmin(tot))
        if tot[k] < best[x]:
            best[x], prev[x] = tot[k], a0 + k
    if best[W] >= INF:
        return sub
    out = sub.copy()
    x = int(prev[W])
    while x > 0:
        out[:, max(0, x - 1):x + 1] = 0
        x = int(prev[x])
    return out


def find_candidates(gray: np.ndarray, mask: np.ndarray | None = None, max_candidates: int = 160,
                    split_merged: bool = True):
    """Return (candidates, labels, stats, mask).

    A second, adaptive-threshold pass is used as a *recovery detector* when the
    main contrast mask breaks thin, low-contrast outlines into tiny fragments.
    Recovery candidates are only added in a symbol row where a plausible gap is
    bracketed by already-detected symbols.  This keeps label text and fabric
    weave from exploding the candidate list.
    """
    use_recovery = mask is None
    if mask is None:
        mask = binarize(gray)
    H, W = mask.shape
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # below ~16 px the distinguishing details (dots, bars, digits) are not resolvable
    min_h = max(16, int(0.012 * max(H, W)))
    prim = []
    merged = []  # components that may be several touching symbols / symbols inside a ruled grid
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if max(h, w) < min_h or min(h, w) < 0.5 * min_h:
            continue
        if h > 0.97 * H and w > 0.97 * W:
            continue
        ar = w / max(h, 1)
        if ar > 1.65 or ar < 1 / 1.65:
            touches_b = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
            if not touches_b and min(h, w) >= min_h and max(ar, 1 / ar) <= 16:
                merged.append(i)
        if ar < 0.3 or ar > 3.2:
            continue
        fill = a / float(w * h)
        if fill > 0.85 and min(w, h) > 3 * min_h:  # big solid blob: shadow, stain, logo
            continue
        # anything touching the image border is a label edge, seam, garment region or a
        # symbol cut by the frame – none of them can be recognised reliably
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if touches:
            continue
        if h > 0.9 * H and w > 0.9 * W:
            continue
        prim.append(i)
    prim.sort(key=lambda i: -stats[i, 3] * stats[i, 2])
    used = np.zeros(n, bool)
    used[0] = True
    cands: list[Candidate] = []
    # component geometry as arrays: the grouping test below is evaluated for all
    # components at once (same rules as before, vectorised - texture-heavy photos
    # have tens of thousands of components)
    SX, SY, SW, SH = (stats[:, k].astype(np.float64) for k in range(4))
    CX, CY = SX + SW / 2, SY + SH / 2
    for i in prim:
        if used[i]:
            continue
        x, y, w, h, a = stats[i]
        # absorb region: parts inside the shape, bars/steam-cross below it
        ex0, ex1 = x - 0.10 * w, x + w + 0.10 * w
        ey0, ey1 = y - 0.12 * h, y + h + 0.48 * h
        free = ~used
        free[i] = False
        free &= ~((SH >= 0.75 * h) & (SW >= 0.75 * w))  # a comparable object -> a separate symbol
        inside = ((x <= CX) & (CX <= x + w) & (y <= CY) & (CY <= y + h) & (SX >= x - 0.05 * w)
                  & (SX + SW <= x + w + 0.05 * w) & (SY >= y - 0.05 * h) & (SY + SH <= y + h + 0.05 * h))
        below = ((x - 0.4 * w <= SX) & (SX + SW <= x + w + 0.4 * w) & (y + 0.85 * h <= CY) & (CY <= ey1)
                 & (((SW >= 0.45 * w) & (SH <= 0.4 * h) & (SW >= 2.5 * SH))
                    | ((SH <= 0.45 * h) & (SW <= 0.9 * w) & (SY < y + h + 0.2 * h))))
        dist = np.maximum.reduce([x - (SX + SW), SX - (x + w), y - (SY + SH), SY - (y + h), np.zeros_like(SX)])
        near = ((ex0 <= CX) & (CX <= ex1) & (ey0 <= CY) & (CY <= ey1) & (SH < 0.5 * h) & (SW < 0.5 * w)
                & (dist <= 0.12 * max(w, h)))
        parts = [int(j) for j in np.flatnonzero(free & (inside | below | near))]
        if len(parts) > 14:
            if split_merged and i not in merged and min(w, h) >= min_h:
                merged.append(i)  # a ruled frame around symbols: the frame lines get removed below
            continue  # a frame/region enclosing lots of things (label border, text box) – not a symbol
        bx0, by0, bx1, by1 = x, y, x + w, y + h
        for j in parts:
            jx, jy, jw, jh, _ = stats[j]
            bx0, by0 = min(bx0, jx), min(by0, jy)
            bx1, by1 = max(bx1, jx + jw), max(by1, jy + jh)
        cands.append(Candidate((int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)), i, parts, (int(x), int(y), int(w), int(h))))
        used[i] = True
        used[parts] = True
        if len(cands) >= max_candidates:
            break

    if use_recovery and cands and len(cands) < max_candidates:
        # The contrast-relative mask is excellent on most symbols but can erase
        # a thin iron / drying outline printed on textured woven fabric. Adaptive
        # thresholding often recovers the whole contour in those cases.
        h_img, w_img = gray.shape
        block = int(np.clip(round(max(h_img, w_img) / 40), 15, 41)) | 1
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Match the likely foreground polarity: background should be the larger
        # class. This also handles dark/woven labels during the inverted search.
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_ink = float((gray > otsu).mean()) >= 0.5
        threshold_type = cv2.THRESH_BINARY_INV if dark_ink else cv2.THRESH_BINARY
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block, 9
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        recovered, _, _, _ = find_candidates(gray, mask=adaptive, max_candidates=max_candidates)

        def overlap(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        base_centres = []
        for c in cands:
            x, y, w, h = c.box
            base_centres.append((x + w / 2, y + h / 2, w, h, c.box))

        added = []
        for alt in recovered:
            # Use the main connected component box, not the often larger region
            # collected from nearby textile texture and letters below the symbol.
            x, y, w, h = alt.main_box
            if min(w, h) < max(16, int(0.012 * max(h_img, w_img))):
                continue
            box = (int(x), int(y), int(w), int(h))
            if any(overlap(box, c.box) >= 0.18 for c in cands):
                continue
            cx, cy = x + w / 2, y + h / 2
            # Keep only recoveries that line up with the established symbol row
            # and fill a realistic gap between a left and right symbol.
            left = []
            right = []
            aligned = []
            for nx, ny, nw, nh, _ in base_centres:
                scale_h = max(h, nh)
                if abs(cy - ny) > 0.62 * scale_h:
                    continue
                aligned.append((nx, ny, nw, nh))
                dx = cx - nx
                if 0.45 * scale_h <= dx <= 2.4 * scale_h:
                    left.append((dx, scale_h))
                elif -2.4 * scale_h <= dx <= -0.45 * scale_h:
                    right.append((-dx, scale_h))
            if not left or not right or len(aligned) < 2:
                continue
            # Reject tiny text fragments in the row gutter: a recovered symbol
            # should be comparable in size to the symbols already on that row.
            row_heights = np.asarray([item[3] for item in aligned], dtype=np.float32)
            typical_h = float(np.median(row_heights))
            if h < 0.55 * typical_h or h > 1.65 * typical_h:
                continue
            # A candidate must truly occupy a horizontal gap, not overlap the
            # centre of an existing symbol region on the same row. Ignore text
            # candidates at the same x-coordinate on unrelated lines.
            if any(abs(cx - nx) < 0.48 * max(h, nh) for nx, ny, nw, nh in aligned):
                continue
            added.append(Candidate(box=box, main=alt.main, parts=[], main_box=box))
            base_centres.append((cx, cy, w, h, box))
            if len(cands) + len(added) >= max_candidates:
                break
        cands.extend(added)

    # (after the recovery pass, so that split pieces never block a recovered symbol)
    if split_merged and use_recovery and merged:
        # Touching symbols (a crossing line running into the next symbol) and symbols printed
        # in a ruled table form ONE wide component that is rejected above (aspect ratio) or
        # read as one symbol. Remove long ruled lines and cut the strip at ink valleys, then
        # group the pieces again. The pieces are added as additional candidates; duplicate
        # suppression after classification keeps whichever reading is more confident.
        mask2 = np.zeros_like(mask)
        for i in merged:
            x, y, w, h, _ = stats[i]
            sub = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            horizontal = w >= h
            if not horizontal:
                sub = np.ascontiguousarray(sub.T)
            hh = sub.shape[0]
            sub = _remove_long_lines(sub, hh, sub.shape[1])
            ys = np.where(sub.any(1))[0]
            if ys.size:
                hh = int(ys.max() - ys.min() + 1)
            sub = _cut_valleys(sub, max(hh, min_h))
            if not horizontal:
                sub = np.ascontiguousarray(sub.T)
            mask2[y:y + h, x:x + w] = np.maximum(mask2[y:y + h, x:x + w], sub)
        extra, _, _, _ = find_candidates(gray, mask=mask2, max_candidates=max_candidates, split_merged=False)
        for c in extra:
            x, y, w, h = c.box
            if any(_iou(c.box, o.box) > 0.6 for o in cands):
                continue
            ar = w / max(h, 1)
            if ar > 2.6 or ar < 0.38:
                continue  # still a strip: not resolved into single symbols
            c.split = True
            cands.append(c)
            if len(cands) >= max_candidates + 40:  # split pieces get their own budget
                break

    return cands, labels, stats, mask


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def crop(gray: np.ndarray, box, pad_frac: float = 0.08) -> np.ndarray:
    x, y, w, h = box
    p = int(round(max(w, h) * pad_frac)) + 1
    H, W = gray.shape
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(W, x + w + p), min(H, y + h + p)
    return gray[y0:y1, x0:x1]
