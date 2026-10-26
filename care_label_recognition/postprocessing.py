"""Post-processing: duplicate suppression, consistency checks, reading order,
context, decision (recognized / uncertain / dropped)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T_RECOGNIZED = 0.70      # confidence needed to report a symbol as recognized
T_UNCERTAIN = 0.30       # below this a candidate is dropped silently
T_SYMBOLNESS = 0.60      # P(is a care symbol) needed to even mention a candidate
# Classes that are bare geometric primitives (a plain circle, a plain triangle).
# On their own they are indistinguishable from an "O", a ring, a warning sign or
# a logo; they are only reported as recognised next to other care symbols.
GENERIC_SHAPES = {"pro_dryclean", "bleach_any"}


@dataclass
class Det:
    box: list[float]           # x, y, w, h in the ORIGINAL image
    rbox: tuple                # box in the searched (rectified) image
    cls: str
    category: str
    conf: float
    p_symbol: float
    top: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    crop_idx: int = -1
    split: bool = False        # comes from a split merged component
    obox: tuple | None = None  # box in the reading frame (the rotation in which the label was read)


def _ob(d: "Det"):
    return d.obox if d.obox is not None else d.rbox


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0, 0.0
    union = aw * ah + bw * bh - inter
    return inter / union, inter / max(1e-6, min(aw * ah, bw * bh))


def drop_merged(dets: list[Det]) -> list[Det]:
    """A wide box that holds two or more confidently recognised symbols side by side
    (or stacked) is a merge of touching symbols, not a symbol: drop it in favour of the
    pieces, even when the merged crop happens to look like a single class (1.3.0)."""
    out = []
    for d in dets:
        x, y, w, h = d.rbox
        ar = w / max(h, 1)
        drop = False
        if ar > 1.6 or ar < 1 / 1.6:
            horiz = ar > 1.6
            inner = []
            for o in dets:
                if o is d or o.conf < T_UNCERTAIN or o.p_symbol < T_SYMBOLNESS:
                    continue
                ox, oy, ow, oh = o.rbox
                ix = max(0.0, min(x + w, ox + ow) - max(x, ox))
                iy = max(0.0, min(y + h, oy + oh) - max(y, oy))
                if ix * iy < 0.8 * ow * oh:
                    continue
                if (horiz and oh >= 0.55 * h) or (not horiz and ow >= 0.55 * w):
                    inner.append(o)
            # the pieces must contradict the merged reading: at least one piece is confidently a
            # symbol of ANOTHER category than the merged box (pieces of one wide symbol - e.g. the
            # two halves of a "40°C" tub read as "30" and "iron" - share the parent's category
            # in their confident part and do not count)
            foreign = [o for o in inner if o.conf >= T_RECOGNIZED and o.category != d.category]
            if len(inner) >= 2:
                span = (max(o.rbox[0] + o.rbox[2] for o in inner) - min(o.rbox[0] for o in inner)) if horiz else \
                       (max(o.rbox[1] + o.rbox[3] for o in inner) - min(o.rbox[1] for o in inner))
                drop = bool(foreign) and span >= 0.7 * (w if horiz else h)
        if drop:
            continue
        out.append(d)
    return out


def suppress_duplicates(dets: list[Det]) -> list[Det]:
    """Two candidates on one physical symbol -> keep the more reliable one."""
    dets = drop_merged(dets)
    # a piece of a split component is only an answer where the merged parent was rejected:
    # inside a surviving whole-symbol candidate it is a fragment of that symbol
    whole = [d for d in dets if not d.split and d.p_symbol >= T_SYMBOLNESS and d.conf >= T_UNCERTAIN]
    dets = [d for d in dets if not (d.split and any(iou(d.rbox, o.rbox)[1] > 0.6 and
                                                    o.rbox[2] * o.rbox[3] > 1.3 * d.rbox[2] * d.rbox[3] for o in whole))]
    dets = sorted(dets, key=lambda d: -(d.conf * (1.0 if d.p_symbol >= T_SYMBOLNESS else 0.5)))
    keep: list[Det] = []
    for d in dets:
        dup = False
        for k in keep:
            u, cont = iou(d.rbox, k.rbox)
            if u > 0.3 or cont > 0.6:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def size_consistency(dets: list[Det]) -> list[Det]:
    """Symbols on one label have similar sizes. With >= 2 confident symbols,
    much smaller uncertain blobs are dropped and size outliers lose confidence."""
    good = [d for d in dets if d.conf >= T_RECOGNIZED]
    if len(good) < 2:
        return dets
    med = float(np.median([max(d.rbox[2], d.rbox[3]) for d in good]))
    out = []
    for d in dets:
        s = max(d.rbox[2], d.rbox[3])
        if d.conf < T_RECOGNIZED and s < 0.4 * med:
            continue  # texture speck / text glyph next to real symbols
        if d.conf < T_RECOGNIZED and not _has_row_neighbour(d, good) and (d.split or len(good) >= 3):
            # unsure and outside every row/column of the (>= 3) confidently read symbols: a letter
            # of the brand / composition text, a logo detail or a piece of a split texture blob
            continue
        if s < 0.45 * med or s > 2.3 * med:
            d.conf *= 0.8
            d.flags.append("size_outlier")
        if s < 0.6 * med and len(good) >= 3 and not _has_row_neighbour(d, good):
            # much smaller than the label's symbols and not part of any row/column of
            # similar symbols: typically a letter of rotated text, a ®, a logo detail
            d.conf *= 0.6
            d.flags.append("isolated_small")
        out.append(d)
    return out


def _has_row_neighbour(d: Det, others: list[Det]) -> bool:
    x, y, w, h = d.rbox
    s = max(w, h)
    for o in others:
        if o is d:
            continue
        ox, oy, ow, oh = o.rbox
        so = max(ow, oh)
        if not (0.7 * s <= so <= 1.4 * s):
            continue
        hov = min(y + h, oy + oh) - max(y, oy)
        vov = min(x + w, ox + ow) - max(x, ox)
        hgap = max(ox - (x + w), x - (ox + ow))
        vgap = max(oy - (y + h), y - (oy + oh))
        if (hov >= 0.5 * min(h, oh) and hgap <= 1.2 * s) or (vov >= 0.5 * min(w, ow) and vgap <= 1.2 * s):
            return True
    return False


def reading_order(dets: list[Det]) -> tuple[list[Det], str]:
    """Rows top->bottom, left->right inside a row; a single column top->bottom."""
    if len(dets) <= 1:
        return dets, "horizontal"
    cx = np.array([_ob(d)[0] + _ob(d)[2] / 2 for d in dets])
    cy = np.array([_ob(d)[1] + _ob(d)[3] / 2 for d in dets])
    hmed = float(np.median([_ob(d)[3] for d in dets]))
    vertical = (np.ptp(cy) > 1.5 * np.ptp(cx)) and np.ptp(cx) < hmed
    if vertical:
        return [dets[i] for i in np.argsort(cy)], "vertical"
    order = np.argsort(cy)
    rows, cur = [], [order[0]]
    for i in order[1:]:
        if abs(cy[i] - np.mean(cy[cur])) <= 0.6 * hmed:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)
    out = []
    for r in rows:
        out += [dets[i] for i in sorted(r, key=lambda i: cx[i])]
    return out, "horizontal"


def row_index(dets: list[Det]) -> list[int]:
    """1-based row of every (already reading-ordered) detection: a new row starts
    when the next symbol is clearly below the current row."""
    rows, r, cur = [], 1, []
    for d in dets:
        cy, h = _ob(d)[1] + _ob(d)[3] / 2, _ob(d)[3]
        if cur and abs(cy - np.mean(cur)) > 0.6 * max(h, 1):
            r += 1
            cur = []
        cur.append(cy)
        rows.append(r)
    return rows


def grid_rows(dets: list[Det]):
    """Rows of a possibly tilted symbol grid: the rotation that yields the fewest rows.
    Returns list of rows, each a list of detections ordered along the row."""
    if not dets:
        return []
    hmed = float(np.median([_ob(d)[3] for d in dets]))
    C = np.array([[_ob(d)[0] + _ob(d)[2] / 2, _ob(d)[1] + _ob(d)[3] / 2] for d in dets])
    best = None
    for ang in np.radians(np.arange(-25, 25.5, 1.0)):
        y = -C[:, 0] * np.sin(ang) + C[:, 1] * np.cos(ang)
        x = C[:, 0] * np.cos(ang) + C[:, 1] * np.sin(ang)
        order = np.argsort(y)
        rows, cur = [], [order[0]]
        for i in order[1:]:
            if abs(y[i] - np.mean(y[cur])) <= 0.6 * hmed:
                cur.append(i)
            else:
                rows.append(cur)
                cur = [i]
        rows.append(cur)
        if best is None or len(rows) < len(best[0]):
            best = (rows, x)
    rows, x = best
    return [[dets[i] for i in sorted(r, key=lambda i: x[i])] for r in rows]


def grid_variants(dets: list[Det], min_shared: int = 3) -> dict[int, Det]:
    """Multi-country care grids repeat the same row of instructions; some countries draw a symbol
    differently (crossed filled triangle, crossed empty square ...). For a row that repeats an
    earlier row (>= ``min_shared`` identical confident ids), a symbol of the SAME category in the
    SAME column as a symbol of the earlier row is that row's variant of the same instruction.
    Returns {id(det): reference det}."""
    conf = [d for d in dets if d.conf >= T_RECOGNIZED and d.p_symbol >= T_SYMBOLNESS]
    rows = grid_rows(conf)
    out: dict[int, Det] = {}
    for k in range(1, len(rows)):
        ids_k = {d.cls for d in rows[k]}
        for q in range(k):
            if len(ids_k & {d.cls for d in rows[q]}) < min_shared:
                continue
            ref = rows[q]
            for j, d in enumerate(rows[k]):
                if d.cls in {r.cls for r in ref}:
                    continue
                cand = ref[j] if len(ref) == len(rows[k]) else None
                if cand is None:  # unequal rows: nearest symbol along the row
                    cx = _ob(d)[0] + _ob(d)[2] / 2
                    cand = min(ref, key=lambda r: abs(_ob(r)[0] + _ob(r)[2] / 2 - cx))
                    if abs(_ob(cand)[0] + _ob(cand)[2] / 2 - cx) > 0.8 * _ob(d)[3]:
                        cand = None
                if cand is not None and cand.category == d.category:
                    out[id(d)] = cand
            break
    return out


def context_bonus(dets: list[Det], category_order: list[str]):
    """Canonical order wash→bleach→dry→iron→professional is weak supporting
    evidence. It never changes a class; it only adds a small bonus to symbols
    that are already confident."""
    conf = [d for d in dets if d.conf >= T_RECOGNIZED]
    if len(conf) < 3:
        return
    idx = [category_order.index(d.category) for d in conf]
    if all(a <= b for a, b in zip(idx, idx[1:])):
        for d in conf:
            d.conf = min(0.99, d.conf * 1.03)
            d.flags.append("canonical_order")


def text_context(dets: list[Det]) -> None:
    """Letters look like symbols in isolation (O ≈ plain dryclean circle, E ≈ flat
    drying ...). A candidate whose immediate neighbours of the same size along a
    text line are rejected as text is most likely a letter inside a word: its
    confidence is halved. Text lines are checked horizontally and - for labels
    photographed sideways (1.3.0) - vertically. Real symbol rows are not affected
    because their neighbours are symbols."""
    for d in dets:
        x, y, w, h = d.rbox
        close_text = 0
        above = below = 0
        for o in dets:
            if o is d or o.p_symbol >= 0.3:
                continue
            ox, oy, ow, oh = o.rbox
            _, cont = iou(d.rbox, o.rbox)
            if cont > 0.25:
                continue  # an overlapping fragment of the same symbol (e.g. a split piece), not a neighbour
            # horizontal text line
            if 0.7 * h <= oh <= 1.4 * h:
                ov = min(y + h, oy + oh) - max(y, oy)
                gap = max(ox - (x + w), x - (ox + ow))
                if ov >= 0.6 * min(h, oh) and gap <= 0.45 * h:
                    close_text += 1
                    continue
            # vertical text line (label rotated by 90 degrees): text on BOTH sides along the
            # column - a symbol row with a text line under it must not be affected
            if 0.7 * w <= ow <= 1.4 * w:
                ov = min(x + w, ox + ow) - max(x, ox)
                if ov >= 0.6 * min(w, ow):
                    if 0 <= oy - (y + h) <= 0.45 * w:
                        below += 1
                    elif 0 <= y - (oy + oh) <= 0.45 * w:
                        above += 1
        if above and below:
            close_text += 1
        if close_text >= 1:
            d.conf *= 0.5
            d.flags.append("text_context")


def require_context_for_generic(dets: list[Det]) -> None:
    specific = [d for d in dets if d.conf >= T_RECOGNIZED and d.cls not in GENERIC_SHAPES]
    if specific:
        return
    for d in dets:
        if d.cls in GENERIC_SHAPES and d.conf >= T_RECOGNIZED:
            d.conf = T_RECOGNIZED - 0.05
            d.flags.append("generic_shape_without_context")
