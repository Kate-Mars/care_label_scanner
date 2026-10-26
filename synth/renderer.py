"""Procedural renderer for care symbols.

``render_symbol(spec, rng)`` draws one symbol described by a taxonomy ``render``
spec (see training/build_taxonomy.py) and returns an ink mask (float32, 1 = ink).
Geometry, stroke width, fonts, dot layout, bar spacing etc. are randomised so
that the classifier learns the *structure* of a symbol rather than one drawing.

The semantic attributes (temperature, number of dots/bars, crossing, letter)
are never randomised – augmentation must not change the meaning of a symbol.

``render_negative(rng)`` draws things that are *not* care symbols but are found
on labels: words, numbers, sizes, ®/© style logos, warning triangles, ...
"""
from __future__ import annotations

import math
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import fonts as F

LO, HI = -0.25, 1.25  # unit-coordinate range of the canvas


class Pen:
    def __init__(self, size: int, stroke: float, rng: np.random.Generator):
        self.S = size
        self.img = Image.new("L", (size, size), 0)
        self.d = ImageDraw.Draw(self.img)
        self.stroke = stroke
        self.rng = rng

    def p(self, x, y):
        s = self.S / (HI - LO)
        return ((x - LO) * s, (y - LO) * s)

    def w(self, k=1.0):
        return max(1, int(round(self.stroke * k * self.S / (HI - LO))))

    def line(self, pts, k=1.0):
        pts = [self.p(*q) for q in pts]
        self.d.line(pts, fill=255, width=self.w(k), joint="curve")
        r = self.w(k) / 2.0
        for x, y in (pts[0], pts[-1]):  # round caps
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def ellipse(self, cx, cy, rx, ry, fill=False, k=1.0):
        x0, y0 = self.p(cx - rx, cy - ry)
        x1, y1 = self.p(cx + rx, cy + ry)
        if fill:
            self.d.ellipse([x0, y0, x1, y1], fill=255)
        else:
            self.d.ellipse([x0, y0, x1, y1], outline=255, width=self.w(k))

    def dot(self, cx, cy, r):
        self.ellipse(cx, cy, r, r, fill=True)

    def polygon(self, pts):
        self.d.polygon([self.p(*q) for q in pts], fill=255)

    def text(self, s, cx, cy, h, font_path, wscale=1.0):
        """Draw text with its ink box centred at (cx, cy) and ink height h (unit coords)."""
        px_h = max(6, int(h * self.S / (HI - LO)))
        font = ImageFont.truetype(font_path, size=int(px_h * 1.35))
        tmp = Image.new("L", (px_h * 3 * max(1, len(s)) + 20, px_h * 3), 0)
        ImageDraw.Draw(tmp).text((10, 10), s, fill=255, font=font)
        a = np.asarray(tmp)
        ys, xs = np.where(a > 0)
        if len(ys) == 0:
            return
        g = tmp.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        scale = px_h / g.height
        g = g.resize((max(1, int(g.width * scale * wscale)), px_h), Image.LANCZOS)
        X, Y = self.p(cx, cy)
        self.img.paste(255, (int(X - g.width / 2), int(Y - g.height / 2)), g)

    def mask(self):
        return np.asarray(self.img, dtype=np.float32) / 255.0


def _bez(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return [tuple(q) for q in pts]


def _j(rng, a, s):
    return a + rng.uniform(-s, s)


# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------

def _bars(pen, rng, n, x0, x1, y, gap):
    for i in range(n):
        yy = y + i * gap
        pen.line([(x0, yy), (x1, yy)], k=rng.uniform(0.8, 1.1))


def _cross(pen, rng, box):
    x0, y0, x1, y1 = box
    o = rng.uniform(-0.03, 0.05)
    k = rng.uniform(0.8, 1.2)
    pen.line([(x0 - o, y0 - o), (x1 + o, y1 + o)], k)
    pen.line([(x1 + o, y0 - o), (x0 - o, y1 + o)], k)


def _dots(pen, rng, n, cx, cy, r, sp, layout=None):
    if n <= 0:
        return
    rows = {1: [1], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2], 6: [3, 3]}[n]
    if n == 3 and layout == "tri":
        rows = [1, 2]
    if n == 4 and layout == "row":
        rows = [4]
    tot_h = (len(rows) - 1) * sp
    for ri, k in enumerate(rows):
        yy = cy - tot_h / 2 + ri * sp
        for i in range(k):
            xx = cx + (i - (k - 1) / 2) * sp
            pen.dot(xx + rng.normal(0, r * 0.08), yy + rng.normal(0, r * 0.08), r)


def _tub(pen, rng, spec, font_pool):
    yt = _j(rng, 0.20, 0.04)
    yb = _j(rng, 0.70, 0.04)
    xl0, xr0 = _j(rng, 0.08, 0.03), _j(rng, 0.92, 0.03)
    inset = rng.uniform(0.07, 0.15)
    xl1, xr1 = xl0 + inset, xr0 - inset
    over = rng.uniform(0.0, 0.07)
    yw = yt + over
    pen.line([(xl0, yt), (xl1, yb), (xr1, yb), (xr0, yt)])
    # wavy water line between the walls
    lx = xl0 + (xl1 - xl0) * (yw - yt) / (yb - yt)
    rx = xr0 - (xr0 - xr1) * (yw - yt) / (yb - yt)
    periods = rng.choice([2.5, 3.0, 3.5, 4.0])
    amp = rng.uniform(0.022, 0.045)
    phase = rng.choice([0.0, math.pi])
    xs = np.linspace(lx, rx, 60)
    ys = yw + amp * np.sin(phase + 2 * math.pi * periods * (xs - lx) / (rx - lx))
    pen.line(list(zip(xs, ys)), k=rng.uniform(0.8, 1.0))
    inner_top, inner_bot = yw + amp + 0.04, yb - 0.04
    cx = (xl1 + xr1) / 2
    if spec.get("temp"):
        if spec.get("temp_style") == "dots":
            n = {30: 1, 40: 2, 50: 3, 60: 4, 70: 5, 95: 6}[spec["temp"]]
            r = rng.uniform(0.028, 0.04)
            sp = r * rng.uniform(2.6, 3.6)
            rows = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}[n]
            cy = inner_bot - r - (rows - 1) * sp / 2 - rng.uniform(0.0, 0.08)
            _dots(pen, rng, n, cx, cy, r, sp)
        else:
            if spec.get("hand_temp"):
                h = rng.uniform(0.13, 0.19)
                cy = inner_bot - h * 0.58 + rng.uniform(-0.012, 0.012)
            else:
                h = rng.uniform(0.22, 0.32)
                cy = (inner_top + inner_bot) / 2 + rng.uniform(-0.02, 0.03)
            h = min(h, inner_bot - inner_top)
            # Real labels print the limit as "30", "30°", "30°C", "30C" or "30 °C" (1.3.0: the
            # renderer used to draw the bare number only). The number itself is never changed.
            t = str(spec["temp"])
            r = rng.random()
            if r < 0.22:
                t, k = t + "°C", 0.74
            elif r < 0.34:
                t, k = t + "°", 0.88
            elif r < 0.42:
                t, k = t + "C", 0.8
            else:
                k = 1.0
            pen.text(t, cx, cy, h * k, rng.choice(font_pool), wscale=rng.uniform(0.85, 1.2) if k == 1.0 else rng.uniform(0.75, 1.0))
    if spec.get("bars"):
        gap = rng.uniform(0.07, 0.10)
        _bars(pen, rng, spec["bars"], xl0 + 0.02, xr0 - 0.02, yb + gap, gap)
    if spec.get("cross"):
        _cross(pen, rng, (xl0, yt - 0.05, xr0, yb + 0.05))
    return (xl0, yt, xr0, yb)


def _hand(pen, rng, temperature_below: bool = False):
    """Draw a hand over the tub; lift/shorten it when a numeral is below it.

    Some care labels print a maximum temperature (e.g. 30 °C) inside the
    hand-wash tub. The older renderer drew the hand over the numeral, often
    obscuring it in synthetic training crops and teaching the model to ignore
    the temperature.
    """
    cx = _j(rng, 0.52, 0.04)
    cy = _j(rng, 0.21 if temperature_below else 0.30, 0.025 if temperature_below else 0.04)
    style = rng.random()
    if style < 0.25:
        # 1.3.1: outline hand reaching in DIAGONALLY from the upper right (common on Italian /
        # European woven labels): two arm lines crossing the rim, curled finger hooks in the water
        ex, ey = cx + rng.uniform(0.26, 0.34), cy - rng.uniform(0.18, 0.26)     # arm end, outside the tub
        hx, hy = cx - rng.uniform(0.0, 0.06), cy + rng.uniform(0.04, 0.08)      # back of the hand
        w = rng.uniform(0.07, 0.10)
        dx, dy = hx - ex, hy - ey
        L = math.hypot(dx, dy)
        px, py = -dy / L, dx / L                      # perpendicular to the arm
        for o in (-w, w):
            pen.line([(ex + o * px, ey + o * py), (hx + o * px, hy + o * py)], k=rng.uniform(0.8, 1.0))
        n = int(rng.integers(3, 5))
        fy = hy + (0.06 if temperature_below else rng.uniform(0.10, 0.16))
        sp = rng.uniform(0.085, 0.11)
        for i in range(n):
            fx = hx - 0.22 + i * sp
            r = rng.uniform(0.035, 0.055)
            t = np.linspace(0.0, math.pi * rng.uniform(1.0, 1.4), 12)
            pts = [(fx + r * math.cos(a), fy + r * math.sin(a) * 1.3) for a in t]
            pen.line([(fx + r, fy - rng.uniform(0.06, 0.1))] + pts, k=rng.uniform(0.8, 1.0))
        return
    if style < 0.4:
        # 1.3.0: "fingers only" style (several printed labels and the Kobo icon set): 3-5
        # strokes dipping into the water, no palm
        n = int(rng.integers(3, 6))
        top = cy - rng.uniform(0.10, 0.16)
        for i in range(n):
            fx = cx - 0.09 + i * (0.18 / max(1, n - 1))
            end = cy + (rng.uniform(0.08, 0.14) if temperature_below else rng.uniform(0.14, 0.24))
            pen.line([(fx, top + rng.uniform(0, 0.04)), (fx + rng.uniform(-0.015, 0.015), end)], k=rng.uniform(0.8, 1.2))
        return
    if style < 0.5:
        # outline hand entering from the upper right: wrist, palm outline and finger strokes
        pen.line([(cx + 0.20, cy - 0.20), (cx + 0.10, cy - 0.04), (cx - 0.10, cy - 0.02), (cx - 0.14, cy + 0.04)], k=0.9)
        pen.line([(cx + 0.28, cy - 0.12), (cx + 0.14, cy + 0.05)], k=0.9)
        for i in range(4):
            fx = cx - 0.12 + i * 0.07
            end = cy + (0.12 if temperature_below else rng.uniform(0.18, 0.25))
            pen.line([(fx, cy + 0.0), (fx + rng.uniform(-0.01, 0.01), end)], k=rng.uniform(0.8, 1.1))
        return
    filled = rng.random() < 0.5
    # palm
    pen.ellipse(cx, cy, 0.13, 0.11, fill=filled)
    # fingers dipping into the water
    for i in range(4):
        fx = cx - 0.09 + i * 0.06
        finger_end = rng.uniform(0.15, 0.22) if temperature_below else rng.uniform(0.22, 0.30)
        pen.line([(fx, cy + 0.05), (fx + rng.uniform(-0.01, 0.01), cy + finger_end)], k=rng.uniform(0.9, 1.4))
    # thumb
    pen.line([(cx - 0.11, cy - 0.02), (cx - 0.22, cy + 0.10)], k=1.2)


def _twist(pen, rng):
    a, b = _j(rng, 0.10, 0.03), _j(rng, 0.90, 0.03)
    t, bt = _j(rng, 0.32, 0.03), _j(rng, 0.68, 0.03)
    m = 0.5
    pen.line([(a, t), (a, bt)])
    pen.line([(b, t), (b, bt)])
    pen.line([(a, t), (m - 0.06, m - 0.07)])
    pen.line([(a, bt), (m - 0.06, m + 0.07)])
    pen.line([(b, t), (m + 0.06, m - 0.07)])
    pen.line([(b, bt), (m + 0.06, m + 0.07)])
    pen.ellipse(m, m, rng.uniform(0.08, 0.12), rng.uniform(0.07, 0.1))
    return (a, t, b, bt)


def _triangle(pen, rng, spec, font_pool):
    top = (_j(rng, 0.5, 0.02), _j(rng, 0.10, 0.03))
    bl = (_j(rng, 0.06, 0.03), _j(rng, 0.86, 0.03))
    br = (_j(rng, 0.94, 0.03), bl[1])
    k = 0.7 if spec.get("thin") else 1.0
    pen.line([top, bl, br, top], k=k)
    if spec.get("filled"):  # legacy / JIS-style solid triangle (e.g. crossed black triangle = do not bleach)
        pen.polygon([top, bl, br])
    inner = spec.get("inner")
    if inner == "stripes":
        for f in (rng.uniform(0.28, 0.36), rng.uniform(0.50, 0.60)):
            x0 = bl[0] + (br[0] - bl[0]) * f
            # parallel to the left edge, up to the right edge
            dx, dy = top[0] - bl[0], top[1] - bl[1]
            # intersect with right edge top->br
            t = (1 - f) * rng.uniform(0.85, 1.0)
            pen.line([(x0, bl[1]), (x0 + dx * t, bl[1] + dy * t)], k=0.85)
    elif inner == "CL":
        pen.text("CL", 0.5, 0.62, 0.22, rng.choice(font_pool))
    if spec.get("cross"):
        _cross(pen, rng, (0.08, 0.14, 0.92, 0.92))
    return (bl[0], top[1], br[0], bl[1])


def _square(pen, rng, spec):
    a = _j(rng, 0.10, 0.02)
    b = _j(rng, 0.90, 0.02)
    pen.line([(a, a), (b, a), (b, b), (a, b), (a, a)])
    c = (a + b) / 2
    inner = spec.get("inner")
    if inner == "circle":
        r = (b - a) / 2 * rng.uniform(0.82, 1.0)
        if spec.get("filled"):
            pen.ellipse(c, c, r * rng.uniform(0.85, 1.0), r * rng.uniform(0.85, 1.0), fill=True)
        else:
            pen.ellipse(c, c, r, r)
        n = spec.get("dots", 0)
        if n:
            rr = rng.uniform(0.035, 0.05)
            _dots(pen, rng, n, c, c, rr, rr * rng.uniform(2.6, 3.6), layout=rng.choice(["row", "tri"]) if n == 3 and rng.random() < 0.3 else None)
    elif inner == "sag":
        depth = rng.uniform(0.25, 0.40)
        pen.line(_bez((a, a + 0.01), (c, a + 2 * depth), (b, a + 0.01)))
    elif inner == "vlines":
        n = spec["n"]
        span = rng.uniform(0.22, 0.34) if n > 1 else 0
        y0, y1 = a + rng.uniform(0.12, 0.2), b - rng.uniform(0.12, 0.2)
        if n == 1 and rng.random() < 0.5:
            y0 = a  # ISO line drying: line hanging from the top edge
        for i in range(n):
            x = c + (i - (n - 1) / 2) * (span / max(1, n - 1)) if n > 1 else c
            pen.line([(x, y0), (x, y1)])
    elif inner == "hlines":
        n = spec["n"]
        half = rng.uniform(0.18, 0.32)
        for i in range(n):
            y = c + (i - (n - 1) / 2) * rng.uniform(0.12, 0.16)
            pen.line([(c - half, y), (c + half, y)])
    shade = spec.get("shade", 0)
    for i in range(shade):
        off = rng.uniform(0.25, 0.35) + i * rng.uniform(0.09, 0.12)
        pen.line([(a, a + off), (a + off, a)], k=0.9)
    if spec.get("bars"):
        gap = rng.uniform(0.07, 0.10)
        _bars(pen, rng, spec["bars"], a, b, b + gap, gap)
    if spec.get("cross"):
        _cross(pen, rng, (a, a, b, b))
    return (a, a, b, b)


def _iron(pen, rng, spec):
    x0, x1 = _j(rng, 0.08, 0.03), _j(rng, 0.92, 0.03)
    yb = _j(rng, 0.72, 0.03)
    yt = yb - rng.uniform(0.26, 0.34)
    xn = x0 + rng.uniform(0.30, 0.42)  # where the flat top starts
    curve = _bez((x0, yb), (x0 + rng.uniform(0.04, 0.14), yt + rng.uniform(0.0, 0.06)), (xn, yt))
    pen.line(curve + [(x1, yt), (x1, yb), (x0, yb)])
    top = yt
    if rng.random() < 0.55:  # handle
        yh = yt - rng.uniform(0.10, 0.16)
        hx0 = xn + rng.uniform(-0.04, 0.06)
        pen.line([(hx0, yt), (hx0 - 0.02, yh), (x1 - 0.04, yh), (x1, yt)], k=0.9)
        top = yh
    n = spec.get("dots", 0)
    if n:
        r = rng.uniform(0.03, 0.042)
        cx = (xn + x1) / 2 + rng.uniform(-0.06, 0.02)
        _dots(pen, rng, n, cx, (yt + yb) / 2 + rng.uniform(0.0, 0.05), r, r * rng.uniform(2.6, 3.6))
    if spec.get("steam_cross"):
        y0, y1 = yb + 0.05, yb + rng.uniform(0.22, 0.30)
        mx = (x0 + x1) / 2 + rng.uniform(-0.05, 0.05)
        hw = rng.uniform(0.15, 0.24)
        pen.line([(mx - hw, y0), (mx + hw, y1)])
        pen.line([(mx + hw, y0), (mx - hw, y1)])
        if rng.random() < 0.6:
            for dx in (-0.08, 0.0, 0.08):
                pen.line([(mx + dx, yb), (mx + dx, y0 + 0.03)], k=0.8)
        return (x0, top, x1, y1)
    if spec.get("cross"):
        _cross(pen, rng, (x0, top - 0.03, x1, yb + 0.08))
    return (x0, top, x1, yb)


def _circle(pen, rng, spec, font_pool):
    cx, cy = 0.5, 0.45
    r = _j(rng, 0.38, 0.03)
    ry = r * rng.uniform(0.95, 1.05)
    if spec.get("filled"):
        pen.ellipse(cx, cy, r, ry, fill=True)
    else:
        pen.ellipse(cx, cy, r, ry)
    if spec.get("letter"):
        h = rng.uniform(0.30, 0.42)
        pen.text(spec["letter"], cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02), h,
                 rng.choice(font_pool), wscale=rng.uniform(0.85, 1.15))
    if spec.get("bars"):
        gap = rng.uniform(0.07, 0.10)
        _bars(pen, rng, spec["bars"], cx - r * rng.uniform(1.0, 1.2), cx + r * rng.uniform(1.0, 1.2), cy + ry + gap * 1.2, gap)
    tick = spec.get("tick")
    if tick:
        ang = {"ur": -45, "ul": -135, "ll": 135, "lr": 45}[tick] + rng.uniform(-8, 8)
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        d = r + rng.uniform(0.06, 0.12)
        mx, my = cx + d * dx, cy + d * dy
        L = rng.uniform(0.12, 0.2)
        tx, ty = -dy, dx
        pen.line([(mx - L * tx, my - L * ty), (mx + L * tx, my + L * ty)])
    if spec.get("cross"):
        _cross(pen, rng, (cx - r - 0.02, cy - ry - 0.02, cx + r + 0.02, cy + ry + 0.02))
    return (cx - r, cy - ry, cx + r, cy + ry)


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

def render_symbol(spec: dict, rng: np.random.Generator, size: int = 192, split: str = "train",
                  stroke: float | None = None) -> np.ndarray:
    """Render symbol `spec` -> ink mask float32 (size, size)."""
    if stroke is None:
        stroke = rng.uniform(0.035, 0.085)
    pen = Pen(size, stroke, rng)
    fp = F.pool(split, symbol=True)
    shape = spec["shape"]
    if shape == "tub":
        _tub(pen, rng, spec, fp)
    elif shape == "tub_hand":
        # Preserve temperature / bar attributes when the label combines a hand
        # pictogram with an explicit limit (e.g. hand wash 30 °C).
        tub_spec = {k: val for k, val in spec.items() if k != "shape"}
        if spec.get("temp"):
            tub_spec["hand_temp"] = True
        _tub(pen, rng, {"shape": "tub", **tub_spec}, fp)
        _hand(pen, rng, temperature_below=bool(spec.get("temp")))
    elif shape == "twist":
        box = _twist(pen, rng)
        if spec.get("cross"):
            _cross(pen, rng, (box[0], 0.2, box[2], 0.8))
    elif shape == "triangle":
        _triangle(pen, rng, spec, fp)
    elif shape == "square":
        _square(pen, rng, spec)
    elif shape == "iron":
        _iron(pen, rng, spec)
    elif shape == "circle":
        _circle(pen, rng, spec, fp)
    else:
        raise ValueError(shape)
    return pen.mask()


WORDS = ["100%", "COTTON", "POLYESTER", "WOOL", "SILK", "LINEN", "VISCOSE", "ELASTANE", "NYLON",
         "MADE", "IN", "ITALY", "CHINA", "TURKEY", "SIZE", "XS", "S", "M", "L", "XL", "XXL", "40", "30", "60",
         "95", "42", "38", "RN", "CA", "EUR", "USA", "UK", "WASH", "DRY", "IRON", "COLD", "LOW", "DO", "NOT",
         "BLEACH", "TUMBLE", "WITH", "SIMILAR", "COLOURS", "REVERSE", "SIDE", "ART.", "LOT", "N°", "%", "°C",
         "40°C", "30°C", "A", "P", "F", "W", "CL", "BAUMWOLLE", "COTON", "ХЛОПОК", "РАЗМЕР"]


def _rand_word(rng):
    r = rng.random()
    if r < 0.55:
        return str(rng.choice(WORDS))
    if r < 0.75:
        return "".join(rng.choice(list(string.ascii_uppercase + string.digits), size=rng.integers(1, 6)))
    if r < 0.9:
        return str(rng.integers(0, 100000))
    return "".join(rng.choice(list(string.ascii_lowercase), size=rng.integers(2, 8)))


def render_negative(rng: np.random.Generator, size: int = 192, split: str = "train") -> np.ndarray:
    """Things that appear on labels but are NOT care symbols."""
    pen = Pen(size, rng.uniform(0.03, 0.08), rng)
    fp = F.pool(split)
    kind = rng.choice(["word", "word", "word", "char", "num", "circle_letter", "tri_mark", "square_letter",
                       "logo", "star", "barcode", "lines", "blob", "partial", "dots", "arrow", "text_block"],
                      p=[.12, .08, .08, .1, .1, .1, .04, .04, .07, .03, .03, .04, .03, .06, .02, .02, .04])
    if kind in ("word",):
        w = _rand_word(rng)
        pen.text(w, 0.5, 0.5, rng.uniform(0.2, 0.5), rng.choice(fp), wscale=rng.uniform(0.7, 1.2))
    elif kind == "char":
        pen.text(str(rng.choice(list("ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789%&@#?"))), 0.5, 0.5, rng.uniform(0.5, 0.9), rng.choice(fp))
    elif kind == "num":
        s = str(rng.choice(["30", "40", "50", "60", "70", "95", "30°", "40°C", "60°", "110", "150", "200", "1", "2", "3"]))
        pen.text(s, 0.5, 0.5, rng.uniform(0.3, 0.7), rng.choice(fp), wscale=rng.uniform(0.8, 1.2))
    elif kind == "circle_letter":  # ® © like marks – letters that are NOT A/P/F/W
        pen.ellipse(0.5, 0.5, 0.4, 0.4)
        pen.text(str(rng.choice(list("RCBDEGHKMNOSTUVXYZ0123456789"))), 0.5, 0.5, rng.uniform(0.3, 0.45), rng.choice(fp))
    elif kind == "tri_mark":
        pen.line([(0.5, 0.1), (0.06, 0.88), (0.94, 0.88), (0.5, 0.1)])
        pen.text(str(rng.choice(list("!123456789"))), 0.5, 0.62, 0.3, rng.choice(fp))
    elif kind == "square_letter":
        pen.line([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)])
        pen.text(str(rng.choice(list("ABCDEFGHKLMNRSTXYZ0123456789"))), 0.5, 0.5, 0.45, rng.choice(fp))
    elif kind == "logo":
        for _ in range(rng.integers(2, 5)):
            pen.ellipse(rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35), fill=rng.random() < 0.3)
        if rng.random() < 0.5:
            pen.text(_rand_word(rng)[:4], 0.5, 0.5, 0.2, rng.choice(fp))
    elif kind == "star":
        n = rng.integers(4, 9)
        pts = []
        for i in range(2 * n):
            rr = 0.42 if i % 2 == 0 else rng.uniform(0.12, 0.25)
            a = math.pi * i / n
            pts.append((0.5 + rr * math.cos(a), 0.5 + rr * math.sin(a)))
        if rng.random() < 0.5:
            pen.polygon(pts)
        else:
            pen.line(pts + [pts[0]])
    elif kind == "barcode":
        x = 0.05
        while x < 0.95:
            w = rng.uniform(0.008, 0.04)
            pen.polygon([(x, 0.2), (x + w, 0.2), (x + w, 0.8), (x, 0.8)])
            x += w + rng.uniform(0.01, 0.05)
    elif kind == "lines":
        for _ in range(rng.integers(1, 4)):
            y = rng.uniform(0.1, 0.9)
            pen.line([(rng.uniform(0, 0.3), y), (rng.uniform(0.7, 1.0), y + rng.uniform(-0.05, 0.05))])
    elif kind == "blob":
        pen.ellipse(0.5, 0.5, rng.uniform(0.2, 0.45), rng.uniform(0.2, 0.45), fill=True)
    elif kind == "partial":
        # bars only / dots only / a lone circle without context etc.
        sub = rng.choice(["bars", "dots", "wave", "x"])
        if sub == "bars":
            _bars(pen, rng, int(rng.integers(1, 3)), 0.1, 0.9, 0.45, 0.1)
        elif sub == "dots":
            _dots(pen, rng, int(rng.integers(1, 4)), 0.5, 0.5, 0.06, 0.16)
        elif sub == "wave":
            xs = np.linspace(0.05, 0.95, 50)
            pen.line(list(zip(xs, 0.5 + 0.06 * np.sin(xs * 20))))
        else:
            _cross(pen, rng, (0.1, 0.1, 0.9, 0.9))
    elif kind == "dots":
        for _ in range(rng.integers(3, 12)):
            pen.dot(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.02, 0.06))
    elif kind == "arrow":
        pen.line([(0.1, 0.5), (0.9, 0.5)])
        pen.line([(0.65, 0.3), (0.9, 0.5), (0.65, 0.7)])
    elif kind == "text_block":
        for i in range(rng.integers(2, 4)):
            pen.text(" ".join(_rand_word(rng) for _ in range(2)), 0.5, 0.2 + i * 0.25, 0.14, rng.choice(fp))
    return pen.mask()
