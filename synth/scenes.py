"""Synthetic care-label "photos" with ground truth.

A scene is a label (printed or woven, light or dark) with an unknown number of
symbols in a row/column plus brand, composition, size, country and article
text, sometimes a logo, placed on a garment with perspective, rotation,
uneven light, shadows, blur, crumpling, noise and JPEG compression.

Negative scenes contain no symbols at all: text-only labels, logo labels,
bare fabric, other objects, a face-like drawing.

These scenes are *not* a replacement for real photos – see docs/evaluation.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import fonts as F
from .photo import blur, fabric_texture, illumination, noise_jpeg, shadow
from .renderer import WORDS, render_negative, render_symbol

ROOT = Path(__file__).resolve().parents[1]
TAX = json.loads((ROOT / "care_label_recognition" / "data" / "taxonomy.json").read_text(encoding="utf-8"))
CLASSES = TAX["classes"]
BY_CAT = {}
for _c in CLASSES:
    BY_CAT.setdefault(_c["category"], []).append(_c)
ORDER = TAX["categoryOrder"]

# Vector icon designs imported from the attached open datasets (training/import_vector_symbols.py).
# Only used when a scene asks for them (difficulty["vector_p"] > 0) and only for the requested
# sources, so the default scenes - and therefore the historical evaluation suites - are unchanged.
VECTOR_DIR = ROOT / "training" / "cache" / "vector"
_VECTOR = None


def vector_icons(sources=None):
    """{class: [ink mask float32 1=ink]} for the imported vector icon sets."""
    global _VECTOR
    if _VECTOR is None:
        _VECTOR = []
        man = VECTOR_DIR / "manifest.json"
        if man.exists():
            for m in json.loads(man.read_text(encoding="utf-8")):
                g = cv2.imread(str(VECTOR_DIR / m["file"]), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                ink = 1.0 - g.astype(np.float32) / 255.0
                ys, xs = np.where(ink > 0.3)
                if len(ys):
                    _VECTOR.append((m["source"], m["class"], ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    out = {}
    for src, cls, ink in _VECTOR:
        if sources is None or src in sources:
            out.setdefault(cls, []).append(ink)
    return out

TEXT_LINES = ["100% COTTON", "80% WOOL 20% POLYAMIDE", "MADE IN ITALY", "MADE IN CHINA", "SIZE M", "SIZE 42",
              "WASH WITH SIMILAR COLORS", "DO NOT TUMBLE DRY", "MACHINE WASH COLD", "RN 54867", "CA 12345",
              "ART. 4021-40", "LOT 30/2024", "95% VISCOSE 5% ELASTANE", "EUR 40  USA 8  UK 12", "XL",
              "WASH INSIDE OUT", "IRON ON REVERSE", "60% BAUMWOLLE 40% POLYESTER", "СОСТАВ: ХЛОПОК 100%",
              "SHELL: 100% NYLON", "LINING: 100% POLYESTER", "STYLE 4040", "40", "P", "A"]


def _text(draw, xy, s, h, font_path, fill):
    font = ImageFont.truetype(font_path, size=max(6, int(h)))
    draw.text(xy, s, font=font, fill=fill)
    return draw.textbbox(xy, s, font=font)


def _pick_symbols(rng, k):
    if rng.random() < 0.7:
        cats = sorted(rng.choice(ORDER, size=min(k, 5), replace=False), key=ORDER.index)
        while len(cats) < k:
            cats.append(rng.choice(ORDER))
    else:
        cats = list(rng.choice(ORDER, size=k))
    out = []
    for c in cats:
        cl = BY_CAT[c][rng.integers(len(BY_CAT[c]))]
        v = cl["variants"][rng.integers(len(cl["variants"]))]
        out.append((cl["id"], v["render"]))
    return out


def make_label(rng, split="eval", kind="symbols", n_symbols=None, sym_px=None, opts=None):
    """Return (label_gray float32 0..255 with dark ink, ink_layer, annotations, meta).

    ``opts`` (all optional, default = historical behaviour):
      vector_p      probability to draw a symbol from the imported vector icon sets
      vector_sources  which sets ("kobo", "hechen"); default both
      gap           symbol gap as a fraction of the symbol size (0 = touching symbols)
      grid          draw a table: every symbol in its own ruled cell (cells touch each other)
      rows          number of symbol rows (multi-row / multi-standard labels)
      classes       explicit list of class ids (overrides the random choice)
    """
    opts = opts or {}
    fonts_text = F.pool(split)
    Lw = int(rng.uniform(520, 1100))
    vertical = kind == "symbols" and rng.random() < 0.15 and n_symbols is None
    k = n_symbols if n_symbols is not None else int(rng.choice([1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7]))
    s = rng.uniform(42, 110) if not vertical else rng.uniform(42, 90)
    if sym_px:
        s = float(sym_px)
    gap = rng.uniform(0.18, 0.8) * s
    if "gap" in opts:
        gap = float(opts["gap"]) * s
    n_rows = int(opts.get("rows", 1))
    if opts.get("classes"):
        k = len(opts["classes"])
    per_row = int(np.ceil(k / n_rows)) if n_rows > 1 else k
    text_h = rng.uniform(0.18, 0.4) * s
    top_lines = int(rng.integers(0, 4))
    bot_lines = int(rng.integers(0, 4))
    if kind == "text":
        top_lines, bot_lines = int(rng.integers(2, 6)), int(rng.integers(1, 4))
    margin = rng.uniform(0.25, 0.6) * s
    if vertical:
        Lw = int(max(Lw * 0.5, s * 3.2))
        sym_block_h = k * s + (k - 1) * gap
    else:
        need = per_row * s * 1.25 + (per_row - 1) * gap + 2 * margin
        Lw = int(max(Lw, need))
        sym_block_h = s * 1.3 * n_rows
    Lh = int(margin * 2 + (top_lines + bot_lines) * text_h * 1.6 + (sym_block_h + text_h if kind == "symbols" else 0) + (s if kind == "logo" else 0))
    Lh = max(Lh, int(s * 1.6))
    img = Image.new("L", (Lw, Lh), 0)  # ink layer: 255 = ink
    d = ImageDraw.Draw(img)
    y = margin
    for _ in range(top_lines):
        _text(d, (margin, y), str(rng.choice(TEXT_LINES)), text_h, rng.choice(fonts_text), 255)
        y += text_h * 1.6
    ann = []
    ink = np.zeros((Lh, Lw), np.float32)
    if kind == "symbols":
        y += text_h * 0.5
        if opts.get("classes"):
            byid = {c["id"]: c for c in CLASSES}
            syms = []
            for cid in opts["classes"]:
                v = byid[cid]["variants"][rng.integers(len(byid[cid]["variants"]))]
                syms.append((cid, v["render"]))
        else:
            syms = _pick_symbols(rng, k)
        x = x_start = margin + rng.uniform(0, 0.3) * s
        row_y0 = y
        stroke = rng.uniform(0.04, 0.08)
        vec = vector_icons(opts.get("vector_sources")) if opts.get("vector_p", 0) > 0 else {}
        cell_pad = 0.12 * s
        for si, (cid, spec) in enumerate(syms):
            if n_rows > 1 and si and si % per_row == 0:  # next row
                x = x_start
                y += s * 1.3
            if vec.get(cid) and rng.random() < opts["vector_p"]:
                icons = vec[cid]
                m = icons[rng.integers(len(icons))].copy()
                if rng.random() < 0.5:  # vector icons have one stroke width: vary it a little
                    kk = np.ones((3, 3), np.uint8)
                    m = cv2.dilate(m, kk) if rng.random() < 0.6 else cv2.erode(m, kk)
            else:
                m = render_symbol(spec, rng, size=int(s * 2.2), split=split, stroke=stroke * rng.uniform(0.9, 1.1))
            ys, xs = np.where(m > 0.3)
            if len(ys) == 0:
                continue
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sc = s / max(m.shape)
            if spec.get("shape") in ("tub", "tub_hand", "iron", "twist") and not spec.get("bars"):
                sc *= rng.uniform(0.95, 1.15)
            m = cv2.resize(m, (max(4, int(m.shape[1] * sc)), max(4, int(m.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            mh, mw = m.shape
            if vertical:
                px, py = int(margin + (s - mw) / 2 + rng.uniform(0, s * 0.5)), int(y)
            else:
                px, py = int(x), int(y + (s - mh) / 2 + rng.uniform(-0.05, 0.05) * s)
            if px + mw >= Lw or py + mh >= Lh or px < 0 or py < 0:
                continue
            ink[py:py + mh, px:px + mw] = np.maximum(ink[py:py + mh, px:px + mw], m)
            ann.append({"class": cid, "bbox": [px, py, mw, mh]})
            if opts.get("grid"):
                lw = max(1, int(round(stroke * s * 0.6)))
                c0x, c0y = int(px - cell_pad), int(y - cell_pad)
                c1x, c1y = int(px + mw + gap / 2 + 1), int(y + s + cell_pad)
                cv2.rectangle(ink, (max(0, c0x), max(0, c0y)), (min(Lw - 1, c1x), min(Lh - 1, c1y)), 1.0, lw)
            if vertical:
                y += mh + gap
            else:
                x += mw + gap
        y = (y + text_h * 0.5) if vertical else (y + s * 1.3)
        del row_y0
    elif kind == "logo":
        m = render_negative(rng, size=int(s * 1.6), split=split)
        ys, xs = np.where(m > 0.3)
        if len(ys):
            m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            px, py = int(margin), int(y)
            mh, mw = min(m.shape[0], Lh - py - 1), min(m.shape[1], Lw - px - 1)
            ink[py:py + mh, px:px + mw] = np.maximum(ink[py:py + mh, px:px + mw], m[:mh, :mw])
        y += s
    for _ in range(bot_lines):
        if y + text_h * 1.2 >= Lh:
            break
        _text(d, (margin, y), str(rng.choice(TEXT_LINES)), text_h, rng.choice(fonts_text), 255)
        y += text_h * 1.6
    ink = np.maximum(ink, np.asarray(img, np.float32) / 255.0)
    return ink, ann, {"vertical": vertical, "n": len(ann), "symbol_px": s}


def _elastic(img, rng, amp):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h // 16 + 2, w // 16 + 2)).astype(np.float32), (0, 0), 1.5)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h // 16 + 2, w // 16 + 2)).astype(np.float32), (0, 0), 1.5)
    dx = cv2.resize(dx, (w, h)) * amp
    dy = cv2.resize(dy, (w, h)) * amp
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return dx, dy, X + dx, Y + dy


def make_scene(rng, split="eval", kind=None, n_symbols=None, difficulty=None):
    """Return (bgr uint8, annotations [{'class','bbox'}], meta)."""
    kind = kind or rng.choice(["symbols"] * 8 + ["text", "logo", "fabric", "object", "face"])
    difficulty = difficulty or {}
    meta = {"kind": kind}
    PH, PW = int(rng.uniform(800, 1400)), int(rng.uniform(900, 1600))
    garment = fabric_texture(PH, PW, rng, base=rng.uniform(40, 220), amp=rng.uniform(4, 25))
    garment_rgb = np.stack([garment * rng.uniform(0.6, 1.2) for _ in range(3)], -1)
    ann = []
    photo = np.clip(garment_rgb, 0, 255)
    if kind in ("symbols", "text", "logo"):
        lopts = {k: difficulty[k] for k in ("vector_p", "vector_sources", "gap", "grid", "rows", "classes") if k in difficulty}
        ink, lab_ann, lmeta = make_label(rng, split, kind, n_symbols, difficulty.get("sym_px"), lopts)
        meta.update(lmeta)
        Lh, Lw = ink.shape
        dark_label = rng.random() < difficulty.get("p_dark", 0.15)
        if dark_label:
            base, ink_val = rng.uniform(20, 70), rng.uniform(170, 240)
        else:
            base, ink_val = rng.uniform(185, 250), rng.uniform(0, 80)
        if "contrast" in difficulty:  # faded print / woven label: ink close to the label colour
            ink_val = base + (ink_val - base) * float(difficulty["contrast"])
        meta["dark_label"] = bool(dark_label)
        tex = fabric_texture(Lh, Lw, rng, base=base, amp=rng.uniform(2, 10))
        label = tex * (1 - ink) + ink_val * ink
        tint = np.array([rng.uniform(0.9, 1.05) for _ in range(3)])
        label_rgb = np.clip(label[..., None] * tint, 0, 255)
        # placement with perspective
        closeup = rng.random() < difficulty.get("p_closeup", 0.25)
        frac = rng.uniform(0.85, 1.25) if closeup else difficulty.get("frac", rng.uniform(0.4, 0.85))
        sc = frac * PW / Lw
        if Lh * sc > PH * (1.3 if closeup else 0.9):
            sc = PH * (1.3 if closeup else 0.9) / Lh
        persp = difficulty.get("persp", rng.choice([0.0, 0.04, 0.08, 0.14], p=[0.25, 0.35, 0.25, 0.15]))
        rot = difficulty.get("rot", rng.uniform(-12, 12))
        meta.update({"closeup": bool(closeup), "persp": float(persp), "rot": float(rot)})
        src = np.float32([[0, 0], [Lw, 0], [Lw, Lh], [0, Lh]])
        cx, cy = PW / 2 + rng.uniform(-0.1, 0.1) * PW, PH / 2 + rng.uniform(-0.1, 0.1) * PH
        dst = (src - [Lw / 2, Lh / 2]) * sc
        th = np.radians(rot)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dst = dst @ R.T + [cx, cy]
        dst += rng.uniform(-persp, persp, (4, 2)) * np.array([Lw * sc, Lh * sc])
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(label_rgb.astype(np.float32), M, (PW, PH), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpPerspective(np.ones((Lh, Lw), np.float32), M, (PW, PH), flags=cv2.INTER_LINEAR)[..., None]
        photo = photo * (1 - alpha) + warped * alpha
        for a in lab_ann:
            x, y, w, h = a["bbox"]
            pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
            p = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            if x0 < 0 or y0 < 0 or x1 > PW or y1 > PH:
                continue  # symbol cut by the frame: not counted as present
            ann.append({"class": a["class"], "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]})
    elif kind == "object":
        for _ in range(rng.integers(3, 9)):
            col = rng.uniform(0, 255, 3)
            c = (int(rng.uniform(0, PW)), int(rng.uniform(0, PH)))
            if rng.random() < 0.5:
                cv2.circle(photo, c, int(rng.uniform(20, 200)), col.tolist(), -1 if rng.random() < 0.6 else int(rng.integers(2, 12)))
            else:
                cv2.rectangle(photo, c, (c[0] + int(rng.uniform(30, 400)), c[1] + int(rng.uniform(30, 300))), col.tolist(), -1 if rng.random() < 0.6 else int(rng.integers(2, 12)))
    elif kind == "face":
        c = (PW // 2, PH // 2)
        skin = [rng.uniform(120, 200), rng.uniform(140, 210), rng.uniform(170, 240)]
        cv2.ellipse(photo, c, (int(PW * 0.18), int(PH * 0.3)), 0, 0, 360, skin, -1)
        for dx in (-1, 1):
            cv2.ellipse(photo, (c[0] + dx * int(PW * 0.06), c[1] - int(PH * 0.06)), (int(PW * 0.025), int(PH * 0.015)), 0, 0, 360, (40, 30, 30), -1)
        cv2.ellipse(photo, (c[0], c[1] + int(PH * 0.12)), (int(PW * 0.06), int(PH * 0.02)), 0, 0, 180, (60, 40, 120), 4)
        cv2.line(photo, (c[0], c[1] - int(PH * 0.03)), (c[0] - 10, c[1] + int(PH * 0.05)), (90, 90, 120), 3)
    # photographic degradations
    g = illumination(PH, PW, rng, strength=difficulty.get("light", None))
    photo = photo * g[..., None] * difficulty.get("exposure", rng.uniform(0.75, 1.15))
    if rng.random() < difficulty.get("p_shadow", 0.3):
        photo = photo * shadow(PH, PW, rng)[..., None]
        meta["shadow"] = True
    if rng.random() < difficulty.get("p_crumple", 0.2):
        dx, dy, mx, my = _elastic(photo, rng, amp=rng.uniform(2, 6))
        photo = cv2.remap(photo.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # approximate GT shift: inverse of the mean displacement inside the box
        for a in ann:
            x, y, w, h = map(int, a["bbox"])
            a["bbox"][0] -= float(dx[y:y + h, x:x + w].mean()) if w > 0 and h > 0 else 0
            a["bbox"][1] -= float(dy[y:y + h, x:x + w].mean()) if w > 0 and h > 0 else 0
        meta["crumpled"] = True
    sig = difficulty.get("blur", None)
    if sig is not None:
        photo = cv2.GaussianBlur(photo, (0, 0), sig) if sig > 0 else photo
    else:
        photo = blur(photo, rng, max_sigma=2.2)
    photo = noise_jpeg(np.clip(photo, 0, 255).astype(np.float32), rng, max_noise=difficulty.get("noise", 8.0))
    return photo.astype(np.uint8), ann, meta
