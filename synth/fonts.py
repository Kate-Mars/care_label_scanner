"""Font pools for digits/letters inside symbols and for label text.

Two disjoint pools: TRAIN fonts are used to generate training data, EVAL fonts
only for the synthetic evaluation scenes, so evaluation measures generalisation
to unseen typefaces.
"""
from __future__ import annotations

import glob
import os

_D = "/usr/share/fonts"
_T = "/usr/share/texmf/fonts/opentype/public"


def _existing(paths):
    out = []
    for p in paths:
        out.extend(sorted(glob.glob(p)))
    return [p for p in out if os.path.exists(p)]


TRAIN_FONTS = _existing([
    f"{_D}/truetype/dejavu/DejaVuSans*.ttf",
    f"{_D}/truetype/liberation/LiberationSans-*.ttf",
    f"{_D}/truetype/liberation/LiberationSerif-*.ttf",
    f"{_D}/truetype/freefont/FreeSans*.ttf",
    f"{_D}/opentype/inter/Inter-*.otf",
    f"{_D}/truetype/crosextra/Carlito-*.ttf",
    f"{_T}/tex-gyre/texgyreheros-*.otf",
    f"{_T}/tex-gyre/texgyretermes-*.otf",
    f"{_T}/lm/lmsans10-*.otf",
])
EVAL_FONTS = _existing([
    f"{_D}/truetype/google-fonts/Poppins-*.ttf",
    f"{_D}/truetype/crosextra/Caladea-*.ttf",
    f"{_D}/truetype/liberation/LiberationMono-*.ttf",
    f"{_D}/truetype/freefont/FreeSerif*.ttf",
    f"{_T}/tex-gyre/texgyreadventor-*.otf",
    f"{_T}/tex-gyre/texgyreheroscn-*.otf",
    f"{_D}/opentype/inter/InterDisplay-*.otf",
])
# fonts whose glyphs are too thin to survive camera blur are dropped from symbol interiors
_THIN = ("Thin", "ExtraLight", "Light")
TRAIN_SYMBOL_FONTS = [f for f in TRAIN_FONTS if not any(t in f for t in _THIN)]
EVAL_SYMBOL_FONTS = [f for f in EVAL_FONTS if not any(t in f for t in _THIN)]


def pool(split: str, symbol: bool = False):
    if split == "eval":
        return EVAL_SYMBOL_FONTS if symbol else EVAL_FONTS
    return TRAIN_SYMBOL_FONTS if symbol else TRAIN_FONTS
