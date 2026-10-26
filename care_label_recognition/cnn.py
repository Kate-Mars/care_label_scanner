"""Small convolutional network for symbol classification – pure NumPy inference.

The network is deliberately tiny (≈0.2 M parameters) so that it runs on CPU
with nothing but NumPy and can be shipped as a single ``.npz`` file. Weights
live in ``models/`` and are produced by ``training/train_cnn.py``; the model is
a replaceable component – any classifier exposing ``predict_proba`` with the
same class list can be swapped in.

Architecture (NHWC, input 64×64×1 ink map):
    conv3×3(16)-ReLU-maxpool → conv3×3(32)-ReLU-maxpool → conv3×3(64)-ReLU-maxpool
    → conv3×3(96)-ReLU-maxpool → fc(256)-ReLU → two heads:
       * ``fine``      – all symbol classes + not_symbol
       * ``category``  – wash / bleach / dry / iron / professional_care / not_symbol
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONV_CHANNELS = (16, 32, 64, 96)
FC = 256


def conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    """x (N,H,W,C), w (9C, K) -> (N,H,W,K), cols (for backward)."""
    n, h, wd, c = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = np.empty((n, h, wd, 3, 3, c), dtype=x.dtype)
    for i in range(3):
        for j in range(3):
            cols[:, :, :, i, j, :] = xp[:, i:i + h, j:j + wd, :]
    cols = cols.reshape(n * h * wd, 9 * c)
    out = cols @ w + b
    return out.reshape(n, h, wd, -1), cols


def maxpool2(x: np.ndarray):
    n, h, w, c = x.shape
    r = x.reshape(n, h // 2, 2, w // 2, 2, c)
    out = r.max(axis=(2, 4))
    return out, r


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def init_params(n_fine: int, n_cat: int, rng: np.random.Generator) -> dict:
    p = {}
    cin = 1
    for li, k in enumerate(CONV_CHANNELS):
        p[f"c{li}w"] = (rng.standard_normal((9 * cin, k)) * np.sqrt(2.0 / (9 * cin))).astype(np.float32)
        p[f"c{li}b"] = np.zeros(k, np.float32)
        cin = k
    flat = 4 * 4 * CONV_CHANNELS[-1]
    p["f0w"] = (rng.standard_normal((flat, FC)) * np.sqrt(2.0 / flat)).astype(np.float32)
    p["f0b"] = np.zeros(FC, np.float32)
    p["hfw"] = (rng.standard_normal((FC, n_fine)) * np.sqrt(1.0 / FC)).astype(np.float32)
    p["hfb"] = np.zeros(n_fine, np.float32)
    p["hcw"] = (rng.standard_normal((FC, n_cat)) * np.sqrt(1.0 / FC)).astype(np.float32)
    p["hcb"] = np.zeros(n_cat, np.float32)
    return p


def forward(p: dict, x: np.ndarray, cache: bool = False):
    """x (N,64,64) float32 in [0,1] -> (fine_logits, cat_logits[, cache])."""
    a = x[..., None].astype(np.float32)
    caches = []
    for li in range(len(CONV_CHANNELS)):
        z, cols = conv3x3(a, p[f"c{li}w"], p[f"c{li}b"])
        r = np.maximum(z, 0)
        o, rr = maxpool2(r)
        caches.append((a.shape, cols, z, rr, o))
        a = o
    flat = a.reshape(a.shape[0], -1)
    h = flat @ p["f0w"] + p["f0b"]
    hr = np.maximum(h, 0)
    fine = hr @ p["hfw"] + p["hfb"]
    cat = hr @ p["hcw"] + p["hcb"]
    if cache:
        return fine, cat, (caches, flat, h, hr)
    return fine, cat


class SymbolCNN:
    """Loads ``models/symbol_cnn.npz`` + ``models/symbol_cnn.json`` (class lists)."""

    def __init__(self, model_dir: Path):
        meta = json.loads((model_dir / "symbol_cnn.json").read_text(encoding="utf-8"))
        self.fine_classes: list[str] = meta["fine_classes"]
        self.cat_classes: list[str] = meta["category_classes"]
        self.fine_to_cat = np.array([self.cat_classes.index(c) for c in meta["fine_category"]])
        w = np.load(model_dir / "symbol_cnn.npz")
        self.p = {k: w[k] for k in w.files}
        self.meta = meta

    def predict(self, norms: np.ndarray, batch: int = 64):
        """norms (N,64,64) -> dict of probability arrays.

        ``hier``: hierarchical probability  P(category) · P(class | category)
        """
        fine_all, cat_all = [], []
        for i in range(0, len(norms), batch):
            f, c = forward(self.p, norms[i:i + batch])
            fine_all.append(softmax(f))
            cat_all.append(softmax(c))
        pf = np.concatenate(fine_all) if fine_all else np.zeros((0, len(self.fine_classes)))
        pc = np.concatenate(cat_all) if cat_all else np.zeros((0, len(self.cat_classes)))
        # P(class | category) from the fine head, renormalised inside each category
        denom = np.zeros_like(pc)
        for k in range(pc.shape[1]):
            denom[:, k] = pf[:, self.fine_to_cat == k].sum(1)
        cond = pf / np.maximum(denom[:, self.fine_to_cat], 1e-9)
        hier = pc[:, self.fine_to_cat] * cond
        return {"fine": pf, "category": pc, "hier": hier}
