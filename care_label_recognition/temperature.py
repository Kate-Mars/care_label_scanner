"""Second-look reader for the temperature printed inside a wash tub (1.3.0).

In the 64x64 normalised crop of a whole symbol, a printed "30 °C" is only ~8-10 px high, and the
symbol classifier then confuses 30/50/60 or 40/60. This module looks again at the INTERIOR of a
crop that the main classifier already considers a wash symbol: the central part of the crop
(where the number is, bars below and wave line above mostly excluded) is normalised to 64x64 on
its own, so the digits are ~2x larger, and a small CNN (same NumPy architecture as cnn.py)
predicts one of: none, 30, 40, 50, 60, 70, 95 (ASTM dots are learned as well).

The result is fused with the main classifier by re-weighting ONLY the wash classes:
    q(c) = p(c) * P_temp(temp(c))   renormalised to the same total wash probability.
So it can never turn a symbol into another category, never invents a class, and bars / hand /
crossing still come from the main classifier and the structural verifier.

Weights: models/temperature_cnn.npz/.json (training/train_temperature.py). If the files are
missing, the reader is disabled and results are exactly those of the main classifier.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cnn import forward, softmax
from .normalize import normalize_crop

TEMP_CLASSES = ["none", "30", "40", "50", "60", "70", "95"]
REGION = (0.10, 0.12, 0.90, 0.82)  # x0, y0, x1, y1 as fractions of the symbol crop


def interior(crop: np.ndarray) -> np.ndarray:
    """Central part of a symbol crop where a wash temperature is printed."""
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = REGION
    return crop[int(y0 * h):max(int(y0 * h) + 4, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 4, int(x1 * w))]


def temp_of_spec(spec: dict) -> str:
    t = spec.get("temp")
    return str(t) if t else "none"


class TemperatureReader:
    def __init__(self, model_dir: Path):
        npz, js = model_dir / "temperature_cnn.npz", model_dir / "temperature_cnn.json"
        self.ok = npz.exists() and js.exists()
        if not self.ok:
            return
        meta = json.loads(js.read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        w = np.load(npz)
        self.p = {k: w[k] for k in w.files}
        self.meta = meta

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """crops (gray uint8, dark ink) -> (N, len(classes)) probabilities."""
        x = np.stack([normalize_crop(interior(c)) for c in crops]).astype(np.float32)
        f, _ = forward(self.p, x)
        return softmax(f)
