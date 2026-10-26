"""Template-matching classifier (method A, model-free baseline).

Every reference image in ``templates/`` (DOCX references + procedural
canonical drawings) is normalised exactly like a candidate. A candidate is
compared with each template by normalised cross-correlation of slightly
blurred ink maps over ±3 px shifts. Used on its own for comparison and, in
the ensemble, as an independent second opinion.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .normalize import normalize_crop

SHIFT = 3


class TemplateMatcher:
    def __init__(self, templ_dir: Path):
        man = json.loads((templ_dir / "manifest.json").read_text(encoding="utf-8"))
        self.entries = man
        imgs, cls = [], []
        for m in man:
            g = cv2.imread(str(templ_dir / m["file"]), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            imgs.append(self._prep(normalize_crop(g)))
            cls.append(m["class"])
        self.T = np.stack(imgs).astype(np.float32)
        self.cls = np.array(cls)
        self.classes = sorted(set(cls))

    @staticmethod
    def _prep(norm):
        return cv2.GaussianBlur(norm.astype(np.float32), (0, 0), 1.2)

    def scores(self, norm: np.ndarray) -> dict[str, float]:
        c = np.pad(self._prep(norm), SHIFT)
        best: dict[str, float] = {}
        for t, k in zip(self.T, self.cls):
            s = float(cv2.matchTemplate(c, t, cv2.TM_CCOEFF_NORMED).max())
            if s > best.get(k, -1.0):
                best[k] = s
        return best

    def top(self, norm: np.ndarray, k: int = 3):
        s = self.scores(norm)
        return sorted(s.items(), key=lambda kv: -kv[1])[:k]
