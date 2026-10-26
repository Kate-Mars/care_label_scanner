"""Image quality assessment.

Returns a score in [0, 1] and a hard ``too_low`` flag. ``too_low`` is only set
when recognition is physically impossible (tiny image, no contrast, extreme
blur) – moderate problems lower the confidence instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    score: float
    too_low: bool
    contrast: float
    blur_sigma: float
    brightness: float
    min_side: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"score": round(self.score, 3), "tooLow": self.too_low, "contrast": round(self.contrast, 1),
                "blurSigmaPx": round(self.blur_sigma, 2), "brightness": round(self.brightness, 1),
                "minSide": self.min_side, "reasons": self.reasons}


def estimate_blur_sigma(gray: np.ndarray) -> float:
    """Edge-width based blur estimate (px at the given resolution).

    For a step edge of height R blurred with a Gaussian of sigma s the maximum
    gradient is R / (sqrt(2*pi) * s). We compare strong gradients to the local
    contrast range.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    k = 31
    local_range = cv2.dilate(g, np.ones((k, k))) - cv2.erode(g, np.ones((k, k)))
    strong = local_range > max(25.0, np.percentile(local_range, 90) * 0.5)
    if strong.sum() < 50:
        return 99.0
    ratio = mag[strong] / np.maximum(local_range[strong], 1.0)
    r = float(np.percentile(ratio, 95))
    return float(1.0 / (np.sqrt(2 * np.pi) * max(r, 1e-3)))


def assess(gray: np.ndarray) -> QualityReport:
    h, w = gray.shape
    reasons = []
    p1, p99 = np.percentile(gray, [1, 99])
    contrast = float(p99 - p1)
    brightness = float(gray.mean())
    # blur measured at a normalised resolution (long side 1000 px)
    s = 1000.0 / max(h, w)
    g = cv2.resize(gray, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    sigma = estimate_blur_sigma(g)
    too_low = False
    if min(h, w) < 64:
        too_low = True
        reasons.append("resolution_too_low")
    if contrast < 20:
        too_low = True
        reasons.append("no_contrast")
    if sigma > 6.0:
        too_low = True
        reasons.append("too_blurry")
    if brightness < 20 or brightness > 250:
        reasons.append("extreme_exposure")
        if contrast < 35:
            too_low = True
    q_contrast = float(np.clip((contrast - 20) / 120, 0, 1))
    q_blur = float(np.clip(1.0 - (sigma - 0.8) / 5.0, 0, 1))
    q_res = float(np.clip((min(h, w) - 64) / 400, 0, 1))
    score = 0.45 * q_blur + 0.35 * q_contrast + 0.2 * q_res
    return QualityReport(score=score, too_low=too_low, contrast=contrast, blur_sigma=sigma,
                         brightness=brightness, min_side=min(h, w), reasons=reasons)
