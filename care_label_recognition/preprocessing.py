"""Image loading/validation and generic preprocessing."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}
WORK_MAX_SIDE = 1800      # large camera photos are downscaled for processing
MIN_SIDE = 64             # below this nothing meaningful can be recognised


class InvalidImageError(ValueError):
    pass


def load_image(image) -> np.ndarray:
    """Accept a path, raw bytes, a PIL image or a numpy array; return BGR uint8.

    EXIF orientation of camera photos is applied.
    """
    if isinstance(image, np.ndarray):
        a = image
        if a.size == 0 or a.ndim not in (2, 3):
            raise InvalidImageError("empty or malformed array")
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8) if a.max() > 1.0 else (a * 255).astype(np.uint8)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        elif a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
        return a
    if isinstance(image, Image.Image):
        pil = image
    else:
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                raise InvalidImageError(f"file not found: {p}")
            data = p.read_bytes()
        elif isinstance(image, (bytes, bytearray)):
            data = bytes(image)
        else:
            raise InvalidImageError(f"unsupported input type: {type(image).__name__}")
        if not data:
            raise InvalidImageError("empty file")
        try:
            pil = Image.open(io.BytesIO(data))
            fmt = pil.format
            pil.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise InvalidImageError(f"cannot decode image: {e}") from e
        if fmt not in SUPPORTED_FORMATS:
            raise InvalidImageError(f"unsupported format: {fmt}")
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:  # broken EXIF must not break recognition
        pass
    if pil.mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGBA")
        bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        bg.alpha_composite(pil)
        pil = bg
    rgb = np.asarray(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_working(bgr: np.ndarray, max_side: int = WORK_MAX_SIDE):
    """Downscale for processing. Returns (gray, scale) with original = working / scale."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, s


def ink_is_dark(gray: np.ndarray) -> bool:
    """Polarity: printed labels are dark ink on light fabric, woven labels are
    often light thread on dark fabric. The background is the majority class."""
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray > thr).mean() >= 0.5


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a smooth background estimate (removes gradients / soft shadows).
    Gentle: no smoothing of the strokes themselves."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 8) | 1)
    small = cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 4 | 1),) * 2))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 16))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * 235.0
    return np.clip(out, 0, 255).astype(np.uint8)
