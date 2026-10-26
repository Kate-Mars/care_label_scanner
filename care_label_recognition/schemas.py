"""Result structures. ``RecognitionResult.to_dict()`` is the public JSON format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUSES = ("success", "partial", "none", "image_quality_too_low", "invalid_image", "error")

HIGH, MEDIUM = 0.90, 0.70


def confidence_level(c: float) -> str:
    if c >= HIGH:
        return "high"
    if c >= MEDIUM:
        return "medium"
    return "low"


@dataclass
class RecognizedSymbol:
    id: str
    category: str
    confidence: float
    bbox: list[int]                     # [x, y, width, height] in the ORIGINAL image
    name: str = ""
    components: list[str] = field(default_factory=list)   # base symbols of the DOCX taxonomy
    variant: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)  # structural verification (debug)
    attributes: dict[str, Any] = field(default_factory=dict)  # meaning of digits / dots / bars / letters
    position: int | None = None          # 1-based reading order (rows top->bottom, left->right)
    row: int | None = None               # 1-based row index for multi-row labels
    repeats: int = 0                     # how many more times the same symbol was found (dedupe)
    variants: list[str] = field(default_factory=list)  # other drawings of it in repeated rows

    def to_dict(self, verbose: bool = False) -> dict:
        d = {"id": self.id, "category": self.category, "confidence": round(float(self.confidence), 3),
             "confidenceLevel": confidence_level(self.confidence), "bbox": [int(v) for v in self.bbox],
             "name": self.name, "components": self.components}
        # additive fields (1.3.0): older clients can ignore them
        if self.attributes:
            d["attributes"] = self.attributes
        if self.position is not None:
            d["position"] = self.position
        if self.row is not None:
            d["row"] = self.row
        if self.repeats:
            d["repeats"] = self.repeats
        if self.variants:
            d["variantsInRepeatedRows"] = sorted(set(self.variants))
        if verbose:
            d["checks"] = self.checks
        return d


@dataclass
class UncertainSymbol:
    bbox: list[int]
    confidence: float
    possibleClasses: list[str]
    probabilities: list[float] = field(default_factory=list)
    reason: str = ""

    def to_dict(self, verbose: bool = False) -> dict:
        d = {"bbox": [int(v) for v in self.bbox], "confidence": round(float(self.confidence), 3),
             "confidenceLevel": confidence_level(self.confidence), "possibleClasses": self.possibleClasses}
        if verbose:
            d["probabilities"] = [round(float(p), 3) for p in self.probabilities]
            d["reason"] = self.reason
        return d


@dataclass
class RecognitionResult:
    status: str
    recognizedSymbols: list[RecognizedSymbol] = field(default_factory=list)
    uncertainSymbols: list[UncertainSymbol] = field(default_factory=list)
    imageQuality: float | None = None
    message: str = ""
    labelDetected: bool | None = None
    orientation: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    duplicatesRemoved: int | None = None

    def to_dict(self, verbose: bool = False) -> dict:
        d: dict[str, Any] = {"status": self.status}
        if self.imageQuality is not None:
            d["imageQuality"] = round(float(self.imageQuality), 3)
        d["recognizedSymbols"] = [s.to_dict(verbose) for s in self.recognizedSymbols]
        if self.uncertainSymbols or self.status == "partial":
            d["uncertainSymbols"] = [s.to_dict(verbose) for s in self.uncertainSymbols]
        if self.message:
            d["message"] = self.message
        if self.labelDetected is not None:
            d["labelDetected"] = self.labelDetected
        if self.orientation is not None:
            d["layout"] = self.orientation
        if self.duplicatesRemoved:
            d["duplicatesRemoved"] = self.duplicatesRemoved
        if verbose and self.debug:
            d["debug"] = self.debug
        return d
