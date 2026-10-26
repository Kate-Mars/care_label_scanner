"""Access to the symbol taxonomy (data/taxonomy.json) – no class names are
hard-coded in the vision code."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "taxonomy.json"


class Taxonomy:
    def __init__(self, path: Path = DATA):
        self.raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.classes = {c["id"]: c for c in self.raw["classes"]}
        self.base = {b["id"]: b for b in self.raw["baseSymbols"]}
        self.reject = self.raw["rejectClass"]
        self.category_order = self.raw["categoryOrder"]

    def category(self, cid: str) -> str:
        return self.classes[cid]["category"] if cid in self.classes else self.reject

    def name(self, cid: str) -> str:
        return self.classes[cid]["name"] if cid in self.classes else ""

    def components(self, cid: str) -> list[str]:
        return list(self.classes[cid]["components"]) if cid in self.classes else []

    def render_specs(self, cid: str) -> list[dict]:
        return [v["render"] for v in self.classes[cid]["variants"]] if cid in self.classes else []

    def attributes(self, cid: str) -> dict:
        """Meaning carried by the details of the pictogram, as explicit fields.

        Derived from the class's (first) render spec so that it can never
        disagree with what the class depicts: temperature digits, dots, bars
        under the symbol, letters, crossing. Only keys that apply are returned.
        """
        specs = self.render_specs(cid)
        if not specs:
            return {}
        sp = specs[0]
        shape = sp.get("shape")
        a: dict = {"prohibited": bool(sp.get("cross"))}
        bars = int(sp.get("bars", 0))
        action = {0: "normal", 1: "mild", 2: "very_mild"}.get(bars)
        if shape in ("tub", "tub_hand"):
            a["process"] = "hand_wash" if shape == "tub_hand" else "wash"
            if sp.get("temp"):
                a["maxTemperatureC"] = int(sp["temp"])
            if not sp.get("cross") and shape == "tub":
                a["mechanicalAction"] = action
                a["underlines"] = bars
        elif shape == "twist":
            a["process"] = "wring"
        elif shape == "triangle":
            a["process"] = "bleach"
            if sp.get("inner") == "stripes":
                a["bleachType"] = "non_chlorine_only"
            elif sp.get("inner") == "CL":
                a["bleachType"] = "chlorine"
        elif shape == "square" and sp.get("inner") == "circle":
            a["process"] = "tumble_dry"
            if sp.get("filled"):
                a["heat"] = "none"
            elif "dots" in sp:
                a["heat"] = {0: "any", 1: "low", 2: "medium", 3: "high"}.get(int(sp["dots"]), "any")
            if not sp.get("cross"):
                a["mechanicalAction"] = action
                a["underlines"] = bars
        elif shape == "square":
            a["process"] = "natural_dry"
            inner = sp.get("inner")
            if inner in ("sag", "vlines") and sp.get("n", 1) == 1:
                a["method"] = "line"
            elif inner == "vlines":
                a["method"] = "drip"
            elif inner == "hlines":
                a["method"] = "flat" if sp.get("n", 1) == 1 else "flat_drip"
            if sp.get("shade"):
                a["shade"] = True
        elif shape == "iron":
            a["process"] = "iron"
            d = int(sp.get("dots", 0))
            if d:
                a["dots"] = d
                a["maxSoleplateTemperatureC"] = {1: 110, 2: 150, 3: 200}[d]
            if sp.get("steam_cross"):
                a["steam"] = "prohibited"
        elif shape == "circle":
            letter = sp.get("letter")
            if sp.get("tick"):
                a["restriction"] = {"ll": "short_cycle", "ul": "reduced_moisture", "lr": "low_heat", "ur": "no_steam"}[sp["tick"]]
            a["process"] = "professional_wet_clean" if letter == "W" else "professional_dry_clean"
            if letter and letter != "W":
                a["solvent"] = letter
            if not sp.get("cross") and not sp.get("tick"):
                a["mechanicalAction"] = action
                a["underlines"] = bars
        return a

    def expectations(self, cid: str) -> dict:
        """Structural attributes every variant of the class agrees on.

        Used by the verifiers: ``cross`` (bool), ``dots`` (int or set),
        ``bars`` (int), ``shape``. Attributes on which variants disagree
        (e.g. ISO numerals vs ASTM dots) are returned as sets of allowed values.
        """
        specs = self.render_specs(cid)
        if not specs:
            return {}
        exp = {}
        shapes = {s["shape"] for s in specs}
        exp["shape"] = shapes
        exp["cross"] = {bool(s.get("cross")) or (s["shape"] == "twist" and bool(s.get("cross"))) for s in specs}
        exp["steam_cross"] = {bool(s.get("steam_cross")) for s in specs}
        dots = set()
        for s in specs:
            if s["shape"] == "tub" and s.get("temp"):
                dots.add({30: 1, 40: 2, 50: 3, 60: 4, 70: 5, 95: 6}[s["temp"]] if s.get("temp_style") == "dots" else "numeral")
            elif s["shape"] in ("iron",) or (s["shape"] == "square" and s.get("inner") == "circle"):
                dots.add(int(s.get("dots", 0)))
            elif s["shape"] == "tub":
                dots.add(0)
        exp["dots"] = dots
        exp["bars"] = {int(s.get("bars", 0)) for s in specs}
        exp["filled"] = {bool(s.get("filled")) for s in specs}
        return exp


@lru_cache(maxsize=4)
def load(path: str | None = None) -> Taxonomy:
    return Taxonomy(Path(path) if path else DATA)
