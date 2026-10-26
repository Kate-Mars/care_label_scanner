"""Ensemble classification of symbol candidates.

1. CNN (hierarchical): P(category) · P(class | category), plus P(not_symbol).
2. Structural verification (verify.py): crossing / dots / bars re-measured.
3. Template matching (templates.py): independent second opinion.

Fusion never invents a class: the candidates are the CNN's top-k symbol
classes; verifiers and templates can only *lower* a class's score. When they
disagree with the CNN's first choice, the resulting confidence is low and the
symbol is reported as uncertain instead of being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import verify
from .cnn import SymbolCNN
from .taxonomy import Taxonomy
from .temperature import TemperatureReader, temp_of_spec
from .templates import TemplateMatcher

PKG = Path(__file__).resolve().parent
TEMPLATE_TRUST = 0.75   # template correlation above which the matcher may veto
OCCLUSION_MIN = 0.03     # verify.border_solid above this: crop corrupted by a shadow / foreign object
OCCLUSION_FACTOR = 0.8   # ... confidence multiplied (usually -> "uncertain" instead of a guess)
TEMP_FUSE_BELOW = 0.80  # temperature reader is consulted when the best wash class has < 80 % of the wash mass
TEMP_READER_TRUST = 0.85  # ... and only used when the reader itself is at least this sure


@dataclass
class Classification:
    cls: str
    category: str
    confidence: float
    p_symbol: float
    top: list[tuple[str, float]] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    cnn_top: str = ""
    template_top: str = ""


class SymbolClassifier:
    def __init__(self, taxonomy: Taxonomy, model_dir: Path = PKG / "models", templ_dir: Path = PKG / "templates",
                 use_templates: bool = True, use_verifier: bool = True, use_temperature: bool = True):
        self.tax = taxonomy
        self.cnn = SymbolCNN(model_dir)
        # second look at the number inside wash tubs (models/temperature_cnn.*; disabled if absent)
        self.temp = TemperatureReader(model_dir) if use_temperature else None
        if self.temp is not None and not self.temp.ok:
            self.temp = None
        self._wash_idx, self._wash_temp = [], []
        if self.temp is not None:
            for j, c in enumerate(self.cnn.fine_classes):
                specs = taxonomy.render_specs(c)
                if taxonomy.category(c) == "wash" and specs and all(sp.get("shape") in ("tub", "tub_hand") for sp in specs):
                    self._wash_idx.append(j)
                    self._wash_temp.append(self.temp.classes.index(temp_of_spec(specs[0])))
            self._wash_idx = np.array(self._wash_idx, int)
            self._wash_temp = np.array(self._wash_temp, int)
            self._wash_cat = self.cnn.cat_classes.index("wash")
        self.tm = TemplateMatcher(templ_dir) if use_templates else None
        self.use_verifier = use_verifier
        self.reject_idx = self.cnn.fine_classes.index(taxonomy.reject)
        self.cat_reject_idx = self.cnn.cat_classes.index(taxonomy.reject)
        self._exp = {c: taxonomy.expectations(c) for c in taxonomy.classes}
        # A plain outline template often matches a detailed variant because its
        # correlation is dominated by the outer shape (e.g. an empty triangle
        # can outscore a triangle containing "CL"). Cache the rendered shapes
        # so such a generic template cannot veto the CNN's fine-grained result.
        self._render_specs = {c: taxonomy.render_specs(c) for c in taxonomy.classes}
        self._shapes = {
            c: {spec.get("shape") for spec in specs if spec.get("shape")}
            for c, specs in self._render_specs.items()
        }

    def _template_can_veto(self, template_cls: str, candidate_cls: str,
                           candidate_is_cnn_top: bool, candidate_probability: float) -> bool:
        """Whether a template winner provides useful evidence against a candidate.

        Template correlation is unreliable for internal details when a plain
        outline wins on silhouette alone (e.g. an empty triangle vs. a triangle
        marked "CL"). Protect only a strong CNN top prediction in that exact
        case; weaker alternatives still receive the template disagreement
        penalty, which helps preserve precision on ambiguous crops.
        """
        specs = self._render_specs.get(template_cls, [])
        outline_only = bool(specs) and all(set(spec) <= {"shape"} for spec in specs)
        same_shape = bool(self._shapes.get(template_cls, set()) & self._shapes.get(candidate_cls, set()))
        generic_outline_conflict = outline_only and same_shape and candidate_is_cnn_top and candidate_probability >= 0.90
        # 1.3.0: the same holds for any template that is a strict SUBSET of the candidate's drawing
        # (e.g. a crossed empty square "dry_no" vs a crossed square with a circle "dry_tumble_no"):
        # the simpler drawing correlates well with the richer one and must not veto it
        cand_specs = self._render_specs.get(candidate_cls, [])
        subset_conflict = (candidate_is_cnn_top and candidate_probability >= 0.90 and bool(specs) and bool(cand_specs)
                           and any(all(cs.get(k) == v for k, v in ts.items() if v) and
                                   len([v for v in cs.values() if v]) > len([v for v in ts.values() if v])
                                   for ts in specs for cs in cand_specs))
        return not (generic_outline_conflict or subset_conflict)

    def classify(self, norms: np.ndarray, crops: list[np.ndarray], topk: int = 5) -> list[Classification]:
        if len(norms) == 0:
            return []
        P = self.cnn.predict(norms)
        hier, cat = P["hier"], P["category"]
        temp_info = {}
        if self.temp is not None and len(self._wash_idx):
            # fuse the temperature reader into the wash classes of crops that are mainly washing symbols
            # only where the main classifier is unsure WHICH wash class it is (the number is the
            # usual reason); a confident main reading is never overridden
            sel = [i for i in range(len(norms)) if cat[i, self._wash_cat] >= 0.6
                   and float(hier[i, self._wash_idx].max()) < TEMP_FUSE_BELOW * float(cat[i, self._wash_cat])]
            if sel:
                pt = self.temp.predict([crops[i] for i in sel])
                hier = hier.copy()
                for k, i in enumerate(sel):
                    if float(pt[k].max()) < TEMP_READER_TRUST:
                        continue  # the reader is not sure either: leave the main result (uncertain)
                    w = hier[i, self._wash_idx]
                    mass = float(w.sum())
                    q = w * pt[k, self._wash_temp]
                    if mass > 0 and q.sum() > 0:
                        hier[i, self._wash_idx] = q / q.sum() * mass
                    temp_info[i] = {self.temp.classes[j]: round(float(pt[k, j]), 3) for j in np.argsort(-pt[k])[:2]}
        out = []
        for i in range(len(norms)):
            p_sym = float(1.0 - cat[i, self.cat_reject_idx])
            ps = hier[i].copy()
            ps[self.reject_idx] = -1
            order = np.argsort(-ps)[:topk]
            cands = [(self.cnn.fine_classes[j], float(hier[i, j])) for j in order]
            cnn_top = cands[0][0]
            checks, tmpl_top = {}, ""
            if i in temp_info:
                checks["temperature_reader"] = temp_info[i]
            adj = dict(cands)
            if p_sym >= 0.5 and cands[0][1] >= 0.2:
                if self.use_verifier:
                    meas = verify.measure(crops[i])
                    checks["measured"] = {k: v for k, v in meas.items() if k in ("dots", "bars", "cross", "glyphs", "diag_cov", "src_px", "dots_merged")}
                    for c, p in cands:
                        f, det = verify.check(meas, self._exp.get(c, {}))
                        adj[c] = p * f
                        if c == cnn_top:
                            checks["cnn_top"] = det
                if self.tm is not None:
                    ts = self.tm.scores(norms[i])
                    tbest_cls, tbest = max(ts.items(), key=lambda kv: kv[1])
                    tmpl_top = tbest_cls
                    # the matcher is a weak classifier on its own (see docs/evaluation.md), so its
                    # opinion only counts when it is itself confident
                    if tbest >= TEMPLATE_TRUST:
                        for c, cnn_probability in cands:
                            # A generic outline (e.g. empty triangle) must not
                            # veto a detailed symbol with the same outer shape.
                            # It may simply have failed to resolve the small
                            # interior mark, even when the CNN recognises it.
                            if not self._template_can_veto(tbest_cls, c, c == cnn_top, cnn_probability):
                                continue
                            gap = tbest - ts.get(c, 0.0)
                            if gap > 0.08:  # templates clearly prefer another class
                                adj[c] *= float(np.clip(1.0 - (gap - 0.08) * 1.5, 0.7, 1.0))
                    checks["template"] = {"best": tbest_cls, "score": round(tbest, 3)}
            occ = verify.border_solid(norms[i])
            if occ >= OCCLUSION_MIN:
                adj = {c: v * OCCLUSION_FACTOR for c, v in adj.items()}
                checks["occluded"] = round(occ, 3)
            best = max(adj.items(), key=lambda kv: kv[1])
            top = sorted(adj.items(), key=lambda kv: -kv[1])
            out.append(Classification(cls=best[0], category=self.tax.category(best[0]), confidence=float(best[1]),
                                      p_symbol=p_sym, top=top, checks=checks, cnn_top=cnn_top, template_top=tmpl_top))
        return out
