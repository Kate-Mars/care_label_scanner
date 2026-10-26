"""Public entry point: ``recognize_care_label(image) -> dict``.

photo → validation → quality → label search + perspective correction →
symbol candidates → hierarchical CNN + structural checks + templates →
duplicate suppression / consistency / order → JSON.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import postprocessing as pp
from .classifier import SymbolClassifier
from .label_detector import detect_label, estimate_skew, map_box, rotate_for_search
from .normalize import normalize_crop
from .preprocessing import InvalidImageError, ink_is_dark, load_image, to_working
from .quality import assess
from .schemas import RecognitionResult, RecognizedSymbol, UncertainSymbol
from .symbol_detector import crop, enhance_for_detection, find_candidates
from .taxonomy import load as load_taxonomy

log = logging.getLogger("care_label_recognition")

MIN_SKEW_DEG = 6.0         # photos rotated more than this are also searched de-rotated
FULL_FRAME_PENALTY = 0.95   # no label found: symbols searched in the whole photo
SMALL_SYMBOL_PX = 26        # symbols smaller than this (px in the searched image) lose 20 % confidence
MSG_NONE = "Символы ухода за одеждой не обнаружены."
MSG_QUALITY = "Качество изображения слишком низкое для распознавания символов."


def _rot_box_back(box, k, shape_rot):
    """Map [x, y, w, h] found in ``np.rot90(img, k)`` back to ``img`` coordinates."""
    x, y, w, h = box
    Hr, Wr = shape_rot
    for _ in range(k % 4):
        # one CCW step: rotated (col, row) <- original (x = Hr - row, y = col)
        x, y, w, h = Hr - (y + h), x, h, w
        Hr, Wr = Wr, Hr
    return [x, y, w, h]


class CareLabelRecognizer:
    def __init__(self, use_templates: bool = True, use_verifier: bool = True, deskew: bool = True,
                 enhance: bool = True, model_dir: str | Path | None = None, dedupe: bool = True):
        """``model_dir``: folder with symbol_cnn.npz/.json. Default: the bundled 1.3.0 weights;
        ``care_label_recognition/models/v1.2.1`` is the previous model (rollback). The
        environment variable CARE_LABEL_MODEL_DIR overrides the default as well."""
        self.deskew = deskew
        self.enhance = enhance
        # 1.3.1: every symbol id is reported once (repeated rows of multi-language labels);
        # dedupe=False restores the 1.3.0 behaviour (every physical symbol reported)
        self.dedupe = dedupe and os.environ.get("CARE_LABEL_DEDUPE", "1") != "0"
        self.tax = load_taxonomy()
        md = model_dir or os.environ.get("CARE_LABEL_MODEL_DIR")
        kw = {"model_dir": Path(md)} if md else {}
        self.clf = SymbolClassifier(self.tax, use_templates=use_templates, use_verifier=use_verifier, **kw)

    # ------------------------------------------------------------------
    def _analyse(self, img: np.ndarray, rot_k: int, det_img: np.ndarray | None = None):
        """Detect + classify in one (polarity-corrected, rotated) gray image.

        ``det_img`` (same geometry) is used for candidate detection only - e.g. a
        texture-suppressed copy; crops for classification always come from ``img``."""
        g = np.ascontiguousarray(np.rot90(img, rot_k)) if rot_k else img
        gd = g if det_img is None else (np.ascontiguousarray(np.rot90(det_img, rot_k)) if rot_k else det_img)
        cands, _, _, _ = find_candidates(gd)
        crops, norms, keep = [], [], []
        for c in cands:
            cr = crop(g, c.box)
            if cr.size < 64:
                continue
            crops.append(cr)
            norms.append(normalize_crop(cr))
            keep.append(c)
        res = self.clf.classify(np.array(norms, np.float32), crops) if norms else []
        dets = []
        for idx, (c, r) in enumerate(zip(keep, res)):
            rb = _rot_box_back(c.box, rot_k, g.shape) if rot_k else list(c.box)
            dets.append(pp.Det(box=[], rbox=tuple(rb), cls=r.cls, category=r.category, conf=r.confidence,
                               p_symbol=r.p_symbol, top=r.top, checks=r.checks, crop_idx=idx, split=c.split,
                               # reading frame: only for a label read sideways (90/270 deg); an upside-down
                               # choice (180 deg) is often made on symmetric symbols of an upright photo, and
                               # reading in that frame would reverse the row
                               obox=tuple(c.box) if rot_k in (1, 3) else None))
        pp.text_context(dets)
        return dets, crops, keep, g

    @staticmethod
    def _row_may_continue(dets, shape) -> bool:
        """True when the confident symbols reach the edge of the searched image: the
        label outline may have cut the row (a wrong quad) and more symbols can lie
        outside it - then the full frame must be searched as well."""
        good = [d for d in dets if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]
        if not good:
            return False
        H, W = shape[:2]
        for d in good:
            x, y, w, h = d.rbox
            m = 0.6 * max(w, h)
            if x < m or y < m or x + w > W - m or y + h > H - m:
                return True
        return False

    @staticmethod
    def _score(dets):
        # confident symbols count fully, uncertain ones a little (junk must not win)
        return sum(d.conf if d.conf >= pp.T_RECOGNIZED else 0.15 * d.conf
                   for d in dets if d.p_symbol >= pp.T_SYMBOLNESS and d.conf >= pp.T_UNCERTAIN)

    def _search(self, gray: np.ndarray, polarities, enhanced: float = 0.0, rotations=(0, 1, 3, 2)):
        best = None
        for dark in polarities:
            img = gray if dark else 255 - gray
            det = enhance_for_detection(img, enhanced) if enhanced else None
            for k in rotations:
                dets, crops, cands, g = self._analyse(img, k, det)
                dets = pp.suppress_duplicates(dets)
                s = self._score(dets)
                if best is None or s > best[0] * 1.15:
                    best = (s, dets, crops, k, dark, g, cands)
                good = [d for d in dets if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]
                if k == 0 and good and np.mean([d.conf for d in good]) >= 0.85:
                    break  # upright reading is convincing – no need to try rotations
            good = [d.conf for d in best[1] if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]
            if good and np.mean(good) >= 0.85:
                break  # this polarity is convincing
        return best

    # ------------------------------------------------------------------
    def recognize(self, image, debug: bool = False, debug_dir: str | Path | None = None, verbose: bool = False) -> dict:
        t0 = time.time()
        try:
            bgr = load_image(image)
        except InvalidImageError as e:
            return RecognitionResult("invalid_image", message=str(e)).to_dict()
        try:
            return self._recognize(bgr, debug, debug_dir, verbose, t0)
        except Exception as e:  # never crash the caller
            log.exception("recognition failed")
            return RecognitionResult("error", message=f"{type(e).__name__}: {e}").to_dict()

    def _recognize(self, bgr, debug, debug_dir, verbose, t0):
        gray, scale = to_working(bgr)
        q = assess(gray)
        dbg = {"quality": q.as_dict(), "workingScale": round(scale, 4)}
        featureless = "no_contrast" in q.reasons or q.blur_sigma >= 99  # no edges at all
        if q.too_low and featureless and "resolution_too_low" not in q.reasons and 40 <= q.brightness <= 248:
            # a well exposed but featureless image (bare fabric, empty wall): nothing to recognise
            r = RecognitionResult("none", imageQuality=q.score, message=MSG_NONE, debug=dbg)
            return r.to_dict(verbose)
        if q.too_low:
            r = RecognitionResult("image_quality_too_low", imageQuality=q.score, message=MSG_QUALITY, debug=dbg)
            if debug:
                self._save_debug(debug_dir, bgr, gray, None, None, [], [], r)
            return r.to_dict(verbose)
        label = detect_label(gray)
        dbg["label"] = {"found": label.found, "score": round(label.score, 3), "method": label.method}
        searches = []
        if label.found:
            searches.append(("label", label.rectified, label.Minv, [ink_is_dark(label.rectified), not ink_is_dark(label.rectified)], 1.0))
        I = np.eye(3)
        searches.append(("full_frame", gray, I, [True, False], FULL_FRAME_PENALTY))
        skew = estimate_skew(gray) if self.deskew else 0.0
        dbg["skew"] = round(skew, 1)
        if abs(skew) >= MIN_SKEW_DEG:
            # rotated label without a clean outline: search a de-rotated copy as well
            rot, Mrot = rotate_for_search(gray, skew)
            searches.append(("deskewed", rot, Mrot, [ink_is_dark(rot), not ink_is_dark(rot)], FULL_FRAME_PENALTY))
        chosen = None
        for name, img, Minv, pols, penalty in searches:
            best = self._search(img, pols)
            s, dets = best[0], best[1]
            if chosen is None or s > chosen[1][0] * 1.25:
                chosen = (name, best, Minv, penalty, img)
            good = [d.conf for d in dets if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]
            if (name == "label" and label.score >= 0.7 and len(good) >= 2 and np.mean(good) >= 0.85
                    and not self._row_may_continue(dets, img.shape)):
                break  # convincing result on the rectified label – full frame not needed
        # Recovery for faded prints, woven labels and light-on-dark labels: when the result is
        # weak (fewer than 3 confident symbols), repeat the search in the chosen geometry with
        # a texture-suppressed, contrast-equalised image for DETECTION (classification still
        # uses the original pixels). The result is only taken when it is clearly better.
        good = [d for d in chosen[1][1] if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]
        if self.enhance and len(good) < 3:
            name0, best0, Minv0, penalty0, img0 = chosen
            cand_searches = [(name0, img0, Minv0, penalty0)]
            for nm, im, Mv, pols, pen in searches:
                if nm != name0:
                    cand_searches.append((nm, im, Mv, pen))
            for strength in (1.0, 2.5):
                for nm, im, Mv, pen in cand_searches[:2]:
                    pols = [best0[4], not best0[4]] if nm == name0 else [ink_is_dark(im), not ink_is_dark(im)]
                    rots = (best0[3],) if nm == name0 else (0,)
                    b = self._search(im, pols, enhanced=strength, rotations=rots)
                    if b[0] > chosen[1][0] * 1.25 + 0.3:
                        chosen = (f"{nm}+enhanced{strength:g}", b, Mv, pen, im)
                if len([d for d in chosen[1][1] if d.conf >= pp.T_RECOGNIZED and d.p_symbol >= pp.T_SYMBOLNESS]) >= 3:
                    break
        name, (s, dets, crops, rot_k, dark, g_used, cands), Minv, penalty, search_img = chosen
        dbg.update({"searchedIn": name, "rotation": int(rot_k * 90), "darkInk": bool(dark), "candidates": len(crops)})
        qf = 0.85 + 0.15 * q.score
        for d in dets:
            d.conf *= penalty * qf
            if max(d.rbox[2], d.rbox[3]) < SMALL_SYMBOL_PX:  # tiny symbols: details not resolvable
                d.conf *= 0.8
                d.flags.append("small")
            wb = map_box(d.rbox, Minv)
            d.box = [v / scale for v in wb]
        dets = [d for d in dets if d.p_symbol >= pp.T_SYMBOLNESS and d.conf >= pp.T_UNCERTAIN]
        dets = pp.size_consistency(dets)
        dets, layout = pp.reading_order(dets)
        pp.context_bonus(dets, self.tax.category_order)
        pp.require_context_for_generic(dets)
        rec, unc = [], []
        rows = pp.row_index(dets)
        seen: dict[str, RecognizedSymbol] = {}
        n_dup = 0
        variant_of = pp.grid_variants(dets) if self.dedupe else {}
        for d, row in zip(dets, rows):
            ref = variant_of.get(id(d))
            if d.conf >= pp.T_RECOGNIZED and ref is not None and ref.cls in seen and d.cls not in seen:
                # a row that repeats an earlier row (multi-country grid): a different drawing of the
                # same category in the same column is that country's variant of the same instruction
                seen[ref.cls].repeats += 1
                seen[ref.cls].variants.append(d.cls)
                n_dup += 1
                continue
            if d.conf >= pp.T_RECOGNIZED and self.dedupe and d.cls in seen:
                # the same symbol again (multi-language / multi-standard labels repeat the whole
                # row): reported once, at its first position in reading order
                seen[d.cls].repeats += 1
                n_dup += 1
                continue
            if d.conf >= pp.T_RECOGNIZED:
                rec.append(RecognizedSymbol(id=d.cls, category=d.category, confidence=d.conf,
                                            bbox=[round(v) for v in d.box], name=self.tax.name(d.cls),
                                            components=self.tax.components(d.cls), checks={**d.checks, "flags": d.flags},
                                            attributes=self.tax.attributes(d.cls), position=len(rec) + 1,
                                            row=row if max(rows, default=1) > 1 else None))
                seen[d.cls] = rec[-1]
            else:
                alts = [c for c, p in d.top[:3] if p >= 0.05] or [d.cls]
                unc.append(UncertainSymbol(bbox=[round(v) for v in d.box], confidence=d.conf, possibleClasses=alts,
                                           probabilities=[p for c, p in d.top[:3] if p >= 0.05],
                                           reason="low_confidence" if not d.checks.get("cnn_top") else json.dumps(d.checks.get("cnn_top"), ensure_ascii=False)))
        if self.dedupe and seen:
            # an unsure candidate that may be one of the already recognised symbols adds nothing
            keep = [u for u in unc if not (set(u.possibleClasses) & set(seen))]
            n_dup += len(unc) - len(keep)
            unc = keep
        if not rec:
            status, msg = "none", MSG_NONE
        elif unc:
            status, msg = "partial", "Часть символов распознана неуверенно."
        else:
            status, msg = "success", ""
        dbg["timeMs"] = int((time.time() - t0) * 1000)
        r = RecognitionResult(status, rec, unc, imageQuality=q.score, message=msg, labelDetected=label.found,
                              orientation=layout, debug=dbg, duplicatesRemoved=n_dup if self.dedupe else None)
        if debug:
            self._save_debug(debug_dir, bgr, gray, label, (search_img, rot_k, dark, g_used, cands, crops), dets, rec + unc, r, scale)
        return r.to_dict(verbose)

    # ------------------------------------------------------------------
    def _save_debug(self, debug_dir, bgr, gray, label, search, dets, syms, result=None, scale=1.0):
        """original.jpg, detected_label.jpg, rectified_label.jpg, candidate_regions/,
        classified_symbols/, annotated.jpg, result.json – written locally only."""
        if result is None:
            result = syms
            syms = []
        d = Path(debug_dir or "debug_output")
        (d / "candidate_regions").mkdir(parents=True, exist_ok=True)
        (d / "classified_symbols").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "original.jpg"), bgr)
        if label is not None:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if label.found:
                cv2.polylines(vis, [label.quad.astype(np.int32)], True, (0, 200, 0), 3)
            cv2.imwrite(str(d / "detected_label.jpg"), vis)
            cv2.imwrite(str(d / "rectified_label.jpg"), label.rectified)
        if search is not None:
            search_img, rot_k, dark, g_used, cands, crops = search
            for i, c in enumerate(crops):
                cv2.imwrite(str(d / "candidate_regions" / f"{i:03d}.png"), c)
            for det in dets:
                if 0 <= det.crop_idx < len(crops):
                    cv2.imwrite(str(d / "classified_symbols" / f"{det.crop_idx:03d}_{det.cls}_{det.conf:.2f}.png"), crops[det.crop_idx])
        ann = bgr.copy()
        th = max(2, int(max(bgr.shape[:2]) / 500))
        for det in dets:
            x, y, w, h = [int(round(v)) for v in det.box]
            col = (0, 170, 0) if det.conf >= pp.T_RECOGNIZED else (0, 140, 255)
            cv2.rectangle(ann, (x, y), (x + w, y + h), col, th)
            cv2.putText(ann, f"{det.cls} {det.conf:.2f}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, th * 0.25), col, max(1, th // 2), cv2.LINE_AA)
        cv2.imwrite(str(d / "annotated.jpg"), ann)
        (d / "result.json").write_text(json.dumps(result.to_dict(verbose=True), ensure_ascii=False, indent=2), encoding="utf-8")


_DEFAULT: CareLabelRecognizer | None = None


def recognize_care_label(image, debug: bool = False, debug_dir: str | Path | None = None, verbose: bool = False) -> dict:
    """Recognise care symbols on a photo of a garment label.

    ``image``: path, bytes, numpy array (BGR/gray) or PIL image.
    Returns a JSON-serialisable dict, see docs/algorithm.md.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CareLabelRecognizer()
    return _DEFAULT.recognize(image, debug=debug, debug_dir=debug_dir, verbose=verbose)
