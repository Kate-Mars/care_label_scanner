"""Classifier, structural verifiers and template matcher (needs models/symbol_cnn.npz)."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.helpers import ROOT, TAX
from care_label_recognition.classifier import SymbolClassifier
from care_label_recognition.normalize import normalize_crop
from care_label_recognition.templates import TemplateMatcher
from care_label_recognition.verify import check, measure
from synth.photo import isolated_sample
from synth.renderer import render_negative, render_symbol

PKG = ROOT / "care_label_recognition"


def sample(cid, rng, px=80, variant=None):
    specs = TAX.render_specs(cid)
    spec = specs[variant if variant is not None else rng.integers(len(specs))]
    return isolated_sample(render_symbol(spec, rng, split="eval"), rng, out_px=px)


class TestTaxonomy(unittest.TestCase):
    def test_every_docx_row_is_covered(self):
        covered = {c for cl in TAX.classes.values() for c in cl["components"]}
        docx = [b["id"] for b in TAX.base.values() if b["source"] == "docx"]
        self.assertEqual(len(docx), 49)
        self.assertTrue(set(docx) <= covered)

    def test_temperature_and_bars_are_separate_classes(self):
        for cid in ("wash_30", "wash_40", "wash_40_gentle", "wash_40_very_gentle", "wash_60"):
            self.assertIn(cid, TAX.classes)
        self.assertEqual(TAX.components("wash_40_gentle"), ["wash_40", "wash_machine_gentle"])

    def test_templates_manifest_matches_taxonomy(self):
        man = json.loads((PKG / "templates" / "manifest.json").read_text(encoding="utf-8"))
        for m in man:
            self.assertIn(m["class"], TAX.classes)
            self.assertTrue((PKG / "templates" / m["file"]).exists())
        self.assertEqual(len([m for m in man if m["role"] in ("app", "alt")]), 60)


class TestVerifiers(unittest.TestCase):
    def test_cross_detection(self):
        rng = np.random.default_rng(1)
        for cid, crossed in [("wash_no", True), ("bleach_no", True), ("iron_no", True), ("pro_no", True),
                             ("dry_tumble_no", True), ("wash_40", False), ("bleach_any", False),
                             ("dry_tumble_no_heat", False), ("iron_high", False), ("pro_dry_p", False)]:
            hits = 0
            for _ in range(6):
                m = measure(sample(cid, rng, px=90))
                hits += m["cross"] == crossed or not m["cross_conf"]
            self.assertGreaterEqual(hits, 5, cid)

    def test_dot_count(self):
        rng = np.random.default_rng(2)
        for cid, n in [("iron_low", 1), ("iron_medium", 2), ("iron_high", 3), ("dry_tumble_low", 1), ("dry_tumble_high", 3)]:
            ok = 0
            for _ in range(8):
                m = measure(sample(cid, rng, px=100))
                ok += m["dots"] == n or m.get("dots_merged", False)
            self.assertGreaterEqual(ok, 6, cid)

    def test_disagreement_lowers_factor(self):
        rng = np.random.default_rng(3)
        g = sample("iron_high", rng, px=110)
        m = measure(g)
        f_true, _ = check(m, TAX.expectations("iron_high"))
        f_wrong, _ = check(m, TAX.expectations("iron_low"))
        self.assertGreaterEqual(f_true, f_wrong)


class TestTemplates(unittest.TestCase):
    def test_docx_references_match_themselves(self):
        tm = TemplateMatcher(PKG / "templates")
        man = [m for m in json.loads((PKG / "templates" / "manifest.json").read_text(encoding="utf-8")) if m["role"] == "app"]
        ok = 0
        for m in man:
            g = cv2.imread(str(PKG / "templates" / m["file"]), cv2.IMREAD_GRAYSCALE)
            ok += tm.top(normalize_crop(g), 1)[0][0] == m["class"]
        self.assertEqual(ok, len(man))


class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = SymbolClassifier(TAX)

    def _classify(self, crops):
        return self.clf.classify(np.array([normalize_crop(c) for c in crops], np.float32), crops)

    def test_docx_reference_images(self):
        man = [m for m in json.loads((PKG / "templates" / "manifest.json").read_text(encoding="utf-8")) if m["role"] in ("app", "alt")]
        crops = []
        for m in man:
            g = cv2.imread(str(PKG / "templates" / m["file"]), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, None, fx=100 / max(g.shape), fy=100 / max(g.shape), interpolation=cv2.INTER_AREA)
            crops.append(np.pad(g, 8, constant_values=255))
        res = self._classify(crops)
        acc = np.mean([r.cls == m["class"] for r, m in zip(res, man)])
        self.assertGreaterEqual(acc, 0.9)

    def test_temperatures_are_distinguished(self):
        rng = np.random.default_rng(4)
        ids = ["wash_30", "wash_40", "wash_50", "wash_60", "wash_70", "wash_95"]
        crops, truth = [], []
        for cid in ids:
            for v in (0, 1):  # ISO numerals and ASTM dots
                for _ in range(3):
                    crops.append(sample(cid, rng, px=100, variant=v))
                    truth.append(cid)
        res = self._classify(crops)
        confident_wrong = sum(r.cls != t and r.confidence >= 0.7 for r, t in zip(res, truth))
        correct = sum(r.cls == t for r, t in zip(res, truth))
        self.assertGreaterEqual(correct / len(truth), 0.8)
        self.assertLessEqual(confident_wrong, 2)

    def test_generic_outline_template_does_not_veto_internal_details(self):
        # An empty triangle can correlate better than the legacy triangle with
        # "CL" because template correlation is dominated by the outer contour.
        # The generic template must not suppress the CNN's detailed class.
        self.assertFalse(self.clf._template_can_veto("bleach_any", "bleach_chlorine", True, 0.97))
        self.assertFalse(self.clf._template_can_veto("bleach_any", "bleach_non_chlorine", True, 0.97))
        # Do not suppress weaker/alternate CNN candidates using the exception.
        self.assertTrue(self.clf._template_can_veto("bleach_any", "bleach_chlorine", False, 0.60))
        self.assertTrue(self.clf._template_can_veto("bleach_any", "bleach_chlorine", True, 0.85))
        # If silhouettes differ, or the template has internal detail, a veto remains useful.
        self.assertTrue(self.clf._template_can_veto("bleach_any", "iron_high", True, 0.97))
        self.assertTrue(self.clf._template_can_veto("bleach_non_chlorine", "bleach_chlorine", True, 0.97))

    def test_letters_inside_circle(self):
        rng = np.random.default_rng(5)
        ids = ["pro_dry_a", "pro_dry_p", "pro_dry_f", "pro_wet"] * 3
        res = self._classify([sample(c, rng, px=90) for c in ids])
        correct = sum(r.cls == t for r, t in zip(res, ids))
        confident_wrong = sum(r.cls != t and r.confidence >= 0.7 for r, t in zip(res, ids))
        self.assertGreaterEqual(correct, 10)
        self.assertEqual(confident_wrong, 0)

    def test_text_and_logos_are_rejected(self):
        rng = np.random.default_rng(6)
        crops = [isolated_sample(render_negative(rng, split="eval"), rng, out_px=80) for _ in range(120)]
        crops = [c for c in crops if c is not None]
        res = self._classify(crops)
        confident_fp = sum(r.p_symbol >= 0.6 and r.confidence >= 0.7 for r in res)
        self.assertLessEqual(confident_fp / len(res), 0.05)


if __name__ == "__main__":
    unittest.main()
