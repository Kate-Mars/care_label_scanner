"""Post-processing units + end-to-end behaviour of recognize_care_label."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.helpers import clean_label, on_background
from care_label_recognition import postprocessing as pp
from care_label_recognition import recognize_care_label


def det(x, y, w, h, cls="wash_40", cat="wash", conf=0.9):
    return pp.Det(box=[x, y, w, h], rbox=(x, y, w, h), cls=cls, category=cat, conf=conf, p_symbol=0.99)


class TestUnits(unittest.TestCase):
    def test_duplicates_suppressed(self):
        d = pp.suppress_duplicates([det(10, 10, 50, 50, conf=0.8), det(12, 11, 49, 52, "wash_30", conf=0.95), det(100, 10, 50, 50)])
        self.assertEqual(len(d), 2)
        self.assertIn("wash_30", [x.cls for x in d])

    def test_reading_order(self):
        ds = [det(300, 10, 50, 50, "c"), det(10, 12, 50, 50, "a"), det(150, 8, 50, 50, "b")]
        o, layout = pp.reading_order(ds)
        self.assertEqual([x.cls for x in o], ["a", "b", "c"])
        self.assertEqual(layout, "horizontal")
        ds = [det(10, 300, 50, 50, "c"), det(12, 10, 50, 50, "a"), det(8, 150, 50, 50, "b")]
        o, layout = pp.reading_order(ds)
        self.assertEqual([x.cls for x in o], ["a", "b", "c"])
        self.assertEqual(layout, "vertical")

    def test_context_never_changes_class(self):
        ds = [det(0, 0, 10, 10, "wash_40", "wash"), det(20, 0, 10, 10, "bleach_no", "bleach"), det(40, 0, 10, 10, "iron_low", "iron")]
        pp.context_bonus(ds, ["wash", "bleach", "dry", "iron", "professional_care"])
        self.assertEqual([d.cls for d in ds], ["wash_40", "bleach_no", "iron_low"])
        self.assertTrue(all(d.conf <= 0.99 for d in ds))


class TestEndToEnd(unittest.TestCase):
    def test_clean_label_success_in_order(self):
        ids = ["wash_40", "bleach_no", "dry_tumble_low", "iron_medium", "pro_dry_p"]
        img, _ = clean_label(ids, text="100% COTTON  MADE IN ITALY")
        r = recognize_care_label(img)
        self.assertIn(r["status"], ("success", "partial"))
        got = [s["id"] for s in r["recognizedSymbols"]]
        self.assertGreaterEqual(sum(g == t for g, t in zip(got, ids)), 4, got)
        for s in r["recognizedSymbols"]:
            self.assertEqual(set(s) >= {"id", "category", "confidence", "confidenceLevel", "bbox"}, True)
            self.assertEqual(len(s["bbox"]), 4)
        json.dumps(r)

    def test_blurry_cl_bleach_symbol_is_recognized(self):
        # Regression: an outline-only empty-triangle template used to veto the
        # CNN's correct "bleach_chlorine" prediction on this low-contrast CL mark.
        image = Path(__file__).resolve().parents[1] / "examples" / "input" / "label_05_long_row_blur.jpg"
        r = recognize_care_label(str(image))
        got = [s["id"] for s in r["recognizedSymbols"]]
        self.assertIn("bleach_chlorine", got, r)
        self.assertEqual(r["status"], "success", r)
        self.assertEqual(len(r.get("uncertainSymbols", [])), 0, r)

    def test_three_symbols_no_invented_categories(self):
        img, _ = clean_label(["wash_30", "bleach_any", "dry_line"])
        r = recognize_care_label(img)
        cats = [s["category"] for s in r["recognizedSymbols"]]
        self.assertLessEqual(len(cats), 3)
        self.assertNotIn("iron", cats)
        self.assertNotIn("professional_care", cats)

    def test_rotated_photo(self):
        img, _ = clean_label(["wash_60", "iron_high", "pro_dry_f"])
        photo, _ = on_background(img, angle=10, persp=0.05)
        r = recognize_care_label(photo)
        self.assertGreaterEqual(len([s for s in r["recognizedSymbols"] if s["id"] in ("wash_60", "iron_high", "pro_dry_f")]), 2)

    def test_none_cases(self):
        blank = np.full((600, 800, 3), 235, np.uint8)
        noise = (np.random.default_rng(0).normal(128, 40, (600, 800, 3))).clip(0, 255).astype(np.uint8)
        text = np.full((400, 900), 240, np.uint8)
        for i, t in enumerate(["100% COTTON", "MADE IN ITALY", "SIZE M  40", "WASH WITH SIMILAR COLORS", "RN 54867  P  A  F"]):
            cv2.putText(text, t, (30, 60 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 20, 3, cv2.LINE_AA)
        for img in (blank, noise, cv2.cvtColor(text, cv2.COLOR_GRAY2BGR)):
            r = recognize_care_label(img)
            self.assertIn(r["status"], ("none", "image_quality_too_low"))
            self.assertEqual(r["recognizedSymbols"], [])
        self.assertEqual(recognize_care_label(blank)["status"], "none")          # featureless, well exposed
        dark = np.full((600, 800, 3), 8, np.uint8)
        self.assertEqual(recognize_care_label(dark)["status"], "image_quality_too_low")  # black frame

    def test_invalid_and_tiny(self):
        self.assertEqual(recognize_care_label(b"garbage")["status"], "invalid_image")
        self.assertEqual(recognize_care_label("/no/such/file.jpg")["status"], "invalid_image")
        self.assertEqual(recognize_care_label(np.full((30, 30, 3), 200, np.uint8))["status"], "image_quality_too_low")

    def test_debug_output(self):
        img, _ = clean_label(["wash_40", "iron_low"])
        with tempfile.TemporaryDirectory() as d:
            recognize_care_label(img, debug=True, debug_dir=d)
            for f in ("original.jpg", "detected_label.jpg", "rectified_label.jpg", "annotated.jpg", "result.json"):
                self.assertTrue((Path(d) / f).exists(), f)
            self.assertTrue(any((Path(d) / "candidate_regions").iterdir()))


if __name__ == "__main__":
    unittest.main()
