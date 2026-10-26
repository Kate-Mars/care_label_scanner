"""Regression tests added in 1.3.0: whole rows of any length, meaningful details, empty results,
touching symbols, ruled grids, faded ink, blur, explicit attributes and reading positions.

All images are generated here (procedural renderer with the TRAIN pool fixed seeds), so the
tests do not depend on the training cache or on downloaded datasets.
"""
import unittest

import cv2
import numpy as np

from care_label_recognition import recognize_care_label
from tests.helpers import clean_label


def ids(result):
    return [s["id"] for s in result.get("recognizedSymbols", [])]


def degrade(img, contrast=1.0, blur=0.0, seed=0):
    """Fade the ink towards the label colour and/or blur (meaning preserved)."""
    g = img.astype(np.float32)
    if contrast < 1.0:
        g = 240 - (240 - g) * contrast
    if blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    g += np.random.default_rng(seed).normal(0, 2.0, g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


ROW15 = ["wash_30_gentle", "wash_40", "wash_60", "bleach_no", "bleach_any", "dry_tumble_low",
         "dry_tumble_no", "dry_flat", "dry_drip", "iron_low", "iron_high", "pro_dry_a",
         "pro_dry_p_very_gentle", "pro_wet", "wash_hand"]


class TestRowLengths(unittest.TestCase):
    """The detector must not stop after 4-5 symbols and must keep the order."""

    def _row(self, expected, seed, **kw):
        img, _ = clean_label(expected, sym_px=90, gap=kw.pop("gap", 22), seed=seed)
        r = recognize_care_label(img)
        self.assertEqual(ids(r), expected)
        self.assertEqual(r.get("uncertainSymbols", []), [])
        self.assertEqual([s["position"] for s in r["recognizedSymbols"]], list(range(1, len(expected) + 1)))
        return r

    def test_rows_of_3_5_8_10_15(self):
        for n, seed in ((3, 301), (5, 302), (8, 303), (10, 304), (15, 305)):
            with self.subTest(n=n):
                self._row(ROW15[:n], seed)


class TestMeaningfulDetails(unittest.TestCase):
    """Digits, bars, dots, letters and crossings change the class - and are reported as attributes."""

    def _check(self, expected, seed):
        img, _ = clean_label(expected, sym_px=96, gap=30, seed=seed)
        r = recognize_care_label(img)
        self.assertEqual(ids(r), expected)
        return {s["id"]: s for s in r["recognizedSymbols"]}

    def test_temperature_and_bars_under_the_tub(self):
        s = self._check(["wash_30", "wash_30_gentle", "wash_30_very_gentle", "wash_40_gentle"], 411)
        self.assertEqual(s["wash_30_gentle"]["attributes"]["maxTemperatureC"], 30)
        self.assertEqual(s["wash_30_gentle"]["attributes"]["underlines"], 1)
        self.assertEqual(s["wash_30_very_gentle"]["attributes"]["underlines"], 2)
        self.assertEqual(s["wash_40_gentle"]["attributes"]["maxTemperatureC"], 40)

    def test_hand_wash_versus_machine_wash_keeps_temperature(self):
        s = self._check(["wash_hand_30", "wash_30", "wash_hand", "wash_40"], 412)
        self.assertEqual(s["wash_hand_30"]["components"], ["wash_hand", "wash_30"])
        self.assertEqual(s["wash_hand_30"]["attributes"]["process"], "hand_wash")
        self.assertEqual(s["wash_hand_30"]["attributes"]["maxTemperatureC"], 30)
        self.assertEqual(s["wash_30"]["attributes"]["process"], "wash")
        self.assertNotIn("maxTemperatureC", s["wash_hand"]["attributes"])

    def test_iron_dots(self):
        s = self._check(["iron_low", "iron_medium", "iron_high", "iron_no"], 413)
        self.assertEqual(s["iron_low"]["attributes"]["dots"], 1)
        self.assertEqual(s["iron_medium"]["attributes"]["dots"], 2)
        self.assertEqual(s["iron_high"]["attributes"]["maxSoleplateTemperatureC"], 200)
        self.assertTrue(s["iron_no"]["attributes"]["prohibited"])

    def test_professional_p_with_bars(self):
        s = self._check(["pro_dry_p", "pro_dry_p_very_gentle", "pro_dry_p_max_gentle", "pro_no"], 414)
        self.assertEqual(s["pro_dry_p"]["attributes"]["underlines"], 0)
        self.assertEqual(s["pro_dry_p_very_gentle"]["attributes"]["underlines"], 1)
        self.assertEqual(s["pro_dry_p_max_gentle"]["attributes"]["underlines"], 2)
        self.assertEqual(s["pro_dry_p_very_gentle"]["attributes"]["solvent"], "P")

    def test_crossed_differs_from_allowed(self):
        self._check(["bleach_any", "bleach_no", "dry_tumble", "dry_tumble_no"], 415)


class TestEmptyResults(unittest.TestCase):
    def test_blank_label_returns_explicit_none(self):
        img = np.full((300, 700, 3), 236, np.uint8)
        img = degrade(img, seed=5)
        r = recognize_care_label(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img)
        self.assertEqual(r["status"], "none")
        self.assertEqual(r["recognizedSymbols"], [])

    def test_text_without_care_symbols(self):
        img = np.full((420, 900), 238, np.uint8)
        for i, line in enumerate(["100% COTTON", "MADE IN PORTUGAL", "RN 54867  CA 12345", "SIZE M / 40"]):
            cv2.putText(img, line, (30, 70 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 25, 3, cv2.LINE_AA)
        r = recognize_care_label(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        self.assertEqual(ids(r), [])
        self.assertEqual(r["status"], "none")


class TestHardPrints(unittest.TestCase):
    def test_faded_and_slightly_blurred_row(self):
        expected = ["wash_40", "bleach_no", "dry_tumble_low", "iron_medium", "pro_no"]
        img, _ = clean_label(expected, sym_px=90, gap=26, seed=421)
        g = degrade(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), contrast=0.45, blur=1.0, seed=421)
        r = recognize_care_label(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        self.assertEqual(ids(r), expected)

    def test_touching_symbols_are_split(self):
        expected = ["wash_30_gentle", "bleach_no", "dry_tumble_low", "iron_low", "pro_no"]
        img, _ = clean_label(expected, sym_px=90, gap=0, seed=422)
        r = recognize_care_label(img)
        self.assertEqual(ids(r), expected)

    def test_symbols_in_a_ruled_table(self):
        expected = ["wash_40", "bleach_no", "iron_medium", "pro_no", "dry_tumble_low"]
        img, boxes = clean_label(expected, sym_px=86, gap=34, seed=423)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y0 = min(b[1] for b in boxes) - 14
        y1 = max(b[1] + b[3] for b in boxes) + 14
        xs = [boxes[0][0] - 16] + [b[0] + b[2] + 17 for b in boxes]
        cv2.rectangle(g, (xs[0], y0), (xs[-1], y1), 20, 4)
        for x in xs[1:-1]:
            cv2.line(g, (x, y0), (x, y1), 20, 4)
        r = recognize_care_label(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        self.assertEqual(ids(r), expected)


if __name__ == "__main__":
    unittest.main()
