"""1.3.1: repeated symbols are reported once; the diagonal outline hand-wash pictogram is read.

The two photos in my_labels/ dated 2026-10-25 were sent by the user and their symbols were ALSO
used as training examples (user request) - these tests are regression checks, not a measure of
generalisation (see docs/improvement_report_1.3.1_ru.md).
"""
import unittest

import cv2
import numpy as np

from care_label_recognition import CareLabelRecognizer, recognize_care_label
from tests.helpers import ROOT, clean_label


def ids(r):
    return [s["id"] for s in r.get("recognizedSymbols", [])]


class TestDedupe(unittest.TestCase):
    ROW = ["wash_40", "bleach_no", "dry_tumble_low", "iron_medium", "pro_no"]

    def _two_rows(self):
        a, _ = clean_label(self.ROW, sym_px=80, gap=30, seed=511)
        b, _ = clean_label(self.ROW, sym_px=80, gap=30, seed=512)
        return np.vstack([a[: a.shape[0] * 2 // 3], b[: b.shape[0] * 2 // 3]])

    def test_repeated_row_reported_once(self):
        r = recognize_care_label(self._two_rows())
        self.assertEqual(ids(r), self.ROW)
        self.assertEqual(len(set(ids(r))), len(ids(r)))
        self.assertGreaterEqual(r.get("duplicatesRemoved", 0), 4)
        self.assertTrue(any(s.get("repeats", 0) >= 1 for s in r["recognizedSymbols"]))

    def test_dedupe_can_be_switched_off(self):
        r = CareLabelRecognizer(dedupe=False).recognize(self._two_rows())
        self.assertGreater(len(ids(r)), len(self.ROW))


class TestUserPhotos20261025(unittest.TestCase):
    def test_grid_with_five_repeated_rows(self):
        r = recognize_care_label(str(ROOT / "my_labels" / "photo_2026-10-25_grid_repeated.png"))
        got = ids(r)
        self.assertEqual(len(got), len(set(got)), got)
        for c in ("wash_no", "bleach_no", "dry_tumble_no", "iron_no", "pro_no"):
            self.assertIn(c, got)
        self.assertTrue(set(got) <= {"wash_no", "bleach_no", "dry_tumble_no", "iron_no", "pro_no", "dry_no"}, got)

    def test_sweater_hand_wash_is_found(self):
        r = recognize_care_label(str(ROOT / "my_labels" / "photo_2026-10-25_sweater_hand_wash.png"))
        self.assertEqual(ids(r), ["wash_hand", "bleach_no", "iron_no", "pro_dry_f_very_gentle", "dry_flat"])
        self.assertEqual(r["recognizedSymbols"][0]["attributes"]["process"], "hand_wash")


if __name__ == "__main__":
    unittest.main()
