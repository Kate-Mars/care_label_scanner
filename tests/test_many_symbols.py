"""Regression test: do not impose a four-symbol cap on long care-label rows."""
import tempfile
import unittest
from pathlib import Path

import cv2

from care_label_recognition import recognize_care_label
from tests.helpers import clean_label


class TestManySymbols(unittest.TestCase):
    def _recognize_row(self, expected, seed):
        image, _ = clean_label(expected, sym_px=90, gap=20, seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long_care_label.jpg"
            self.assertTrue(cv2.imwrite(str(path), image))
            result = recognize_care_label(str(path), verbose=True)
        self.assertEqual(result["status"], "success", result.get("message", ""))
        symbols = result.get("recognizedSymbols", [])
        got = [s["id"] for s in symbols]
        self.assertEqual(len(got), len(expected))
        self.assertEqual(got, expected)
        self.assertEqual(result.get("uncertainSymbols", []), [])

    def test_ten_symbol_row_including_hand_temperature(self):
        expected = [
            "wash_hand_30", "wash_30_gentle", "wash_40", "wash_60",
            "bleach_no", "bleach_any", "dry_tumble_low", "dry_tumble_no",
            "dry_flat", "dry_drip",
        ]
        self._recognize_row(expected, seed=20260)

    def test_fifteen_symbol_row_is_read_in_order(self):
        expected = [
            "wash_30_gentle", "wash_40", "wash_60", "bleach_no", "bleach_any",
            "dry_tumble_low", "dry_tumble_no", "dry_flat", "dry_drip", "iron_low",
            "iron_high", "pro_dry_a", "pro_dry_p_very_gentle", "pro_wet", "wash_hand",
        ]
        self._recognize_row(expected, seed=20262)



if __name__ == "__main__":
    unittest.main()
