"""Regression checks for the three user-supplied My Labels photographs."""
import unittest
from pathlib import Path

from care_label_recognition import recognize_care_label
from tests.helpers import ROOT


class TestMyLabelsRegression(unittest.TestCase):
    def check_photo(self, filename, expected_ids):
        path = ROOT / "my_labels" / filename
        self.assertTrue(path.exists(), f"Missing regression photo: {path}")
        result = recognize_care_label(str(path), verbose=True)
        self.assertEqual(result["status"], "success", result.get("message", ""))
        symbols = result.get("recognizedSymbols", [])
        got = [s["id"] for s in symbols]
        self.assertEqual(got, expected_ids)
        self.assertEqual(result.get("uncertainSymbols", []), [])
        return symbols

    def test_first_photo(self):
        self.check_photo("photo_2026-10-04_17-08-37.jpg", [
            "wash_30_gentle", "bleach_no", "dry_tumble_no", "iron_low", "pro_no",
        ])

    def test_hand_wash_keeps_30_degree_temperature(self):
        symbols = self.check_photo("photo_2026-10-04_17-08-37 (2).jpg", [
            "wash_hand_30", "iron_low", "bleach_no", "dry_drip",
        ])
        self.assertEqual(symbols[0]["components"], ["wash_hand", "wash_30"])

    def test_five_symbol_photo_including_gentle_p_circle(self):
        self.check_photo("photo_2026-10-04_17-08-37 (3).jpg", [
            "wash_30_gentle", "bleach_no", "dry_tumble_low", "iron_low",
            "pro_dry_p_very_gentle",
        ])


if __name__ == "__main__":
    unittest.main()
