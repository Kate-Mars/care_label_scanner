import unittest

import cv2
import numpy as np

from tests.helpers import clean_label, on_background
from care_label_recognition.label_detector import detect_label, map_box
from care_label_recognition.postprocessing import iou
from care_label_recognition.recognizer import _rot_box_back
from care_label_recognition.symbol_detector import find_candidates


def covered(gt, boxes, thr=0.5):
    return all(max([iou(g, b)[0] for b in boxes] or [0]) >= thr for g in gt)


class TestCandidates(unittest.TestCase):
    def test_parts_are_grouped(self):
        # dots, digits, bars and letters must end up in the same candidate as their outline
        ids = ["wash_40_gentle", "iron_high", "dry_tumble_medium", "pro_dry_p_very_gentle", "wash_hand"]
        img, gt = clean_label(ids)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cands, *_ = find_candidates(g)
        self.assertTrue(covered(gt, [c.box for c in cands], 0.75), [c.box for c in cands])

    def test_unknown_number_of_symbols(self):
        for ids in (["bleach_no"], ["wash_30", "bleach_any", "dry_tumble_low"],
                    ["wash_60", "bleach_no", "dry_tumble", "iron_medium", "pro_dry_p", "dry_line", "wash_no_wring"]):
            img, gt = clean_label(ids)
            cands, *_ = find_candidates(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            self.assertTrue(covered(gt, [c.box for c in cands]))

    def test_astm_ticks_are_grouped(self):
        img, gt = clean_label(["pro_short_cycle", "pro_no_steam"])
        cands, *_ = find_candidates(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.assertTrue(covered(gt, [c.box for c in cands], 0.7))

    def test_low_contrast_iron_between_symbols_is_recovered(self):
        # Real woven label: the iron outline is low contrast and was fragmented
        # by the primary local-contrast mask. The adaptive recovery pass must
        # find its region between the tumble-dry and professional-care symbols.
        from pathlib import Path
        from tests.helpers import ROOT
        photo = ROOT / "my_labels" / "photo_2026-10-04_17-08-37.jpg"
        self.assertTrue(photo.exists())
        g = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
        cands, *_ = find_candidates(g)
        boxes = [c.box for c in cands]
        # Expected iron centre is about (526, 658) in the source photo.
        self.assertTrue(any(x <= 526 <= x + w and y <= 658 <= y + h and 0.8 <= w / h <= 2.0
                            for x, y, w, h in boxes), boxes)


class TestLabel(unittest.TestCase):
    def test_label_found_and_rectified(self):
        lab, gt = clean_label(["wash_40", "bleach_no", "iron_low"], text="100% COTTON")
        photo, M = on_background(lab, angle=12, persp=0.06)
        g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        L = detect_label(g)
        self.assertTrue(L.found)
        h, w = lab.shape[:2]
        # rectified aspect close to the label's aspect
        rh, rw = L.rectified.shape
        self.assertAlmostEqual(rw / rh, w / h, delta=0.35 * w / h)
        # symbols found in the rectified label map back onto the right places
        cands, *_ = find_candidates(L.rectified)
        boxes = [map_box(c.box, L.Minv) for c in cands]
        gt_photo = []
        for x, y, bw, bh in gt:
            p = cv2.perspectiveTransform(np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2), M).reshape(-1, 2)
            gt_photo.append([*p.min(0), *(p.max(0) - p.min(0))])
        self.assertTrue(covered(gt_photo, boxes, 0.45))

    def test_symbol_frame_is_not_a_label(self):
        # close-up of a single drying symbol: its square must not be taken for the label
        lab, _ = clean_label(["dry_tumble_low"], sym_px=300, gap=0)
        L = detect_label(cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY))
        if L.found:
            self.assertGreater(L.rectified.shape[0] * L.rectified.shape[1], 0.5 * lab.shape[0] * lab.shape[1])

    def test_rotation_mapping(self):
        img = np.zeros((50, 80), np.uint8)
        img[10:20, 30:45] = 1
        for k in range(4):
            r = np.rot90(img, k)
            ys, xs = np.nonzero(r)
            b = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
            self.assertEqual([int(v) for v in _rot_box_back(b, k, r.shape)], [30, 10, 15, 10])


if __name__ == "__main__":
    unittest.main()
