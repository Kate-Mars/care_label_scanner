import io
import unittest

import cv2
import numpy as np
from PIL import Image

from tests.helpers import clean_label  # noqa: F401  (sets sys.path)
from care_label_recognition.normalize import features, ink_map, normalize_crop
from care_label_recognition.preprocessing import InvalidImageError, ink_is_dark, load_image, to_working
from care_label_recognition.quality import assess


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.img, _ = clean_label(["wash_40"])

    def test_formats(self):
        for ext in (".jpg", ".png", ".webp"):
            ok, enc = cv2.imencode(ext, self.img)
            self.assertTrue(ok)
            out = load_image(enc.tobytes())
            self.assertEqual(out.shape[:2], self.img.shape[:2])

    def test_array_and_pil(self):
        self.assertEqual(load_image(self.img).shape, self.img.shape)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(load_image(gray).ndim, 3)
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.assertEqual(load_image(pil).shape, self.img.shape)

    def test_invalid(self):
        with self.assertRaises(InvalidImageError):
            load_image(b"not an image at all")
        with self.assertRaises(InvalidImageError):
            load_image(b"")
        with self.assertRaises(InvalidImageError):
            load_image("/nonexistent/photo.jpg")
        buf = io.BytesIO()
        Image.fromarray(np.zeros((20, 20), np.uint8)).save(buf, format="GIF")
        with self.assertRaises(InvalidImageError):
            load_image(buf.getvalue())

    def test_exif_orientation(self):
        pil = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        exif = pil.getexif()
        exif[0x0112] = 6  # rotate 90 CW on display
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", exif=exif)
        out = load_image(buf.getvalue())
        self.assertEqual(out.shape[:2], self.img.shape[:2][::-1])

    def test_working_scale(self):
        big = cv2.resize(self.img, None, fx=8, fy=8)
        g, s = to_working(big)
        self.assertLessEqual(max(g.shape), 1800)
        self.assertAlmostEqual(g.shape[1] / big.shape[1], s, places=2)


class TestNormalisation(unittest.TestCase):
    def test_blank_crop_has_no_ink(self):
        self.assertEqual(ink_map(np.full((50, 50), 200, np.uint8)).max(), 0)

    def test_dots_survive(self):
        g = np.full((80, 80), 230, np.uint8)
        for x in (25, 40, 55):
            cv2.circle(g, (x, 40), 2, 30, -1)
        cv2.rectangle(g, (5, 5), (75, 75), 30, 2)
        n = normalize_crop(g)
        self.assertEqual(n.shape, (64, 64))
        b = (n > 0.5).astype(np.uint8)
        k, *_ = cv2.connectedComponentsWithStats(b)
        self.assertGreaterEqual(k - 1, 4)  # frame + 3 dots

    def test_shadow_invariance(self):
        g = np.full((100, 100), 220, np.float32)
        cv2.circle(g, (50, 50), 30, 30, 3)
        sh = g.copy()
        sh[:, :50] *= 0.5  # hard shadow over half the symbol
        a, b = ink_map(g.astype(np.uint8)), ink_map(sh.astype(np.uint8))
        self.assertLess(np.abs(a - b).mean(), 0.05)

    def test_aspect_preserved(self):
        g = np.full((40, 120), 230, np.uint8)
        cv2.rectangle(g, (5, 5), (115, 35), 20, 2)
        n = normalize_crop(g)
        ys, xs = np.where(n > 0.5)
        self.assertGreater((xs.max() - xs.min()) / max(1, ys.max() - ys.min()), 2.0)

    def test_features_dim(self):
        self.assertEqual(features(np.zeros((64, 64), np.float32)).shape, (2344,))

    def test_polarity(self):
        img, _ = clean_label(["iron_low"])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.assertTrue(ink_is_dark(g))
        self.assertFalse(ink_is_dark(255 - g))


class TestQuality(unittest.TestCase):
    def test_good_and_bad(self):
        img, _ = clean_label(["wash_40", "iron_low"])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.assertFalse(assess(g).too_low)
        self.assertTrue(assess(np.full((300, 300), 128, np.uint8)).too_low)       # no contrast
        self.assertTrue(assess(g[:40, :40]).too_low)                               # tiny
        blurred = cv2.GaussianBlur(g, (0, 0), 12)
        self.assertTrue(assess(blurred).too_low)


if __name__ == "__main__":
    unittest.main()
