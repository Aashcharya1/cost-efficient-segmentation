"""Independent oracle tests for scoring, boundary rules and prediction contract."""

import unittest
from src import runtime

runtime.configure()
import numpy as np
from src.utils import Metrics, compute_boundary_f1, ces_score, _boundary_map


class MetricsTests(unittest.TestCase):
    def test_ces(self):
        self.assertAlmostEqual(ces_score(0.7, 20, 524288), 0.63525)
        self.assertEqual(ces_score(1, 50.001, 0), 0)

    def test_confusion(self):
        m = Metrics()
        m.update(np.array([[0, 1], [1, 0]]), np.array([[0, 0], [1, 1]]))
        self.assertAlmostEqual(m.result()["miou"], 1 / 3)

    def test_empty_full_and_borders(self):
        z = np.zeros((9, 9), np.uint8)
        o = np.ones_like(z)
        self.assertEqual(compute_boundary_f1(z, z), 1)
        self.assertEqual(compute_boundary_f1(z, o), 0)
        self.assertEqual(int(_boundary_map(o).sum()), 32)
        self.assertEqual(compute_boundary_f1(o, o), 1)

    def test_boundary_against_slow_oracle(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            p = rng.random((11, 13)) > 0.7
            g = rng.random((11, 13)) > 0.7

            def boundary(m):
                h, w = m.shape
                return {
                    (y, x)
                    for y in range(h)
                    for x in range(w)
                    if m[y, x]
                    and any(
                        not (0 <= yy < h and 0 <= xx < w) or not m[yy, xx]
                        for yy, xx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                    )
                }

            pb, gb = boundary(p), boundary(g)
            precision = sum(
                any(abs(y - v) <= 2 and abs(x - u) <= 2 for v, u in gb) for y, x in pb
            ) / len(pb)
            recall = sum(
                any(abs(y - v) <= 2 and abs(x - u) <= 2 for v, u in pb) for y, x in gb
            ) / len(gb)
            expected = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0
            )
            self.assertAlmostEqual(compute_boundary_f1(p, g), expected)


if __name__ == "__main__":
    unittest.main()
