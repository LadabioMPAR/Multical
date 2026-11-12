"""
Numerical regression test: the core must reproduce tests/golden/reference.npz.

Run from the repository root:

    python -m unittest discover -s tests -t .
"""
import os
import unittest

import numpy as np

from ._cases import compute_all

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "reference.npz")

RTOL, ATOL = 1e-7, 1e-12
RTOL_SVR = 1e-5  # libsvm results vary slightly between scikit-learn builds


class GoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_PATH):
            raise unittest.SkipTest("reference.npz missing; run: python -m tests.make_golden")
        with np.load(GOLDEN_PATH) as data:
            cls.golden = {k: data[k] for k in data.files}
        cls.actual = compute_all()

    def test_same_set_of_results(self):
        self.assertEqual(sorted(self.actual), sorted(self.golden))

    def _check_prefix(self, prefix, rtol=RTOL):
        keys = [k for k in self.golden if k.startswith(prefix)]
        self.assertTrue(keys, f"no reference arrays for prefix {prefix!r}")
        for key in keys:
            with self.subTest(key=key):
                expected, actual = self.golden[key], np.asarray(self.actual[key])
                self.assertEqual(actual.shape, expected.shape)
                if np.issubdtype(expected.dtype, np.integer):
                    np.testing.assert_array_equal(actual, expected, err_msg=key)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=ATOL, err_msg=key)

    def test_synthetic_data(self):
        self._check_prefix("data_")

    def test_pretreatments(self):
        self._check_prefix("pre_")

    def test_engine_cross_validation(self):
        self._check_prefix("engine_")

    def test_pls(self):
        self._check_prefix("pls_")

    def test_pcr(self):
        self._check_prefix("pcr_")

    def test_spa(self):
        self._check_prefix("spa_")

    def test_outlier_metrics(self):
        self._check_prefix("outlier_")

    def test_vip(self):
        self._check_prefix("vip_")

    def test_fast_rmsecv_and_ftest(self):
        self._check_prefix("rmsecv_fast_")

    def test_saved_pls_model(self):
        self._check_prefix("saved_pls_")

    def test_svr(self):
        self._check_prefix("svr_", rtol=RTOL_SVR)


if __name__ == "__main__":
    unittest.main()
