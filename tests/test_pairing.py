"""Tests for linking measured values (time in minutes) to the spectra of a time series (seconds)."""
import os
import sys
import tempfile
import unittest

import numpy as np

from ._spp_writer import write_spp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api  # noqa: E402
from multical.io.reference import read_reference_table  # noqa: E402
from multical.io.spp import read_spp, read_spp_with_reference  # noqa: E402

# One spectrum every 120 s, the first one 2 s after the start: 0.03, 2.03, ... 98.03 min.
TIMES_S = 2.0 + 120.0 * np.arange(50)


def _series(points=40, seed=4):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(TIMES_S.size, points))
    return api.Dataset(X, np.arange(points, dtype=float), times=TIMES_S, groups=np.array(["exp1"] * TIMES_S.size))


class MatchTimes(unittest.TestCase):
    def test_exact_takes_the_closest_spectrum(self):
        match = api.match_times(TIMES_S, [0, 30, 31, 60, 98])
        self.assertEqual([idx.tolist() for idx in match.indices], [[0], [15], [15], [30], [49]])
        self.assertTrue(match.matched.all())
        self.assertEqual(match.problems(), [])
        np.testing.assert_allclose(match.distance_min, [2 / 60, 2 / 60, 58 / 60, 2 / 60, 2 / 60])

    def test_exact_leaves_out_samples_outside_the_recorded_period(self):
        match = api.match_times(TIMES_S, [30, 200])
        self.assertEqual(match.matched.tolist(), [True, False])
        self.assertAlmostEqual(match.limit_min, 2.0)
        self.assertEqual(len(match.problems()), 1)
        self.assertIn("200 min", match.problems()[0])
        wide = api.match_times(TIMES_S, [30, 200], max_distance_min=150)
        self.assertEqual(wide.indices[1].tolist(), [49])

    def test_window_takes_every_spectrum_within_the_radius(self):
        match = api.match_times(TIMES_S, [30, 31, 0], mode="window", radius_min=3)
        self.assertEqual([idx.tolist() for idx in match.indices], [[14, 15, 16], [14, 15, 16], [0, 1]])
        narrow = api.match_times(TIMES_S, [30, 31], mode="window", radius_min=0.5)
        self.assertEqual(narrow.matched.tolist(), [True, False])
        self.assertIn("nenhum espectro a até 0.5 min", narrow.problems()[0])

    def test_invalid_requests(self):
        for kwargs in ({"mode": "window"}, {"mode": "window", "radius_min": 0}, {"mode": "media"}):
            with self.assertRaises(ValueError):
                api.match_times(TIMES_S, [30], **kwargs)
        with self.assertRaises(ValueError):
            api.match_times([], [30])
        with self.assertRaises(ValueError):
            api.match_times(TIMES_S, [])
        self.assertEqual(sorted(api.MATCH_MODES), ["exact", "window"])


class PairWithReference(unittest.TestCase):
    def setUp(self):
        self.series = _series()
        self.ref_times = np.array([30.0, 60.0, 200.0, 90.0])
        self.Y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

    def test_exact(self):
        data, match = api.pair_with_reference(self.series, self.ref_times, self.Y, analytes=["gl", "xy"], units="g/L")
        np.testing.assert_array_equal(data.X, self.series.X[[15, 30, 45]])
        np.testing.assert_array_equal(data.Y, self.Y[[0, 1, 3]])       # the 200 min sample has no spectrum
        np.testing.assert_array_equal(data.times, [30.0, 60.0, 90.0])
        self.assertEqual(data.sample_ids, ["exp1_30min", "exp1_60min", "exp1_90min"])
        self.assertEqual((data.analytes, data.units), (["gl", "xy"], "g/L"))
        self.assertEqual(match.matched.tolist(), [True, True, False, True])

    def test_window_averages(self):
        data, _ = api.pair_with_reference(self.series, self.ref_times, self.Y, mode="window", radius_min=3)
        expected = np.vstack([self.series.X[[14, 15, 16]].mean(axis=0), self.series.X[[29, 30, 31]].mean(axis=0),
                              self.series.X[[44, 45, 46]].mean(axis=0)])
        np.testing.assert_allclose(data.X, expected, rtol=1e-12)

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "Nenhuma amostra"):
            api.pair_with_reference(self.series, [500.0], [[1.0]])
        with self.assertRaisesRegex(ValueError, "linhas de valores"):
            api.pair_with_reference(self.series, self.ref_times, self.Y[:3])
        no_times = api.Dataset(self.series.X, self.series.wavelengths)
        with self.assertRaisesRegex(ValueError, "não têm tempo"):
            api.pair_with_reference(no_times, self.ref_times, self.Y)
        two = api.Dataset.concat([self.series, _series(seed=9).subset(np.arange(5))])
        two.groups[-5:] = "exp2"
        with self.assertRaisesRegex(ValueError, "um experimento por vez"):
            api.pair_with_reference(two, self.ref_times, self.Y)

    def test_reading_from_the_file_gives_the_same_as_pairing_in_memory(self):
        stored = self.series.X[:, ::-1]       # as the instrument stores it, highest wavenumber first
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "exp1.spp")
            write_spp(path, stored, TIMES_S, start=39.0, step=-1.0, allocated=80)
            whole = read_spp(path)
            for kwargs in ({}, {"mode": "window", "radius_min": 5}):
                direct, match = read_spp_with_reference(path, self.ref_times, self.Y, analytes=["gl", "xy"], **kwargs)
                memory, _ = api.pair_with_reference(whole, self.ref_times, self.Y, analytes=["gl", "xy"], **kwargs)
                np.testing.assert_allclose(direct.X, memory.X, rtol=1e-12)
                np.testing.assert_array_equal(direct.Y, memory.Y)
                self.assertEqual(direct.sample_ids, memory.sample_ids)
                self.assertEqual(match.matched.tolist(), [True, True, False, True])

    def test_experiments_are_joined_for_calibration(self):
        first, _ = api.pair_with_reference(self.series, self.ref_times, self.Y, analytes=["gl", "xy"])
        other = _series(seed=8)
        other.groups[:] = "exp2"
        second, _ = api.pair_with_reference(other, [30.0, 60.0], self.Y[:2], analytes=["gl", "xy"])
        joined = api.Dataset.concat([first, second])
        self.assertEqual(joined.X.shape, (5, 40))
        self.assertEqual(joined.groups.tolist(), ["exp1"] * 3 + ["exp2"] * 2)
        self.assertEqual(joined.sample_ids[-1], "exp2_60min")
        mismatched, _ = api.pair_with_reference(other, [30.0], [[1.0]], analytes=["gl"])
        with self.assertRaisesRegex(ValueError, "mesmos analitos"):
            api.Dataset.concat([first, mismatched])


class ReferenceTable(unittest.TestCase):
    def _read(self, text, name="refe.txt", encoding="utf-8"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            with open(path, "w", encoding=encoding, newline="") as handle:
                handle.write(text)
            return read_reference_table(path)

    def test_current_text_format(self):
        times, Y, names = self._read("30\t1.586\t7.735\t4.477\n60\t1.54\t8.354\t4.756\n120\t1.585\t9.095\t5.157")
        np.testing.assert_array_equal(times, [30, 60, 120])
        np.testing.assert_array_equal(Y, [[1.586, 7.735, 4.477], [1.54, 8.354, 4.756], [1.585, 9.095, 5.157]])
        self.assertIsNone(names)

    def test_spreadsheet_exports(self):
        times, Y, names = self._read("tempo;glicose;xilose\r\n30;7,735;4,477\r\n60;8,354;4,756\r\n", "refe.csv", "latin-1")
        np.testing.assert_array_equal(times, [30, 60])
        np.testing.assert_array_equal(Y, [[7.735, 4.477], [8.354, 4.756]])
        self.assertEqual(names, ["glicose", "xilose"])
        times, Y, names = self._read("time,gl\n# comentário\n\n30,7.735\n60,8.354\n", "refe.csv")
        np.testing.assert_array_equal(Y, [[7.735], [8.354]])
        self.assertEqual(names, ["gl"])

    def test_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "linha 2"):
            self._read("30\t1.5\n60\tn.d.\n")
        with self.assertRaisesRegex(ValueError, "linha 2"):
            self._read("30\t1.5\t2.5\n60\t1.6\n")
        with self.assertRaisesRegex(ValueError, "duas colunas"):
            self._read("30\n60\n")
        with self.assertRaisesRegex(ValueError, "vazio"):
            self._read("\n\n")
        with self.assertRaisesRegex(ValueError, "CSV"):
            self._read("x", "refe.xlsx")
        with self.assertRaises(FileNotFoundError):
            read_reference_table(os.path.join(tempfile.gettempdir(), "nao_existe_refe.txt"))


if __name__ == "__main__":
    unittest.main()
