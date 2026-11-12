"""
Tests for the .spp reader, on small synthetic files written by `_spp_writer`.

Set the environment variable MULTICAL_SPP_SAMPLES to a folder with real .spp
files to also run the reader over them.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

from ._spp_writer import write_compound_file, write_spp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api
from multical.io.spp import read_spp, read_spp_files, spp_info

SAMPLES = os.environ.get("MULTICAL_SPP_SAMPLES", "")


def _spectra(n=6, points=700, seed=3):
    """Rows as the instrument stores them: first column at the highest wavenumber."""
    rng = np.random.RandomState(seed)
    return rng.normal(size=(n, points)), 2.0 + 120.0 * np.arange(n)


class SppReader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _path(self, name="exp.spp", folder=None):
        folder = self.tmp if folder is None else os.path.join(self.tmp, folder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, name)

    def test_reads_only_the_collected_spectra_in_increasing_wavenumber(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times, start=10000.0, step=-1.0, allocated=40)
        data = read_spp(path)
        self.assertEqual(data.X.shape, (6, 700))
        np.testing.assert_array_equal(data.wavelengths, np.arange(9301.0, 10001.0))
        np.testing.assert_array_equal(data.X, stored[:, ::-1])
        np.testing.assert_array_equal(data.times, times)
        self.assertIsNone(data.Y)
        self.assertEqual(data.sample_ids[0], "exp_1")
        self.assertEqual(list(np.unique(data.groups)), ["exp"])

    def test_increasing_axis_is_kept_as_stored(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times, start=400.0, step=2.0)
        data = read_spp(path)
        np.testing.assert_array_equal(data.wavelengths, 400.0 + 2.0 * np.arange(700))
        np.testing.assert_array_equal(data.X, stored)

    def test_fragmented_file_reads_the_same(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times, reverse_data=True)
        np.testing.assert_array_equal(read_spp(path).X, stored[:, ::-1])

    def test_large_file_with_extended_allocation_table(self):
        # More than 109 allocation sectors (about 7 MB), as in every real file.
        rng = np.random.RandomState(0)
        stored = rng.normal(size=(125, 8000))
        path = self._path()
        write_spp(path, stored, 120.0 * np.arange(125), allocated=126)
        self.assertGreater(os.path.getsize(path), 109 * 128 * 512)
        data = read_spp(path)
        np.testing.assert_array_equal(data.X, stored[:, ::-1])
        np.testing.assert_array_equal(read_spp(path, indices=[124, 0]).X, stored[[124, 0], ::-1])

    def test_wrapped_clock_is_corrected(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times - 2 ** 32 / 1000.0)
        np.testing.assert_allclose(read_spp(path).times, times, atol=1e-6)

    def test_partial_read(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times, allocated=10)
        part = read_spp(path, indices=[4, 1])
        np.testing.assert_array_equal(part.X, stored[[4, 1], ::-1])
        np.testing.assert_array_equal(part.times, times[[4, 1]])
        self.assertEqual(part.sample_ids, ["exp_5", "exp_2"])
        with self.assertRaises(ValueError):
            read_spp(path, indices=[6])     # allocated but never collected

    def test_info_does_not_need_the_spectra(self):
        stored, times = _spectra()
        path = self._path()
        write_spp(path, stored, times, allocated=40)
        info = spp_info(path)
        self.assertEqual((info["n_spectra"], info["n_allocated"], info["n_points"]), (6, 40, 700))
        self.assertEqual((info["x_first"], info["x_last"], info["x_step"]), (9301.0, 10000.0, 1.0))
        self.assertEqual((info["x_unit"], info["y_unit"], info["time_unit"]), ("cm-1", "Abs", "Secs"))
        np.testing.assert_array_equal(info["times"], times)

    def test_file_without_collected_spectra(self):
        stored, times = _spectra()
        never_started = self._path("a.spp")
        write_spp(never_started, np.zeros((4, 700)), np.zeros(4), start=0.0, step=0.0, units=("", "", ""))
        empty = self._path("b.spp")
        write_spp(empty, stored, times, collected=0)
        for path in (never_started, empty):
            self.assertEqual(spp_info(path)["n_spectra"], 0)
            with self.assertRaisesRegex(ValueError, "não contém espectros"):
                read_spp(path)

    def test_invalid_files_give_clear_errors(self):
        text = self._path("texto.spp")
        with open(text, "w") as handle:
            handle.write("Time\t1200\t1201\n" * 100)
        with self.assertRaisesRegex(ValueError, "não é um arquivo .spp válido"):
            read_spp(text)

        other = self._path("outro.spp")
        write_compound_file(other, {"Qualquer": b"x" * 5000})
        with self.assertRaisesRegex(ValueError, "não pôde ser lido"):
            read_spp(other)

        stored, times = _spectra()
        whole = self._path("inteiro.spp")
        write_spp(whole, stored, times)
        cut = self._path("cortado.spp")
        with open(whole, "rb") as src, open(cut, "wb") as dst:
            dst.write(src.read(os.path.getsize(whole) // 2))
        with self.assertRaises(ValueError):
            read_spp(cut)

        with self.assertRaises(FileNotFoundError):
            read_spp(self._path("nao_existe.spp"))

    def test_several_files(self):
        a, ta = _spectra(seed=1)
        b, tb = _spectra(n=4, seed=2)
        first, second = self._path("Untitled.spp", "exp 1"), self._path("Untitled.spp", "exp 2")
        write_spp(first, a, ta)
        write_spp(second, b, tb)
        data = api.IMPORTERS.get("spp").read([first, second])
        self.assertEqual(data.X.shape, (10, 700))
        np.testing.assert_array_equal(data.X, np.vstack([a, b])[:, ::-1])
        self.assertEqual(list(np.unique(data.groups)), ["exp 1/Untitled", "exp 2/Untitled"])
        self.assertEqual(data.sample_ids[6], "exp 2/Untitled_1")

        other_grid = self._path("outra_faixa.spp")
        write_spp(other_grid, a[:, :650], ta)
        with self.assertRaisesRegex(ValueError, "faixa espectral diferente"):
            read_spp_files([first, other_grid])

    def test_spectra_from_a_file_can_be_calibrated(self):
        rng = np.random.RandomState(5)
        conc = rng.uniform(0, 10, size=24)
        x = np.linspace(0, 1, 300)
        stored = np.outer(conc, np.exp(-((x - 0.4) / 0.08) ** 2)) + 0.01 * rng.normal(size=(24, 300))
        path = self._path()
        write_spp(path, stored, 120.0 * np.arange(24))
        spectra = read_spp(path)
        data = api.Dataset(spectra.X, spectra.wavelengths, conc, analytes=["gl"], sample_ids=spectra.sample_ids)
        result = api.run_calibration(data)
        self.assertEqual(result.config.pretreatment, [{"id": "sg", "params": {"radius": 7, "poly_order": 2, "deriv": 1}}])
        self.assertGreater(result.per_analyte["gl"].best_r2cv, 0.9)
        np.testing.assert_allclose(api.predict(result, spectra.X, spectra.wavelengths)[:, 0], conc, atol=1.0)


@unittest.skipUnless(os.path.isdir(SAMPLES), "defina MULTICAL_SPP_SAMPLES para testar com arquivos reais")
class RealSamples(unittest.TestCase):
    def test_every_sample_file(self):
        read = 0
        for folder, _, names in os.walk(SAMPLES):
            for name in names:
                if not name.lower().endswith(".spp"):
                    continue
                path = os.path.join(folder, name)
                info = spp_info(path)
                if info["n_spectra"] == 0:
                    continue
                data = read_spp(path)
                read += 1
                self.assertEqual(data.X.shape, (info["n_spectra"], info["n_points"]))
                self.assertTrue(np.isfinite(data.X).all())
                self.assertTrue(np.all(np.abs(data.X).max(axis=1) > 0))
                self.assertTrue(np.all(np.diff(data.wavelengths) > 0))
                self.assertTrue(np.all(np.diff(data.times) > 0))
                self.assertGreaterEqual(data.times[0], 0.0)
        self.assertGreater(read, 0)


if __name__ == "__main__":
    unittest.main()
