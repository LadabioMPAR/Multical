"""Tests for the model file, the plain-language assessment and the export folder."""
import os
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from ._cases import ANALYTES, KMAX, PRETREATMENT, make_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api  # noqa: E402
from multical.api import export  # noqa: E402


class ExportCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wl, X, Y = make_dataset()
        cls.dataset = api.Dataset(X, wl, Y, analytes=list(ANALYTES), units="g/L")
        recipe = api.recipe_from_legacy(PRETREATMENT) + [{"id": "msc", "params": {}}, {"id": "autoscale", "params": {}}]
        cls.result = api.run_calibration(cls.dataset, api.RunConfig(
            pretreatment=recipe, model={"id": "pls", "params": {"max_k": KMAX}}))

    def test_saved_model_predicts_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = api.save_model(self.result, os.path.join(tmp, "modelo"))
            self.assertTrue(path.endswith(".multical"))
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(sorted(archive.namelist()), ["arrays.npz", "manifest.json"])
            model = api.load_model(path)
        self.assertEqual((model.analytes, model.units, model.n_samples), (list(ANALYTES), "g/L", 30))
        self.assertEqual(model.per_analyte["gl"].k, self.result.per_analyte["gl"].best_k)
        new = self.dataset.X[:7] * 1.01
        np.testing.assert_allclose(api.predict(model, new, self.dataset.wavelengths),
                                   api.predict(self.result, new, self.dataset.wavelengths), rtol=1e-12)
        with self.assertRaises(ValueError):
            api.predict(model, new[:, :-3], self.dataset.wavelengths[:-3])

    def test_invalid_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.multical")
            with open(path, "w") as handle:
                handle.write("nada")
            with self.assertRaisesRegex(ValueError, "não é um modelo"):
                api.load_model(path)

    def test_assessment(self):
        items = api.assess(self.result)
        self.assertEqual([i.analyte for i in items], list(ANALYTES))
        for item in items:
            self.assertIn(item.level, ("bom", "atencao", "ruim"))
            self.assertIn("erro típico ±", item.summary)
            self.assertIn("g/L", item.summary)
            self.assertNotIn(".", item.summary.split("±")[1].split(" ")[0])     # decimal comma
        self.assertEqual(api.dataset_warnings(self.dataset), [])
        self.assertEqual(len(api.dataset_warnings(self.dataset.subset(np.arange(8)))), 1)

    def test_export_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "execucao 1")
            written = export.export_run(self.result, folder)
            names = sorted(os.path.relpath(p, folder).replace("\\", "/") for p in written)
            self.assertEqual(names, sorted(
                ["modelo.multical", "resumo.csv", "metricas_por_variaveis_latentes.csv", "predicoes.csv", "config.json",
                 "relatorio.pdf"]
                + [f"figuras/previsto_x_medido_{a}.png" for a in ANALYTES]
                + [f"figuras/erro_por_variaveis_latentes_{a}.png" for a in ANALYTES]))
            for path in written:
                self.assertGreater(os.path.getsize(path), 0)
            with open(os.path.join(folder, "resumo.csv"), encoding="utf-8-sig") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(len(lines), 1 + len(ANALYTES))
            self.assertTrue(lines[1].startswith("cb;g/L;30;"))
            self.assertIn(",", lines[1].split(";")[4])
            with open(os.path.join(folder, "predicoes.csv"), encoding="utf-8-sig") as handle:
                self.assertEqual(len(handle.read().splitlines()), 31)
            again = api.RunConfig.from_json(open(os.path.join(folder, "config.json"), encoding="utf-8").read())
            self.assertEqual(again.to_dict(), self.result.config.to_dict())
            only = export.export_run(self.result, os.path.join(tmp, "b"), items=["config"])
            self.assertEqual([os.path.basename(p) for p in only], ["config.json"])
            with self.assertRaises(ValueError):
                export.export_run(self.result, os.path.join(tmp, "c"), items=["planilha"])


if __name__ == "__main__":
    unittest.main()
