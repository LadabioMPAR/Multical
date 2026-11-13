"""
Tests for the methods added through the registries: PCR, SPA and SVR models,
validation by experiment and Kennard-Stone, and plugin files.

Equivalence tests compare with the existing numerical code in `multical.models`.
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ._cases import ANALYTES, make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from multical import api  # noqa: E402
from multical.api import export  # noqa: E402
from multical.api.splitters import kennard_stone  # noqa: E402
from multical.models.pcr import pcr_model  # noqa: E402
from multical.models.spa import spa_model  # noqa: E402
from multical.models.svr import SVRModel as LegacySVR  # noqa: E402

KMAX = 6


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class MethodsCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, cls.X, cls.Y = make_dataset()
        groups = np.array(["exp1"] * 10 + ["exp2"] * 12 + ["exp3"] * 8)
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L", groups=groups)
        cls.folds = list(api.SPLITTERS.get("kfold").split(30, folds=5, order="venetian"))

    def _config(self, model, params=None, validation="kfold", **extra):
        return api.RunConfig(pretreatment=[], model={"id": model, "params": params or {}},
                             validation={"id": validation, "params": extra.pop("validation_params", {})}, **extra)


class LegacyEquivalence(MethodsCase):
    def test_pcr_matches_pcr_model(self):
        result = api.run_calibration(self.dataset, self._config("pcr", {"max_k": KMAX}))
        for j, name in enumerate(ANALYTES):
            y = self.Y[:, j]
            expected = np.empty((KMAX, 30))
            for train, val in self.folds:
                for k in range(1, KMAX + 1):
                    expected[k - 1, val] = pcr_model(self.X[train], y[train], k, Xt=self.X[val], teste_switch=1)[1].ravel()
            np.testing.assert_allclose(result.per_analyte[name].y_cv, expected, rtol=1e-7, atol=1e-9)
            final = result.per_analyte[name]
            whole = pcr_model(self.X, y, final.best_k, teste_switch=0)[0].ravel()
            np.testing.assert_allclose(final.y_cal[final.best_index], whole, rtol=1e-7, atol=1e-9)

    def test_spa_matches_spa_model(self):
        for cini in (0, 40):
            result = api.run_calibration(self.dataset, self._config("spa", {"max_k": KMAX, "cini": cini}))
            for j, name in enumerate(ANALYTES):
                y = self.Y[:, j]
                expected = np.empty((KMAX, 30))
                for train, val in self.folds:
                    for k in range(1, KMAX + 1):
                        expected[k - 1, val] = spa_model(self.X[train], y[train], k, cini, Xt=self.X[val], teste_switch=0)[1].ravel()
                np.testing.assert_allclose(result.per_analyte[name].y_cv, expected, rtol=1e-7, atol=1e-9)
                final = result.per_analyte[name]
                _, _, legacy = spa_model(self.X, y, final.best_k, cini, teste_switch=0)
                np.testing.assert_array_equal(final.model_state["selected"], legacy["SelectedVars"])
        with self.assertRaisesRegex(ValueError, "além do fim do espectro"):
            api.run_calibration(self.dataset, self._config("spa", {"max_k": 3, "cini": 5000}))

    def test_svr_matches_the_scikit_learn_wrapper(self):
        for kernel, C, epsilon in (("rbf", 10.0, 0.05), ("linear", 1.0, 0.1)):
            params = {"tune": False, "kernel": kernel, "C": C, "epsilon": epsilon, "gamma_factor": 1.0}
            result = api.run_calibration(self.dataset, self._config("svr", params))
            for j, name in enumerate(ANALYTES):
                y = self.Y[:, j]
                expected = np.empty(30)
                for train, val in self.folds:
                    expected[val] = LegacySVR().fit_predict(self.X[train], y[train], self.X[val], C=C, epsilon=epsilon,
                                                            gamma="scale", kernel=kernel).ravel()
                np.testing.assert_allclose(result.per_analyte[name].y_cv[0], expected, rtol=1e-6, atol=1e-8)
                self.assertIsNone(result.per_analyte[name].best_k)
                self.assertIsNone(result.per_analyte[name].best_label)


class SettingsTriedByValidation(MethodsCase):
    def test_svr_keeps_the_setting_with_the_lowest_validation_error(self):
        result = api.run_calibration(self.dataset, self._config("svr", analytes=["gl"]))
        res = result.per_analyte["gl"]
        self.assertEqual(len(res.labels), 24)                      # 4 x 2 x 3 combinations
        self.assertEqual(res.y_cv.shape, (24, 30))
        self.assertEqual(res.best_index, int(np.argmin(res.rmsecv)))
        self.assertTrue(res.best_label.startswith("C="))
        self.assertIsNone(res.best_k)
        # the chosen setting is the one used in the final model
        chosen = dict(part.split("=") for part in res.best_label.replace("largura", "gamma_factor").split("; "))
        params = {"tune": False, "kernel": "rbf", **{key: float(value.replace(",", ".")) for key, value in chosen.items()}}
        fixed = api.run_calibration(self.dataset, self._config("svr", params, analytes=["gl"]))
        np.testing.assert_allclose(api.predict(result, self.X, self.wl), api.predict(fixed, self.X, self.wl), rtol=1e-10)
        np.testing.assert_allclose(fixed.per_analyte["gl"].rmsecv[0], res.best_rmsecv, rtol=1e-10)

        item = api.assess(result)[0]
        self.assertIn("ajuste escolhido: C=", item.summary)
        headers, rows = api.assessment.candidate_table(result, "gl")
        self.assertEqual((headers[0], len(rows)), ("Ajuste testado", 24))
        self.assertEqual(sum("(escolhido)" in row[0] for row in rows), 1)
        linear = api.run_calibration(self.dataset, self._config("svr", {"kernel": "linear"}, analytes=["gl"]))
        self.assertEqual(len(linear.per_analyte["gl"].labels), 8)   # the width does not apply to a straight line

    def test_wording_follows_the_model(self):
        for model, word in (("pls", "variáve"), ("pcr", "componente"), ("spa", "ponto")):
            result = api.run_calibration(self.dataset, self._config(model, {"max_k": 4}, analytes=["cb"]))
            self.assertIn(word, api.assess(result)[0].summary)
            self.assertIn(word, api.assessment.candidate_table(result, "cb")[0][0].lower())


class EveryRegisteredMethod(MethodsCase):
    def test_every_model_runs_saves_and_predicts(self):
        new = self.X[:5] * 1.01
        for model in api.MODELS:
            params = {"max_k": 4} if model.sweep else {}
            with self.subTest(model=model.id):
                result = api.run_calibration(self.dataset, self._config(model.id, params))
                self.assertEqual(sorted(result.per_analyte), sorted(ANALYTES))
                for res in result.per_analyte.values():
                    self.assertEqual(res.y_cv.shape[1], 30)
                    self.assertTrue(np.all(np.isfinite(res.rmsecv)))
                with tempfile.TemporaryDirectory() as tmp:
                    saved = api.load_model(api.save_model(result, os.path.join(tmp, model.id)))
                    np.testing.assert_allclose(api.predict(saved, new, self.wl), api.predict(result, new, self.wl), rtol=1e-10)
                    written = export.export_run(result, os.path.join(tmp, "saida"))
                    self.assertTrue(any(p.endswith("metricas_por_ajuste.csv") for p in written))

    def test_every_validation_scheme_runs(self):
        for splitter in api.SPLITTERS:
            with self.subTest(validation=splitter.id):
                result = api.run_calibration(self.dataset, self._config("pls", {"max_k": 3}, validation=splitter.id, analytes=["gl"]))
                self.assertTrue(result.validated.any())
                self.assertEqual(result.validated.all(), splitter.kind == "cv")


class ValidationSchemes(MethodsCase):
    def test_leave_one_experiment_out(self):
        folds = list(api.SPLITTERS.get("loexo").split(30, groups=self.dataset.groups))
        self.assertEqual([len(val) for _, val in folds], [10, 12, 8])
        for train, val in folds:
            self.assertEqual(len(set(self.dataset.groups[val])), 1)
            self.assertFalse(set(self.dataset.groups[val]) & set(self.dataset.groups[train]))
        result = api.run_calibration(self.dataset, self._config("pls", {"max_k": 3}, validation="loexo"))
        self.assertTrue(result.validated.all())
        one = api.Dataset(self.X, self.wl, self.Y, analytes=list(ANALYTES), groups=np.array(["a"] * 30))
        with self.assertRaisesRegex(ValueError, "pelo menos 2 experimentos"):
            api.run_calibration(one, self._config("pls", {"max_k": 3}, validation="loexo"))
        none = api.Dataset(self.X, self.wl, self.Y, analytes=list(ANALYTES))
        with self.assertRaisesRegex(ValueError, "experimento"):
            api.run_calibration(none, self._config("pls", {"max_k": 3}, validation="loexo"))

    def test_kennard_stone_follows_the_maximin_rule(self):
        rng = np.random.RandomState(1)
        points = rng.normal(size=(25, 4))
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        chosen = [int(i), int(j)]
        while len(chosen) < 15:                                    # straightforward version of the rule
            rest = [r for r in range(25) if r not in chosen]
            chosen.append(max(rest, key=lambda r: distance[r, chosen].min()))
        np.testing.assert_array_equal(kennard_stone(points, 15), sorted(chosen))
        self.assertFalse(np.array_equal(kennard_stone(points, 15, Y=rng.normal(size=25)), sorted(chosen)))

        (train, val), = api.SPLITTERS.get("ks").split(30, X=self.X, Y=self.Y, fraction=0.2)
        self.assertEqual((len(train), len(val)), (24, 6))
        self.assertEqual(sorted(np.concatenate([train, val])), list(range(30)))
        (again, _), = api.SPLITTERS.get("ks").split(30, X=self.X, Y=self.Y, fraction=0.2)
        np.testing.assert_array_equal(train, again)               # no drawing: always the same split
        result = api.run_calibration(self.dataset, self._config("pls", {"max_k": 3}, validation="ks",
                                                                validation_params={"method": "spxy", "fraction": 0.3}))
        self.assertEqual(int(result.validated.sum()), 9)


class Plugins(unittest.TestCase):
    def tearDown(self):
        for item in ("teste_dobro", "log10"):
            api.PRETREATMENTS.unregister(item)
        api.MODELS.unregister("teste_media")
        del api.PLUGIN_ERRORS[:]

    def test_plugin_files_register_methods_and_failures_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "dobro.py"), "w", encoding="utf-8") as handle:
                handle.write(
                    "import numpy as np\n"
                    "from multical.api import MODELS, Model, Param, pretreatment, register\n\n"
                    "@pretreatment(id='teste_dobro', label='Dobro', params=[Param('fator', 'Fator', 'float', default=2.0)])\n"
                    "def dobro(X, wl, fator):\n"
                    "    return X * fator, wl\n\n"
                    "@register(MODELS)\n"
                    "class Media(Model):\n"
                    "    id = 'teste_media'\n"
                    "    label = 'Média'\n"
                    "    def fit(self, X, y):\n"
                    "        return {'mean': float(np.mean(y))}\n"
                    "    def predict(self, state, X):\n"
                    "        return np.full(X.shape[0], state['mean'])\n")
            with open(os.path.join(tmp, "quebrado.py"), "w", encoding="utf-8") as handle:
                handle.write("raise RuntimeError('plugin com defeito')\n")
            with open(os.path.join(tmp, "_ignorado.py"), "w", encoding="utf-8") as handle:
                handle.write("raise RuntimeError('não deve rodar')\n")

            loaded = api.load_plugins([tmp, os.path.join(tmp, "nao_existe")])
            self.assertEqual([os.path.basename(p) for p in loaded], ["dobro.py"])
            self.assertEqual(len(api.PLUGIN_ERRORS), 1)
            self.assertIn("plugin com defeito", api.PLUGIN_ERRORS[0][1])
            self.assertEqual(api.load_plugins([tmp]), [])            # nothing runs twice

        wl, X, Y = make_dataset()
        data = api.Dataset(X, wl, Y, analytes=list(ANALYTES))
        result = api.run_calibration(data, api.RunConfig(
            pretreatment=[{"id": "teste_dobro", "params": {"fator": 3.0}}], model={"id": "teste_media", "params": {}}))
        np.testing.assert_allclose(result.X_pre, 3.0 * X)
        np.testing.assert_allclose(api.predict(result, X[:2], wl), np.tile(Y.mean(axis=0), (2, 1)))

    def test_example_plugin_of_the_repository(self):
        api.PRETREATMENTS.unregister("log10")
        loaded = api.load_plugins([os.path.join(REPO, "plugins")], reload=True)
        self.assertEqual([os.path.basename(p) for p in loaded], ["exemplo_log10.py"])
        self.assertEqual(api.PLUGIN_ERRORS, [])
        wl, X, _ = make_dataset()
        out, _, _ = api.fit_recipe([{"id": "log10", "params": {}}], X, wl)
        np.testing.assert_allclose(out, np.log10(X - X.min(axis=1, keepdims=True) + 1.0))

    def test_folder_from_the_environment_is_loaded_on_import(self):
        code = "from multical import api; print('log10' in api.PRETREATMENTS, len(api.PLUGIN_ERRORS))"
        for folder, expected in ((os.path.join(REPO, "plugins"), "True 0"), ("", "False 0")):
            env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"), MULTICAL_PLUGINS=folder)
            done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO)
            self.assertEqual(done.stdout.strip(), expected, done.stderr)


TEMPLATES = os.path.join(REPO, "plugins", "exemplos")
TEMPLATE_IDS = {
    "pretreatments": ["ex_um_a_cada_n", "ex_subtrair_media"],
    "models": ["ex_ridge", "ex_vizinhos", "ex_ridge_auto"],
    "splitters": ["ex_partes_por_experimento"],
}


def _forget_templates():
    for registry, ids in ((api.PRETREATMENTS, TEMPLATE_IDS["pretreatments"]), (api.MODELS, TEMPLATE_IDS["models"]),
                          (api.SPLITTERS, TEMPLATE_IDS["splitters"])):
        for item in ids:
            registry.unregister(item)
    del api.PLUGIN_ERRORS[:]


class PluginTemplates(MethodsCase):
    """The files in plugins/exemplos are what the tutorial tells people to copy: each one must really work."""

    def setUp(self):
        _forget_templates()
        self.loaded = api.load_plugins([TEMPLATES], reload=True)
        self.addCleanup(_forget_templates)

    def test_every_template_loads_and_registers_what_the_tutorial_says(self):
        self.assertEqual(sorted(os.path.basename(p) for p in self.loaded),
                         sorted(name for name in os.listdir(TEMPLATES) if name.endswith(".py")))
        self.assertEqual(api.PLUGIN_ERRORS, [])
        for registry, ids in ((api.PRETREATMENTS, TEMPLATE_IDS["pretreatments"]), (api.MODELS, TEMPLATE_IDS["models"]),
                              (api.SPLITTERS, TEMPLATE_IDS["splitters"])):
            for item in ids:
                self.assertIn(item, registry)
        # Nothing in the plugin folder itself except the active example: the templates are not loaded by the program.
        active = [n for n in os.listdir(os.path.join(REPO, "plugins")) if n.endswith(".py") and not n.startswith("_")]
        self.assertEqual(active, ["exemplo_log10.py"])

    def test_template_pretreatments(self):
        new = self.X[:4] * 1.05
        thinned, axis, _ = api.fit_recipe([{"id": "ex_um_a_cada_n", "params": {"passo": 3}}], self.X, self.wl)
        np.testing.assert_array_equal(thinned, self.X[:, ::3])
        np.testing.assert_array_equal(axis, self.wl[::3])

        recipe = [{"id": "ex_subtrair_media", "params": {"dividir": True}}]
        centred, _, states = api.fit_recipe(recipe, self.X, self.wl)
        np.testing.assert_allclose(centred.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(centred.std(axis=0, ddof=1), 1)
        applied, _ = api.apply_recipe(recipe, states, new, self.wl)      # new spectra use the calibration mean
        np.testing.assert_allclose(applied, (new - self.X.mean(axis=0)) / self.X.std(axis=0, ddof=1))

    def test_template_models_train_save_and_predict(self):
        new = self.X[:5] * 1.01
        for model_id, params in (("ex_ridge", {"alfa": 0.5}), ("ex_vizinhos", {"max_k": 5}), ("ex_ridge_auto", {})):
            with self.subTest(model=model_id):
                config = api.RunConfig(pretreatment=[{"id": "ex_subtrair_media", "params": {}}],
                                       model={"id": model_id, "params": params},
                                       validation={"id": "ex_partes_por_experimento", "params": {"partes": 3}}, seed=4)
                result = api.run_calibration(self.dataset, config)
                self.assertTrue(result.validated.all())
                with tempfile.TemporaryDirectory() as tmp:
                    saved = api.load_model(api.save_model(result, os.path.join(tmp, "m")))
                    np.testing.assert_allclose(api.predict(saved, new, self.wl), api.predict(result, new, self.wl), rtol=1e-10)
                again = api.run_calibration(self.dataset, config)          # same seed, same draw
                np.testing.assert_array_equal(again.per_analyte["gl"].rmsecv, result.per_analyte["gl"].rmsecv)
        self.assertEqual(result.per_analyte["gl"].labels, ["alfa=0,01", "alfa=0,1", "alfa=1", "alfa=10", "alfa=100"])

        neighbours = api.run_calibration(self.dataset, api.RunConfig(
            pretreatment=[], model={"id": "ex_vizinhos", "params": {"max_k": 4}}, analytes=["gl"]))
        self.assertIn("vizinho", api.assess(neighbours)[0].summary)
        # the fast path of the template gives what fit + predict give, one complexity at a time
        model = api.MODELS.get("ex_vizinhos")
        train, val = self.folds[0]
        y = self.Y[:, 1]
        slow = np.vstack([model.predict(model.fit(self.X[train], y[train], k=k), self.X[val]) for k in range(1, 5)])
        np.testing.assert_allclose(model.cv_path(self.X[train], y[train], self.X[val], 4), slow)

    def test_template_validation_scheme(self):
        splitter = api.SPLITTERS.get("ex_partes_por_experimento")
        groups = self.dataset.groups
        folds = list(splitter.split(30, groups=groups, rng=np.random.RandomState(1), partes=4))
        self.assertEqual(len(folds), 4)
        self.assertEqual(sorted(np.concatenate([val for _, val in folds])), list(range(30)))
        for train, val in folds:
            self.assertFalse(set(train) & set(val))
            for name in ("exp1", "exp2", "exp3"):                          # every experiment has samples on both sides
                self.assertTrue((groups[val] == name).any() and (groups[train] == name).any())
        other = list(splitter.split(30, groups=groups, rng=np.random.RandomState(2), partes=4))
        self.assertFalse(all(np.array_equal(a[1], b[1]) for a, b in zip(folds, other)))

    def test_templates_pass_the_self_check(self):
        from multical_gui import selftest
        pretreatments, models = selftest._check_pretreatments(), selftest._check_models()
        for item in TEMPLATE_IDS["pretreatments"]:
            self.assertIn(item, pretreatments.split(" (")[0])
        for item in TEMPLATE_IDS["models"] + TEMPLATE_IDS["splitters"]:
            self.assertIn(item, models)

    def test_python_example_of_the_tutorial_runs(self):
        with open(os.path.join(REPO, "plugins", "LEIA-ME.md"), encoding="utf-8") as handle:
            text = handle.read()
        code = text.split("```python\nimport numpy as np\nfrom multical import api\n")[1].split("```")[0]
        code = "import numpy as np\nfrom multical import api\n" + code
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"), PYTHONIOENCODING="utf-8", MULTICAL_PLUGINS="")
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO)
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = done.stdout.strip().splitlines()
        self.assertGreater(float(lines[0]), 0)
        self.assertIn("[", lines[1])


class WrongFormats(MethodsCase):
    """A method that returns the wrong thing is named in a readable message."""

    def tearDown(self):
        for item in ("t_linhas", "t_eixo", "t_um_item", "t_estado"):
            api.PRETREATMENTS.unregister(item)
        for item in ("t_formato", "t_estado_modelo", "t_caminho"):
            api.MODELS.unregister(item)
        for item in ("t_sobreposto", "t_fora", "t_vazio"):
            api.SPLITTERS.unregister(item)

    def _run(self, **config):
        base = {"pretreatment": [], "model": {"id": "pls", "params": {"max_k": 2}}}
        return api.run_calibration(self.dataset, api.RunConfig(**{**base, **config}))

    def test_pretreatments(self):
        api.pretreatment(id="t_linhas", label="Remove amostras")(lambda X, wl: (X[:-1], wl))
        api.pretreatment(id="t_eixo", label="Esquece o eixo")(lambda X, wl: (X[:, ::2], wl))
        api.pretreatment(id="t_um_item", label="Devolve só X")(lambda X, wl: X)

        @api.register(api.PRETREATMENTS)
        class TextState(api.Pretreatment):
            id, label = "t_estado", "Estado com texto"

            def fit(self, X, wavelengths):
                return {"modo": "rápido"}

            def apply(self, X, wavelengths, state):
                return X, wavelengths

        cases = (("t_linhas", "“Remove amostras”: recebeu 30 espectros e devolveu 29"),
                 ("t_eixo", "“Esquece o eixo”: os espectros devolvidos têm 75 pontos, mas o eixo devolvido tem 150"),
                 ("t_um_item", "“Devolve só X”: deve devolver dois itens"),
                 ("t_estado", "“Estado com texto”: o item 'modo' do estado não é um número"))
        for item, message in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as caught:
                    self._run(pretreatment=[{"id": item, "params": {}}])
                self.assertIn(message, str(caught.exception))

    def test_models(self):
        @api.register(api.MODELS)
        class WrongShape(api.Model):
            id, label = "t_formato", "Formato errado"

            def fit(self, X, y):
                return {"m": float(np.mean(y))}

            def predict(self, state, X):
                return np.full((X.shape[0], 2), state["m"])

        @api.register(api.MODELS)
        class ObjectState(api.Model):
            id, label = "t_estado_modelo", "Guarda um objeto"

            def fit(self, X, y):
                return {"modelo": object()}

            def predict(self, state, X):
                return np.zeros(X.shape[0])

        @api.register(api.MODELS)
        class ShortPath(api.Model):
            id, label, sweep = "t_caminho", "Caminho curto", "max_k"
            params = (api.Param("max_k", "Máximo", "int", default=3, min=1),)

            def fit(self, X, y, k):
                return {"m": float(np.mean(y))}

            def predict(self, state, X):
                return np.full(X.shape[0], state["m"])

            def cv_path(self, X_train, y_train, X_val, max_k):
                return np.zeros((max_k - 1, X_val.shape[0]))

        cases = (("t_formato", {}, "“Formato errado”: predict recebeu 6 espectros e devolveu uma matriz de formato (6, 2)"),
                 ("t_estado_modelo", {}, "“Guarda um objeto”: o item 'modelo' do estado não é um número"),
                 ("t_caminho", {"max_k": 3}, "“Caminho curto”: cv_path devolveu uma matriz de formato (2, 6); esperado (3, 6)"))
        for item, params, message in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as caught:
                    self._run(model={"id": item, "params": params})
                self.assertIn(message, str(caught.exception))

    def test_validation_schemes(self):
        def scheme(item, name, pairs):
            @api.register(api.SPLITTERS)
            class Scheme(api.Splitter):
                id, label = item, name

                def split(self, n, groups=None, X=None, Y=None, rng=None):
                    return pairs(n)

        scheme("t_sobreposto", "Repete amostras", lambda n: [(np.arange(n), np.arange(5))])
        scheme("t_fora", "Posição inexistente", lambda n: [(np.arange(5, n), np.array([0, n]))])
        scheme("t_vazio", "Sem validação", lambda n: [(np.arange(n), np.array([], dtype=int))])
        cases = (("t_sobreposto", "“Repete amostras”: a mesma amostra aparece no treino e na validação"),
                 ("t_fora", "“Posição inexistente”: as posições devem estar entre 0 e 29"),
                 ("t_vazio", "“Sem validação”: uma das divisões ficou sem amostras"))
        for item, message in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as caught:
                    self._run(validation={"id": item, "params": {}})
                self.assertIn(message, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
