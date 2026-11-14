"""
The optional neural network plugin (plugins/opcionais/rede_neural.py).

Estimating with a trained network needs no PyTorch and is always tested; the
training tests run only where PyTorch is installed.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from ._cases import ANALYTES, make_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from multical import api  # noqa: E402
from multical.api.bundle import SavedAnalyte  # noqa: E402

FOLDER = os.path.join(REPO, "plugins", "opcionais")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def without_pytorch():
    """Inside this block `import torch` fails, as in the installed program."""
    return mock.patch.dict(sys.modules, {"torch": None})


def _by_hand(name, h):
    """The activations written independently of the plugin (as PyTorch defines them)."""
    return {
        "relu": lambda v: np.where(v > 0, v, 0.0),
        "tanh": np.tanh,
        "sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
        "leaky_relu": lambda v: np.where(v > 0, v, 0.01 * v),
        "silu": lambda v: v * (1.0 / (1.0 + np.exp(-v))),
        "elu": lambda v: np.where(v > 0, v, np.exp(v) - 1.0),
    }[name](h)


class NeuralNetworkPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, cls.X, cls.Y = make_dataset()
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L")

    def setUp(self):
        api.MODELS.unregister("ann")
        del api.PLUGIN_ERRORS[:]
        self.loaded = api.load_plugins([FOLDER], reload=True)
        self.addCleanup(api.MODELS.unregister, "ann")
        self.model = api.MODELS.get("ann")

    def test_loads_without_pytorch_and_is_not_active_by_default(self):
        self.assertEqual([os.path.basename(p) for p in self.loaded], ["rede_neural.py"])
        self.assertEqual(api.PLUGIN_ERRORS, [])
        self.assertIn("PyTorch", self.model.label)
        params = api.resolve_params(self.model.params, {})
        self.assertEqual((params["hidden_units"], params["n_layers"], params["epochs"], params["activation"]),
                         (50, 1, 200, "relu"))
        # Only files directly in "plugins" are loaded by the program: this one waits in a subfolder.
        active = [n for n in os.listdir(os.path.join(REPO, "plugins")) if n.endswith(".py") and not n.startswith("_")]
        self.assertNotIn("rede_neural.py", active)

    def test_a_trained_network_estimates_without_pytorch(self):
        rng = np.random.RandomState(3)
        n_points = self.wl.size
        for code, name in enumerate(("relu", "tanh", "sigmoid", "leaky_relu", "silu", "elu")):
            with self.subTest(activation=name):
                weights = [rng.normal(size=(7, n_points)) * 0.05, rng.normal(size=(7, 7)), rng.normal(size=(1, 7))]
                biases = [rng.normal(size=7), rng.normal(size=7), rng.normal(size=1)]
                state = {"camadas": 3, "ativacao": code, "x_media": self.X.mean(axis=0), "x_desvio": self.X.std(axis=0),
                         "y_media": 4.0, "y_desvio": 2.5}
                for i, (w, b) in enumerate(zip(weights, biases)):
                    state[f"pesos{i}"], state[f"vies{i}"] = w.astype(np.float32), b.astype(np.float32)
                h = (self.X - state["x_media"]) / state["x_desvio"]
                h = _by_hand(name, h @ state["pesos0"].T.astype(float) + state["vies0"])
                h = _by_hand(name, h @ state["pesos1"].T.astype(float) + state["vies1"])
                expected = (h @ state["pesos2"].T.astype(float) + state["vies2"]).ravel() * 2.5 + 4.0
                np.testing.assert_allclose(self.model.predict(state, self.X), expected, rtol=1e-12)
                self.assertGreater(np.ptp(expected), 0)

        # The same state inside a model, as it comes back from a model file: numbers only, read without PyTorch.
        saved = api.SavedModel(
            config=api.RunConfig(pretreatment=[], model={"id": "ann", "params": {"activation": "elu"}}),
            analytes=["gl"], units="g/L", wavelengths_in=self.wl, pretreatment_states=[], x_mean=None, x_std=None,
            per_analyte={"gl": SavedAnalyte("gl", state, None, 0.3, 0.9, 0.5, 10.0)})
        np.testing.assert_allclose(api.predict(saved, self.X, self.wl)[:, 0], expected, rtol=1e-12)
        report = api.predict_report(saved, self.dataset)
        np.testing.assert_allclose(report.Y[:, 0], expected, rtol=1e-12)

    def test_training_without_pytorch_is_refused_with_an_explanation(self):
        config = api.RunConfig(pretreatment=[], model={"id": "ann", "params": {"epochs": 2}}, analytes=["gl"])
        with without_pytorch():
            with self.assertRaisesRegex(ValueError, "é preciso o PyTorch.*sem ele"):
                api.run_calibration(self.dataset, config)

    @unittest.skipUnless(HAS_TORCH, "needs PyTorch")
    def test_training_gives_the_estimates_of_the_original_code(self):
        from multical.models.ann_torch import ANNModel
        train, test = np.arange(22), np.arange(22, 30)
        y = self.Y[:, 1]
        cases = (
            {"hidden_units": 12, "n_layers": 1, "epochs": 30, "activation": "relu", "learning_rate": 0.001,
             "batch_size": 32, "early_stopping": False, "patience": 20},
            {"hidden_units": 8, "n_layers": 2, "epochs": 60, "activation": "tanh", "learning_rate": 0.01,
             "batch_size": 8, "early_stopping": True, "patience": 5},
        )
        for params in cases:
            with self.subTest(params=params):
                import torch
                legacy = ANNModel().fit_predict(self.X[train], y[train].reshape(-1, 1), self.X[test],
                                                device=torch.device("cpu"), **params).ravel()
                state = self.model.fit(self.X[train], y[train], **params)
                # The network computes in single precision and the plugin in double: the estimates agree to about 1e-6.
                np.testing.assert_allclose(self.model.predict(state, self.X[test]), legacy, rtol=1e-5, atol=1e-5)
                self.assertGreater(np.ptp(legacy), 0.5)                         # a real fit, not a constant
                again = self.model.fit(self.X[train], y[train], **params)       # same seed inside the original code
                np.testing.assert_array_equal(again["pesos0"], state["pesos0"])

    @unittest.skipUnless(HAS_TORCH, "needs PyTorch")
    def test_trains_from_the_engine_and_survives_the_model_file(self):
        config = api.RunConfig(pretreatment=[], model={"id": "ann", "params": {"epochs": 40, "hidden_units": 10}},
                               analytes=["cb", "gl"])
        result = api.run_calibration(self.dataset, config)
        self.assertEqual(len(result.per_analyte["gl"].rmsecv), 1)
        self.assertTrue(np.isfinite(result.per_analyte["gl"].best_rmsecv))
        new = self.X[:5] * 1.01
        expected = api.predict(result, new, self.wl)
        with tempfile.TemporaryDirectory() as tmp:
            path = api.save_model(result, os.path.join(tmp, "rede"))
            with without_pytorch():                      # the file is read and used where PyTorch does not exist
                saved = api.load_model(path)
                np.testing.assert_allclose(api.predict(saved, new, self.wl), expected, rtol=1e-10)
                report = api.predict_report(saved, api.Dataset(new, self.wl))
        np.testing.assert_allclose(report.Y, expected, rtol=1e-10)
        self.assertEqual(report.typical_error.shape, (2,))


if __name__ == "__main__":
    unittest.main()
