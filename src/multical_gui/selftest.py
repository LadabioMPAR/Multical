"""
Self-check of an installed copy: `Multical.exe --autoteste relatorio.txt`.

Runs, without showing anything, the parts most likely to break when the
program is packaged: worker processes, every registered model, the model
file, the export (figures and PDF), the window and the plugin folders. Writes
a short report and returns 0 when everything passed.
"""
import multiprocessing
import os
import platform
import sys
import tempfile
import traceback

import numpy as np


def _square(value):
    return value * value


def _synthetic():
    """A small calibration set whose spectra follow two concentrations."""
    from multical import api
    rng = np.random.RandomState(0)
    x = np.linspace(0, 1, 120)
    conc = np.column_stack([rng.uniform(1, 10, 24), rng.uniform(0, 5, 24)])
    bands = np.vstack([np.exp(-((x - 0.3) / 0.07) ** 2), np.exp(-((x - 0.7) / 0.1) ** 2)])
    spectra = conc @ bands + 0.01 * rng.normal(size=(24, 120))
    groups = np.array(["exp1"] * 12 + ["exp2"] * 12)
    return api.Dataset(spectra, 4000.0 + np.arange(120), conc, analytes=["a", "b"], units="g/L", groups=groups)


def _check_processes():
    with multiprocessing.Pool(2) as pool:
        result = pool.map(_square, range(8))
    if result != [v * v for v in range(8)]:
        raise RuntimeError(f"resultado inesperado: {result}")
    return "2 processos"


def _check_pretreatments():
    """Every registered pretreatment with its default settings, on spectra it was not fitted with."""
    from multical import api
    from multical.api.contract import ContractError
    data = _synthetic()
    new = data.X[:6] * 1.02 + 0.01
    done, skipped = [], []
    for item in api.PRETREATMENTS:
        recipe = [{"id": item.id, "params": {}}]
        try:
            _, _, states = api.fit_recipe(recipe, data.X, data.wavelengths)
        except ContractError:
            raise
        except ValueError:      # the default settings do not fit these spectra (for instance a region to be typed in)
            skipped.append(item.id)
            continue
        whole, axis = api.apply_recipe(recipe, states, new, data.wavelengths)
        if not np.all(np.isfinite(whole)):
            raise RuntimeError(f"{item.id}: devolveu valores ausentes ou infinitos")
        # A model file is used on batches of any size: one spectrum must give the same result alone or in a group.
        halves = np.vstack([api.apply_recipe(recipe, states, part, data.wavelengths)[0] for part in (new[:3], new[3:])])
        if not np.allclose(whole, halves, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"{item.id}: o resultado de um espectro muda conforme os outros espectros do grupo; "
                "o que depende do conjunto de calibração deve ser aprendido em fit e guardado no estado")
        done.append(item.id)
    return ", ".join(done) + (f" (não testados com os ajustes padrão: {', '.join(skipped)})" if skipped else "")


def _check_models():
    from multical import api
    data = _synthetic()
    done = []
    for model in api.MODELS:
        params = {model.sweep: 3} if model.sweep else {}
        result = api.run_calibration(data, api.RunConfig(model={"id": model.id, "params": params}))
        estimates = api.predict(result, data.X[:3], data.wavelengths)
        if not np.all(np.isfinite(estimates)):
            raise RuntimeError(f"{model.id}: estimativas inválidas")
        done.append(model.id)
    for scheme in api.SPLITTERS:
        api.run_calibration(data, api.RunConfig(model={"id": "pls", "params": {"max_k": 3}},
                                                validation={"id": scheme.id, "params": {}}))
    return f"modelos {', '.join(done)}; validações {', '.join(api.SPLITTERS.ids())}"


def _check_selectors():
    """Every registered variable selection with its default settings, then the model file made with it."""
    from multical import api
    data = _synthetic()
    done = []
    for selector in api.SELECTORS:
        result = api.run_calibration(data, api.RunConfig(
            model={"id": "pls", "params": {"max_k": 3}}, analytes=["a"],
            variable_selection={"id": selector.id, "params": {}}))
        with tempfile.TemporaryDirectory() as folder:
            model = api.load_model(api.save_model(result, os.path.join(folder, "modelo")))
        if not np.allclose(api.predict(model, data.X, data.wavelengths), api.predict(result, data.X, data.wavelengths)):
            raise RuntimeError(f"{selector.id}: o modelo relido não repete as estimativas")
        done.append(f"{selector.id} ({result.per_analyte['a'].n_selected} de {result.wavelengths.size} pontos)")
    return ", ".join(done)


def _check_files():
    from multical import api
    from multical.api import export
    data = _synthetic()
    result = api.run_calibration(data, api.RunConfig(model={"id": "pls", "params": {"max_k": 3}}))
    with tempfile.TemporaryDirectory() as folder:
        written = export.export_run(result, os.path.join(folder, "execucao"))
        model = api.load_model(os.path.join(folder, "execucao", "modelo.multical"))
        report = api.predict_report(model, data)
        if not np.allclose(report.Y, api.predict(result, data.X, data.wavelengths)):
            raise RuntimeError("o modelo relido não repete as estimativas")
        api.export_predictions(report, os.path.join(folder, "estimativas.csv"))
        empty = [os.path.basename(p) for p in written if os.path.getsize(p) == 0]
        if empty:
            raise RuntimeError(f"arquivos vazios: {empty}")
    return f"{len(written)} arquivos exportados e modelo relido"


def _check_window():
    from PySide6.QtWidgets import QApplication

    from .app import prepare
    from .wizard import MainWindow
    app = QApplication.instance() or QApplication(sys.argv[:1])
    prepare(app)
    window = MainWindow()
    image = window.grab()
    count = len(window.flows["create"]) + 1
    window.close()
    if image.isNull():
        raise RuntimeError("a janela não pôde ser desenhada")
    return f"{count} telas, {image.width()}x{image.height()}"


def _check_plugins():
    from multical import api
    from multical.api.plugins import plugin_folders
    if api.PLUGIN_ERRORS:
        raise RuntimeError("; ".join(f"{os.path.basename(path)}: {text.splitlines()[0]}" for path, text in api.PLUGIN_ERRORS))
    return f"pastas {plugin_folders() or 'nenhuma'}; pré-tratamentos {', '.join(api.PRETREATMENTS.ids())}"


CHECKS = (
    ("Processos em paralelo", _check_processes),
    ("Pré-tratamentos", _check_pretreatments),
    ("Modelos e validações", _check_models),
    ("Seleção de variáveis", _check_selectors),
    ("Arquivos (modelo, tabelas, figuras, PDF)", _check_files),
    ("Janela", _check_window),
    ("Plugins", _check_plugins),
)


def run(report_path=None):
    import PySide6
    lines = [
        "Autoteste do Multical",
        f"Programa empacotado: {'sim' if getattr(sys, 'frozen', False) else 'não'} · Python {platform.python_version()} · "
        f"numpy {np.__version__} · PySide6 {PySide6.__version__} · {platform.platform()}",
        "",
    ]
    failures = 0
    for name, check in CHECKS:
        try:
            lines.append(f"OK     {name}: {check()}")
        except Exception as error:      # the report must say what failed, whatever it was
            failures += 1
            lines.append(f"FALHA  {name}: {type(error).__name__}: {error}")
            lines.append(traceback.format_exc(limit=6))
    lines += ["", "Tudo certo." if failures == 0 else f"{failures} verificação(ões) falharam."]
    text = "\n".join(lines) + "\n"
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0 if failures == 0 else 1
