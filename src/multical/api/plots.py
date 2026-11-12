"""
Figures as pure functions: each takes data and returns a matplotlib Figure.

They use `matplotlib.figure.Figure` directly (no pyplot), so no window opens and
the GUI can embed the figure or save it to a file.
"""
import numpy as np
from matplotlib.figure import Figure

from .assessment import number


def plot_spectra(wavelengths, X, title="Espectros", xlabel="Comprimento de onda", ylabel="Absorbância"):
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(np.asarray(wavelengths), np.asarray(X).T, linewidth=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig


def plot_error_curve(result, analyte):
    """Validation and calibration error for each complexity value tried."""
    res = result.per_analyte[analyte]
    unit = f" ({result.units})" if result.units else ""
    fig = Figure(figsize=(6, 4), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(res.candidates, res.rmsecv, marker="o", label="Validação (RMSECV)")
    ax.plot(res.candidates, res.rmsec, marker="o", linestyle="--", label="Calibração (RMSEC)")
    ax.plot(res.candidates[res.best_index], res.rmsecv[res.best_index], "k*", markersize=14, label="Escolhido")
    ax.set(title=f"Erro por variáveis latentes – {analyte}", xlabel="Variáveis latentes", ylabel=f"Erro{unit}")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    return fig


def plot_predicted_vs_measured(result, analyte):
    """Validation predictions of the chosen model against the measured values."""
    res = result.per_analyte[analyte]
    j = result.analytes.index(analyte)
    mask = result.validated
    measured, predicted = result.Y[mask, j], res.y_cv[res.best_index, mask]
    unit = f" ({result.units})" if result.units else ""
    lo, hi = min(measured.min(), predicted.min()), max(measured.max(), predicted.max())
    fig = Figure(figsize=(5, 5), layout="constrained")
    ax = fig.add_subplot()
    ax.scatter(measured, predicted, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], "k--", label="Ideal")
    ax.set(
        title=f"{analyte}: erro típico ±{number(res.best_rmsecv)} {result.units}".rstrip() + f", R² = {number(res.best_r2cv)}",
        xlabel=f"Medido{unit}", ylabel=f"Previsto{unit}",
    )
    ax.legend()
    return fig


X_LABEL = "Número de onda (cm⁻¹)"


def plot_spectra_by_group(wavelengths, X, groups=None, highlight=None, title="Espectros", ylabel="Absorbância"):
    """Every spectrum, one colour per experiment; `highlight` is the position of a spectrum to draw on top."""
    X = np.atleast_2d(np.asarray(X))
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.add_subplot()
    names = ["" for _ in range(X.shape[0])] if groups is None else [str(g) for g in groups]
    for number, name in enumerate(dict.fromkeys(names)):
        rows = [i for i, g in enumerate(names) if g == name]
        lines = ax.plot(np.asarray(wavelengths), X[rows].T, color=f"C{number % 10}", linewidth=0.7, alpha=0.8)
        if name:
            lines[0].set_label(name)
    if highlight is not None and 0 <= highlight < X.shape[0]:
        ax.plot(np.asarray(wavelengths), X[highlight], color="black", linewidth=2.0, label="Selecionada")
    ax.set(title=title, xlabel=X_LABEL, ylabel=ylabel)
    if groups is not None or highlight is not None:
        ax.legend(fontsize=8, loc="best")
    return fig


def plot_before_after(wl_before, X_before, wl_after, X_after):
    """Spectra as imported (top) and after the pretreatment recipe (bottom)."""
    fig = Figure(figsize=(7, 6), layout="constrained")
    top, bottom = fig.subplots(2, 1)
    top.plot(np.asarray(wl_before), np.asarray(X_before).T, linewidth=0.7)
    top.set(title="Antes do pré-tratamento", ylabel="Absorbância")
    bottom.plot(np.asarray(wl_after), np.asarray(X_after).T, linewidth=0.7)
    bottom.set(title="Depois do pré-tratamento", xlabel=X_LABEL, ylabel="Sinal tratado")
    return fig


def plot_prediction_over_time(prediction, analyte):
    """Estimates of one analyte along the time, one line per experiment, with the typical error as a band."""
    j = prediction.analytes.index(analyte)
    unit = f" ({prediction.units})" if prediction.units else ""
    n = len(prediction.sample_ids)
    x = np.asarray(prediction.times, dtype=float) if prediction.times is not None else np.arange(1, n + 1, dtype=float)
    groups = ["" for _ in range(n)] if prediction.groups is None else [str(g) for g in prediction.groups]
    error = prediction.typical_error[j]
    ok = prediction.ok
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.add_subplot()
    for number, name in enumerate(dict.fromkeys(groups)):
        rows = np.array([i for i, g in enumerate(groups) if g == name])
        rows = rows[np.argsort(x[rows])]
        color = f"C{number % 10}"
        ax.plot(x[rows], prediction.Y[rows, j], color=color, linewidth=1.2, marker="o", markersize=2.5, label=name or None)
        if error == error:
            ax.fill_between(x[rows], prediction.Y[rows, j] - error, prediction.Y[rows, j] + error, color=color, alpha=0.15)
    if not ok.all():
        ax.plot(x[~ok], prediction.Y[~ok, j], linestyle="none", marker="x", color="red", markersize=6,
                label="Fora do que o modelo conhece")
    low, high = prediction.ranges[j]
    if low == low:
        ax.axhline(low, color="gray", linestyle=":", linewidth=1)
        ax.axhline(high, color="gray", linestyle=":", linewidth=1, label="Faixa calibrada")
    ax.set(title=f"{analyte} estimado", xlabel="Tempo (min)" if prediction.times is not None else "Amostra",
           ylabel=f"{analyte}{unit}")
    ax.legend(fontsize=8, loc="best")
    return fig
