"""
Figures as pure functions: each takes data and returns a matplotlib Figure.

They use `matplotlib.figure.Figure` directly (no pyplot), so no window opens and
the GUI can embed the figure or save it to a file.
"""
import numpy as np
from matplotlib.figure import Figure


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
        title=f"{analyte}: erro típico ±{res.best_rmsecv:.3g}{unit}, R² = {res.best_r2cv:.3f}",
        xlabel=f"Medido{unit}", ylabel=f"Previsto{unit}",
    )
    ax.legend()
    return fig
