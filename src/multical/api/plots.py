"""
Figures as pure functions: each takes data and returns a matplotlib Figure.

They use `matplotlib.figure.Figure` directly (no pyplot), so no window opens and
the GUI can embed the figure or save it to a file.
"""
import numpy as np
from matplotlib.figure import Figure

from .assessment import error_name, candidate_name, number
from .influence import influence


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
    ax.plot(res.candidates, res.rmsecv, marker="o", label=f"Validação ({error_name(result.config)})")
    ax.plot(res.candidates, res.rmsec, marker="o", linestyle="--", label="Calibração (RMSEC)")
    ax.plot(res.candidates[res.best_index], res.rmsecv[res.best_index], "k*", markersize=14, label="Escolhido")
    name = candidate_name(result)
    if res.labels is not None:
        ax.set_title(f"Erro por ajuste testado – {analyte}\nescolhido: {res.best_label}", fontsize=10)
        ax.set(xlabel=f"{name} (número na tabela)", ylabel=f"Erro{unit}")
    else:
        ax.set(title=f"Erro por {name.lower()} – {analyte}", xlabel=name, ylabel=f"Erro{unit}")
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
        title=f"{analyte}: {error_name(result.config)} = {number(res.best_rmsecv)} {result.units}".rstrip()
              + f", R² = {number(res.best_r2cv)}",
        xlabel=f"Medido{unit}", ylabel=f"Previsto{unit}",
    )
    ax.legend()
    return fig


X_LABEL = "Número de onda (cm⁻¹)"


HIGHLIGHT = "destaque"      # gid of the line that draws the highlighted spectrum


def set_highlight(fig, wavelengths, X, highlight=None, label="Selecionada"):
    """Draw spectrum `highlight` of X on top of a figure of `plot_spectra_by_group` (None: no spectrum), in place."""
    ax = fig.axes[0]
    line = next(item for item in ax.lines if item.get_gid() == HIGHLIGHT)
    X = np.atleast_2d(np.asarray(X))
    if highlight is not None and 0 <= highlight < X.shape[0]:
        line.set_data(np.asarray(wavelengths), X[highlight])
        line.set_label(label)
        line.set_visible(True)
    else:
        line.set_data([], [])
        line.set_label("_nolegend_")
        line.set_visible(False)
    shown = [item for item in ax.lines if not item.get_label().startswith("_")]
    if shown:
        ax.legend(fontsize=8, loc="best")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_spectra_by_group(wavelengths, X, groups=None, highlight=None, title="Espectros", ylabel="Absorbância",
                          highlight_label="Selecionada"):
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
    ax.plot([], [], color="black", linewidth=2.0, gid=HIGHLIGHT)
    ax.set(title=title, xlabel=X_LABEL, ylabel=ylabel)
    set_highlight(fig, wavelengths, X, highlight, highlight_label)
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


def plot_selected_points(result, analyte):
    """Mean pretreated spectrum of the calibration with the points used by the model of one analyte marked."""
    res = result.per_analyte[analyte]
    axis = np.asarray(result.wavelengths, dtype=float)
    mean = np.asarray(result.X_pre).mean(axis=0)
    used = np.ones(axis.size, dtype=bool) if res.selected is None else np.asarray(res.selected, dtype=bool)
    fig = Figure(figsize=(7, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(axis, mean, color="gray", linewidth=0.8, label="Espectro médio (tratado)")
    ax.plot(axis[used], mean[used], linestyle="none", marker="o", markersize=2.5, color="C3", label="Pontos usados pelo modelo")
    ax.set(title=f"{analyte}: {int(used.sum())} de {axis.size} pontos do espectro", xlabel=X_LABEL, ylabel="Sinal tratado")
    ax.legend(fontsize=8, loc="best")
    return fig


T2_LABEL = "T² de Hotelling (distância dentro do que o modelo usa)"
Q_LABEL = "Resíduo Q"       # short: it runs along the height of the figure
MOST_LABELS = 15        # samples named on a Q × T² figure, the farthest first


def _limits(ax, t2_limit, q_limit, confidence):
    percent = f" ({confidence * 100:.0f}%)" if confidence else ""
    if np.isfinite(t2_limit):
        ax.axvline(t2_limit, color="black", linestyle="--", linewidth=1, label=f"Limite de T²{percent}")
    if np.isfinite(q_limit):
        ax.axhline(q_limit, color="black", linestyle=":", linewidth=1, label=f"Limite de Q{percent}")
    ax.set(xlabel=T2_LABEL, ylabel=Q_LABEL)


def plot_influence(result, analyte):
    """Q × T² of the calibration samples for the model of one analyte; samples beyond a limit are named."""
    data = influence(result, analyte)
    outside = data.outside
    fig = Figure(figsize=(6.5, 4.8), layout="constrained")
    ax = fig.add_subplot()
    ax.scatter(data.t2[~outside], data.q[~outside], alpha=0.7, label="Dentro dos limites")
    if outside.any():
        ax.scatter(data.t2[outside], data.q[outside], color="black", marker="x", label="Além de um limite")
        positions = {name: i for i, name in enumerate(data.sample_ids)}
        for name in data.flagged()[:MOST_LABELS]:
            i = positions[name]
            ax.annotate(name, (data.t2[i], data.q[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    _limits(ax, data.t2_limit, data.q_limit, data.confidence)
    if data.source == "pls":
        basis = f"PLS com {data.components} " + ("variável latente" if data.components == 1 else "variáveis latentes")
    else:
        basis = f"resumo dos espectros em {data.components} " + ("componente principal" if data.components == 1 else "componentes principais")
    count = int(outside.sum())
    ax.set_title(f"{analyte}: {count} de {len(data.sample_ids)} amostras além dos limites\n{basis}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    return fig


def plot_prediction_domain(prediction):
    """Q × T² of the estimated spectra against the limits of the model, one colour per experiment."""
    fig = Figure(figsize=(6.5, 4.8), layout="constrained")
    if prediction.t2 is None:
        fig.text(0.5, 0.5, "Este modelo não guarda o resumo dos espectros\nnecessário para o gráfico Q × T².",
                 ha="center", va="center", color="gray", fontsize=11)
        return fig
    ax = fig.add_subplot()
    n = len(prediction.sample_ids)
    groups = ["" for _ in range(n)] if prediction.groups is None else [str(g) for g in prediction.groups]
    inside = np.asarray(prediction.inside, dtype=bool)
    for number_, name in enumerate(dict.fromkeys(groups)):
        rows = np.array([i for i, g in enumerate(groups) if g == name])
        ax.scatter(prediction.t2[rows], prediction.q[rows], color=f"C{number_ % 10}", alpha=0.6, s=18, label=name or "Espectros")
    if not inside.all():
        ax.scatter(prediction.t2[~inside], prediction.q[~inside], color="red", marker="x", s=40,
                   label="Diferente dos usados para criar o modelo")
    _limits(ax, prediction.t2_limit, prediction.q_limit, None)
    ax.set_title(f"{int((~inside).sum())} de {n} espectros além dos limites do modelo", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    return fig


def plot_prediction_over_time(prediction, analyte):
    """Estimates of one analyte along the time, one line per experiment, with ± the model's RMSECV as a band."""
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
            ax.fill_between(x[rows], prediction.Y[rows, j] - error, prediction.Y[rows, j] + error, color=color, alpha=0.15,
                            label=f"± {prediction.error_name}" if number == 0 else None)
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
