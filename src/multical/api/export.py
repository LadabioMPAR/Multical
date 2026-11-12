"""
Writes the outcome of a calibration to one folder: model file, tables, figures,
report and configuration.

Tables are CSV with ';' between columns and a decimal comma, which is what a
Brazilian Excel opens directly.
"""
import csv
import os
import re

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import plots
from .assessment import LEVELS, assess
from .bundle import save_model

EXPORT_ITEMS = {
    "model": "Modelo para uso em rotina (.multical)",
    "tables": "Tabelas de resultados (CSV)",
    "figures": "Figuras (PNG)",
    "report": "Relatório (PDF)",
    "config": "Configuração usada (JSON)",
}


def _safe(name):
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_") or "analito"


def _cell(value):
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:.6g}".replace(".", ",")
    return "" if value is None else str(value)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        writer.writerows([[_cell(v) for v in row] for row in rows])
    return path


def _tables(result, folder):
    summary, per_k = [], []
    for item, name in zip(assess(result), result.analytes):
        res, b = result.per_analyte[name], result.per_analyte[name].best_index
        summary.append([name, result.units, len(result.sample_ids), res.best_k, res.rmsecv[b], res.r2cv[b],
                        res.rmsec[b], res.r2cal[b], LEVELS[item.level]])
        for i, k in enumerate(res.candidates):
            per_k.append([name, int(k), res.rmsecv[i], res.r2cv[i], res.rmsec[i], res.r2cal[i], "sim" if i == b else ""])
    header = ["amostra", "validada"]
    for name in result.analytes:
        header += [f"{name}_medido", f"{name}_previsto_validacao", f"{name}_previsto_calibracao"]
    predictions = []
    for i, sample in enumerate(result.sample_ids):
        row = [sample, "sim" if result.validated[i] else "não"]
        for j, name in enumerate(result.analytes):
            res = result.per_analyte[name]
            row += [result.Y[i, j], res.y_cv[res.best_index, i], res.y_cal[res.best_index, i]]
        predictions.append(row)
    return [
        _write_csv(os.path.join(folder, "resumo.csv"),
                   ["analito", "unidade", "amostras", "variaveis_latentes", "erro_validacao_rmsecv", "r2_validacao",
                    "erro_calibracao_rmsec", "r2_calibracao", "avaliacao"], summary),
        _write_csv(os.path.join(folder, "metricas_por_variaveis_latentes.csv"),
                   ["analito", "variaveis_latentes", "erro_validacao_rmsecv", "r2_validacao", "erro_calibracao_rmsec",
                    "r2_calibracao", "escolhido"], per_k),
        _write_csv(os.path.join(folder, "predicoes.csv"), header, predictions),
    ]


def _summary_page(result):
    fig = Figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, "Multical – resumo da calibração", fontsize=16, weight="bold")
    cfg = result.config
    lines = [
        f"Amostras: {len(result.sample_ids)}    Pontos por espectro após o pré-tratamento: {len(result.wavelengths)}",
        f"Modelo: {cfg.model['id']}  {cfg.model['params']}",
        f"Validação: {cfg.validation['id']}  {cfg.validation['params']}",
        "Pré-tratamento: " + ("; ".join(f"{s['id']} {s['params']}" for s in cfg.pretreatment) or "nenhum"),
        "",
    ]
    for item in assess(result):
        lines.append(f"[{LEVELS[item.level]}]  {item.analyte}: {item.summary}")
        lines += [f"        erro grande na validação – {s}" for s in item.suspects[:5]]
    fig.text(0.08, 0.89, "\n".join(lines), fontsize=9.5, va="top", wrap=True)
    return fig


def export_run(result, folder, items=None):
    """
    Write the chosen `items` (keys of EXPORT_ITEMS; default: all) into `folder`.
    Returns the list of files written.
    """
    items = list(EXPORT_ITEMS) if items is None else list(items)
    unknown = [i for i in items if i not in EXPORT_ITEMS]
    if unknown:
        raise ValueError(f"Item de exportação desconhecido: {', '.join(unknown)}.")
    os.makedirs(folder, exist_ok=True)
    written = []
    if "model" in items:
        written.append(save_model(result, os.path.join(folder, "modelo.multical")))
    if "tables" in items:
        written += _tables(result, folder)
    if "config" in items:
        path = os.path.join(folder, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(result.config.to_json())
        written.append(path)

    figures = []
    if "figures" in items or "report" in items:
        for name in result.analytes:
            figures.append((f"previsto_x_medido_{_safe(name)}", plots.plot_predicted_vs_measured(result, name)))
            if len(result.per_analyte[name].candidates) > 1:
                figures.append((f"erro_por_variaveis_latentes_{_safe(name)}", plots.plot_error_curve(result, name)))
    if "figures" in items:
        os.makedirs(os.path.join(folder, "figuras"), exist_ok=True)
        for stem, fig in figures:
            path = os.path.join(folder, "figuras", stem + ".png")
            fig.savefig(path, dpi=300)
            written.append(path)
    if "report" in items:
        path = os.path.join(folder, "relatorio.pdf")
        with PdfPages(path) as pdf:
            pdf.savefig(_summary_page(result))
            for _, fig in figures:
                pdf.savefig(fig)
        written.append(path)
    return written
