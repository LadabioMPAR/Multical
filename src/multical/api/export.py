"""
Writes the outcome of a calibration to one folder: model file, tables, figures,
report and configuration.

Tables are CSV with ';' between columns and a decimal comma, which is what a
Brazilian Excel opens directly.
"""
import csv
import os
import re
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import plots
from .assessment import LEVELS, assess, candidate_name, selection_notes
from .bundle import save_model
from .influence import influence

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
        chosen = res.best_label if res.labels is not None else res.best_k
        summary.append([name, result.units, len(result.sample_ids), chosen, res.rmsecv[b], res.r2cv[b],
                        res.rmsec[b], res.r2cal[b], LEVELS[item.level],
                        len(result.wavelengths) if res.selected is None else res.n_selected, len(result.wavelengths),
                        res.full_rmsecv])
        for i, k in enumerate(res.candidates):
            tried = res.labels[i] if res.labels is not None else int(k)
            per_k.append([name, tried, res.rmsecv[i], res.r2cv[i], res.rmsec[i], res.r2cal[i], "sim" if i == b else ""])
    tried_header = candidate_name(result).lower().replace(" ", "_")
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
    files = []
    try:
        maps = [influence(result, name) for name in result.analytes]
    except ValueError:              # a result without the summary of the spectra
        maps = []
    if maps:
        header_q = ["amostra"]
        for name in result.analytes:
            header_q += [f"{name}_T2", f"{name}_Q", f"{name}_alem_dos_limites"]
        rows_q = []
        for i, sample in enumerate(result.sample_ids):
            row = [sample]
            for data in maps:
                row += [float(data.t2[i]), float(data.q[i]), "sim" if data.outside[i] else ""]
            rows_q.append(row)
        limits = ["limite"]
        for data in maps:
            limits += [data.t2_limit, data.q_limit, ""]
        files.append(_write_csv(os.path.join(folder, "amostras_q_t2.csv"), header_q, [limits] + rows_q))
    if any(result.per_analyte[name].selected is not None for name in result.analytes):
        points = []
        for i, position in enumerate(result.wavelengths):
            marks = [result.per_analyte[name].selected for name in result.analytes]
            points.append([float(position)] + ["sim" if mark is None or mark[i] else "" for mark in marks])
        files.append(_write_csv(os.path.join(folder, "pontos_escolhidos.csv"),
                                ["posicao_no_eixo"] + [f"{name}_usa" for name in result.analytes], points))
    return files + [
        _write_csv(os.path.join(folder, "resumo.csv"),
                   ["analito", "unidade", "amostras", tried_header, "erro_validacao_rmsecv", "r2_validacao",
                    "erro_calibracao_rmsec", "r2_calibracao", "avaliacao", "pontos_usados", "pontos_do_espectro",
                    "erro_validacao_todos_os_pontos"], summary),
        _write_csv(os.path.join(folder, "metricas_por_ajuste.csv"),
                   ["analito", tried_header, "erro_validacao_rmsecv", "r2_validacao", "erro_calibracao_rmsec",
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
    ]
    if cfg.variable_selection:
        lines.append(f"Seleção de variáveis: {cfg.variable_selection['id']}  {cfg.variable_selection['params']}")
    lines.append("")
    for item in assess(result):
        lines.append(f"[{LEVELS[item.level]}]  {item.analyte}: {item.summary}")
        lines += [f"        erro grande na validação – {s}" for s in item.suspects[:5]]
    for name in result.analytes:
        try:
            odd = influence(result, name).flagged()
        except ValueError:
            break
        if odd:
            more = f" e mais {len(odd) - 8}" if len(odd) > 8 else ""
            lines.append(textwrap.fill(f"Espectros além dos limites de Q × T² – {name}: {', '.join(odd[:8])}{more}", 105,
                                       subsequent_indent="    "))
    notes = selection_notes(result)
    if notes:
        lines += ["", "Seleção de variáveis:"] + [textwrap.fill(note, 105, subsequent_indent="    ") for note in notes]
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
                figures.append((f"erro_por_ajuste_{_safe(name)}", plots.plot_error_curve(result, name)))
            if result.per_analyte[name].selected is not None:
                figures.append((f"pontos_escolhidos_{_safe(name)}", plots.plot_selected_points(result, name)))
            if result.domain or result.config.model["id"] == "pls":
                figures.append((f"q_x_t2_{_safe(name)}", plots.plot_influence(result, name)))
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
