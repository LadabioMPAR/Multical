"""
Reader for the table of measured (reference) values of one experiment.

One row per sample: sampling time in minutes, then one column per analyte.
Text files only (.txt, .csv, .tsv); columns separated by tab, semicolon, comma
or spaces. A first row with names is optional. Decimal commas are accepted
when the separator is not a comma.
"""
import os

import numpy as np


def _split(line):
    for separator in ("\t", ";"):
        if separator in line:
            return [cell.strip() for cell in line.split(separator)], True
    cells = line.split()
    if len(cells) > 1:
        return cells, True
    return [cell.strip() for cell in line.split(",")], False


def _number(cell, decimal_comma):
    return float(cell.replace(",", ".") if decimal_comma else cell)


def read_reference_table(path):
    """Returns `(times_min, Y, analytes)`; `analytes` is None when the file has no header row."""
    name = os.path.basename(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if os.path.splitext(name)[1].lower() in (".xlsx", ".xls", ".ods"):
        raise ValueError(f"{name}: planilhas não são lidas diretamente; salve como CSV ou texto separado por tabulação.")
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("#"):
            cells, decimal_comma = _split(line.strip())
            rows.append((number, cells, decimal_comma))
    if not rows:
        raise ValueError(f"{name} está vazio.")

    analytes = None
    try:
        [_number(cell, rows[0][2]) for cell in rows[0][1]]
    except ValueError:
        analytes = [cell for cell in rows[0][1][1:]]
        rows = rows[1:]
        if not rows:
            raise ValueError(f"{name} só tem a linha de nomes.") from None

    width = len(rows[0][1])
    if width < 2:
        raise ValueError(f"{name} precisa de pelo menos duas colunas: tempo (min) e um valor medido.")
    values = np.empty((len(rows), width))
    for i, (number, cells, decimal_comma) in enumerate(rows):
        if len(cells) != width:
            raise ValueError(f"{name}, linha {number}: {len(cells)} colunas, mas as anteriores têm {width}.")
        try:
            values[i] = [_number(cell, decimal_comma) for cell in cells]
        except ValueError:
            raise ValueError(f"{name}, linha {number}: há um valor que não é número.") from None
    if analytes is not None and len(analytes) != width - 1:
        raise ValueError(f"{name}: a linha de nomes tem {len(analytes) + 1} colunas e os dados têm {width}.")
    return values[:, 0], values[:, 1:], analytes
