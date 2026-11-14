"""
Sorting a handful of files chosen at once into experiments: which files hold
spectra, which hold measured values, and which table goes with which spectra.
No Qt in here.

The pairing is a suggestion made from the file names; the import screen shows
it to the user, who confirms or corrects it before anything is added.
"""
import os
import re

from multical.io.reference import _number, _split

SPECTRA_MIN_COLUMNS = 30    # a text file with at least this many columns holds spectra; a table of measured values has few
LONGEST_ROW = 4_000_000     # bytes read at most for one row (a spectrum of some ten thousand points is far below this)


def file_kind(path):
    """
    "spectra", "table" or "" (not a file this program reads).

    `.spp` files are spectra. A text file counts when its first row of data
    (after an optional row of names) is made of numbers, and is told apart by
    its width: spectra have one column per point of the axis, a table has the
    time and the analytes.
    """
    if str(path).lower().endswith(".spp"):
        return "spectra"
    lines = []
    try:
        with open(path, "rb") as handle:
            while len(lines) < 2:
                raw = handle.readline(LONGEST_ROW)
                if not raw:
                    break
                if b"\x00" in raw[:4096]:           # binary: neither kind
                    return ""
                line = raw.decode("latin-1").strip()
                if line and not line.startswith("#"):
                    lines.append(line)
    except OSError:
        return ""
    if not lines:
        return ""
    cells, decimal_comma = _split(lines[-1])        # the second row, or the only one
    try:
        [_number(cell, decimal_comma) for cell in cells]
    except ValueError:
        return ""
    if len(cells) >= SPECTRA_MIN_COLUMNS:
        return "spectra"
    return "table" if len(cells) >= 2 else ""


def _tokens(path):
    """Words and numbers of a file name, lower case: 'Exp04_nonda.txt' -> {'exp', '4', 'nonda'}."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return {str(int(t)) if t.isdigit() else t for t in re.findall(r"[0-9]+|[^\W\d_]+", stem)}


def _name_order(path):
    """Sorting key that puts exp2 before exp10."""
    name = os.path.basename(path).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", name)], path


def sort_files(paths):
    """
    Returns `(pairs, spare_tables, ignored)`.

    `pairs` has one `(spectra_path, table_path_or_None)` per spectra file, in
    the order of the file names. A table is suggested only when the two names point to each
    other without doubt: each is the other's single best match, counting the
    words and numbers the names share and, on a draw, the ones they do not (so
    exp4_nonda goes with exp4_refe, exp1 is never taken for exp10, and with
    exp8_nonda and exp8_inf next to one exp8_refe nothing is guessed). With exactly one spectra file and one table
    the pair is taken as chosen. `spare_tables` are tables left without spectra
    and `ignored` the files that are neither.
    """
    spectra, tables, ignored = [], [], []
    for path in sorted(dict.fromkeys(os.path.normpath(str(p)) for p in paths), key=_name_order):
        kind = file_kind(path)
        (spectra if kind == "spectra" else tables if kind == "table" else ignored).append(path)

    if len(spectra) == 1 and len(tables) == 1:
        return [(spectra[0], tables[0])], [], ignored

    def score(s, t):
        a, b = _tokens(s), _tokens(t)
        return (len(a & b), -len(a ^ b))

    def single_best(item, others, key):
        ranked = sorted(others, key=key, reverse=True)
        if not ranked or key(ranked[0])[0] == 0:
            return None
        if len(ranked) > 1 and key(ranked[1]) == key(ranked[0]):
            return None
        return ranked[0]

    best_table = {s: single_best(s, tables, lambda t, s=s: score(s, t)) for s in spectra}
    best_spectra = {t: single_best(t, spectra, lambda s, t=t: score(s, t)) for t in tables}
    pairs = [(s, best_table[s] if best_table[s] is not None and best_spectra[best_table[s]] == s else None) for s in spectra]
    used = {t for _, t in pairs if t is not None}
    return pairs, [t for t in tables if t not in used], ignored
