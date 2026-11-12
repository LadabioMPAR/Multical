"""
Writes small synthetic .spp files for the tests (the real ones are far too
large to keep in the repository).

The writer is independent from the reader in `multical.io`: it builds the OLE
compound file and the property set from the format description, so the tests
check the reader against a second implementation of the layout.
"""
import struct

import numpy as np

SECTOR, MINI, CUTOFF = 512, 64, 4096
ENDOFCHAIN, FREESECT, FATSECT, DIFSECT = 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFD, 0xFFFFFFFC
STORAGE_NAME = "\x05Tlj0fw0jAeudbek4Aai40exuMg"


def _text(value):
    raw = (value + "\x00").encode("utf-16-le")
    return struct.pack("<I", len(raw)) + raw + b"\x00" * (-len(raw) % 4)


def _property_set(props):
    """props: list of (id, type, value) with type in 'i2', 'i4', 'r8', 'str', 'stream'."""
    values = []
    for _, kind, value in props:
        if kind == "i2":
            values.append(struct.pack("<Ihxx", 2, value))
        elif kind == "i4":
            values.append(struct.pack("<Ii", 3, value))
        elif kind == "r8":
            values.append(struct.pack("<Id", 5, value))
        elif kind == "str":
            values.append(struct.pack("<I", 8) + _text(value))
        elif kind == "stream":
            values.append(struct.pack("<I", 66) + _text(value))
    table, offset = b"", 8 + 8 * len(props)
    for (prop_id, _, _), value in zip(props, values):
        table += struct.pack("<II", prop_id, offset)
        offset += len(value)
    section = struct.pack("<II", offset, len(props)) + table + b"".join(values)
    header = struct.pack("<HHI", 0xFFFE, 0, 0x00020106) + b"\x00" * 16 + struct.pack("<I", 1)
    return header + b"\x11" * 16 + struct.pack("<I", 48) + section


def write_compound_file(path, streams, reverse=()):
    """`streams`: {name: bytes}. Names in `reverse` get their sectors laid out backwards."""
    sectors, fat = [], []

    def allocate(data, backwards=False):
        count = max(-(-len(data) // SECTOR), 1)
        ids = list(range(len(sectors), len(sectors) + count))
        order = ids[::-1] if backwards else ids
        sectors.extend([b""] * count)
        fat.extend([0] * count)
        for k, sector in enumerate(order):
            sectors[sector] = data[k * SECTOR:(k + 1) * SECTOR].ljust(SECTOR, b"\x00")
            fat[sector] = order[k + 1] if k + 1 < count else ENDOFCHAIN
        return order[0]

    ministream, minifat, entries = b"", [], []
    for name, data in streams.items():
        if len(data) == 0:
            entries.append((name, 2, ENDOFCHAIN, 0))
        elif len(data) < CUTOFF:
            first = len(minifat)
            count = -(-len(data) // MINI)
            minifat.extend(list(range(first + 1, first + count)) + [ENDOFCHAIN])
            ministream += data.ljust(count * MINI, b"\x00")
            entries.append((name, 2, first, len(data)))
        else:
            entries.append((name, 2, allocate(data, name in reverse), len(data)))

    mini_start = allocate(ministream) if ministream else ENDOFCHAIN
    minifat_start, n_minifat = ENDOFCHAIN, 0
    if minifat:
        raw = np.array(minifat, dtype="<u4").tobytes()
        raw += struct.pack("<I", FREESECT) * (-len(minifat) % (SECTOR // 4))
        minifat_start, n_minifat = allocate(raw), len(raw) // SECTOR

    def entry(name, kind, start, size, left=FREESECT, right=FREESECT, child=FREESECT):
        raw = (name + "\x00").encode("utf-16-le")
        return (raw.ljust(64, b"\x00") + struct.pack("<HBB", len(raw), kind, 1)
                + struct.pack("<III", left, right, child) + b"\x00" * 36 + struct.pack("<IQ", start, size))

    directory = entry("Root Entry", 5, mini_start, len(ministream), child=1)
    directory += entry(STORAGE_NAME, 1, 0, 0, child=2 if entries else FREESECT)
    for k, (name, kind, start, size) in enumerate(entries):
        directory += entry(name, kind, start, size, right=k + 3 if k + 1 < len(entries) else FREESECT)
    directory += (b"\x00" * 68 + struct.pack("<III", FREESECT, FREESECT, FREESECT) + b"\x00" * 48) * (-(len(directory) // 128) % 4)
    dir_start = allocate(directory)

    per = SECTOR // 4
    n_fat = 1
    while True:
        n_difat = max(-(-(n_fat - 109) // (per - 1)), 0)
        if (len(sectors) + n_fat + n_difat) <= n_fat * per:
            break
        n_fat += 1
    fat_ids = list(range(len(sectors), len(sectors) + n_fat))
    difat_ids = list(range(fat_ids[-1] + 1, fat_ids[-1] + 1 + n_difat))
    table = fat + [FATSECT] * n_fat + [DIFSECT] * n_difat
    table += [FREESECT] * (n_fat * per - len(table))
    raw_fat = np.array(table, dtype="<u4").tobytes()
    sectors.extend(raw_fat[k * SECTOR:(k + 1) * SECTOR] for k in range(n_fat))
    extra = fat_ids[109:]
    for k, sector in enumerate(difat_ids):
        chunk = extra[k * (per - 1):(k + 1) * (per - 1)]
        chunk += [FREESECT] * (per - 1 - len(chunk))
        chunk.append(difat_ids[k + 1] if k + 1 < n_difat else ENDOFCHAIN)
        sectors.append(np.array(chunk, dtype="<u4").tobytes())

    header = bytes.fromhex("d0cf11e0a1b11ae1") + b"\x00" * 16
    header += struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\x00" * 6
    header += struct.pack("<9I", 0, n_fat, dir_start, 0, CUTOFF, minifat_start, n_minifat,
                          difat_ids[0] if difat_ids else ENDOFCHAIN, n_difat)
    head_difat = fat_ids[:109] + [FREESECT] * (109 - len(fat_ids[:109]))
    header += np.array(head_difat, dtype="<u4").tobytes()
    with open(path, "wb") as handle:
        handle.write(header)
        for sector in sectors:
            handle.write(sector)


def write_spp(path, spectra, times, start=10000.0, step=-1.0, allocated=None, collected=None,
              reverse_data=False, units=("cm-1", "Secs", "Abs")):
    """
    `spectra` are rows in the order the instrument stores them (first column at `start`).
    Rows beyond `collected` up to `allocated` are written as zeros, as the instrument does.
    """
    spectra = np.atleast_2d(np.asarray(spectra, dtype="<f8"))
    times = np.asarray(times, dtype="<f8")
    n, points = spectra.shape
    collected = n if collected is None else collected
    allocated = max(n, collected) if allocated is None else allocated
    data = np.zeros((allocated, points), dtype="<f8")
    data[:n] = spectra
    clock = np.zeros(allocated, dtype="<f8")
    clock[:n] = times
    props = [
        (1, "i2", 1200), (4, "str", r"C:\Users\Administrador\Desktop\teste.spp"),
        (6, "str", units[0]), (7, "str", units[1]), (9, "i4", points), (10, "i4", points),
        (11, "r8", step), (12, "r8", start), (14, "i4", allocated), (15, "i4", collected), (16, "r8", 0.0),
        (17, "str", units[2]), (18, "stream", "prop18"), (19, "stream", "prop19"), (61, "i4", 1),
    ]
    streams = {
        "prop18": data.tobytes(), "prop19": clock.tobytes(), "prop39": b"\x01" + b"\x00" * 357,
        "prop44": b"", "CONTENTS": _property_set(props),
    }
    write_compound_file(path, streams, reverse=("prop18",) if reverse_data else ())
