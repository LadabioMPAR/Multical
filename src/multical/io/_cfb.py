"""
Minimal read-only parser for OLE compound files (the container used by .spp).

Only what the .spp reader needs: find a stream by name and read a byte range of
it. Every pointer read from the file is checked, so a damaged file gives a
`CompoundFileError` instead of a wrong result or an endless loop.
"""
import os
import struct

import numpy as np

SIGNATURE = bytes.fromhex("d0cf11e0a1b11ae1")
ENDOFCHAIN = 0xFFFFFFFE
FREESECT = 0xFFFFFFFF
_STREAM, _ROOT = 2, 5
_HEADER_DIFAT = 109


class CompoundFileError(ValueError):
    pass


class CompoundFile:
    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        try:
            self._parse()
        except Exception:
            self._file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()

    # ---- structure -------------------------------------------------------
    def _parse(self):
        head = self._file.read(512)
        if len(head) < 512 or head[:8] != SIGNATURE:
            raise CompoundFileError("não é um arquivo composto OLE")
        major, _, sector_shift, mini_shift = struct.unpack_from("<HHHH", head, 26)
        if major not in (3, 4) or sector_shift not in (9, 12) or mini_shift != 6:
            raise CompoundFileError("versão de arquivo composto não suportada")
        self._ss, self._ms = 1 << sector_shift, 1 << mini_shift
        (_, n_fat, dir_start, _, self._cutoff, mini_start, n_mini, difat_start, n_difat) = struct.unpack_from("<9I", head, 40)
        self._n_sectors = max(os.path.getsize(self.path) // self._ss - 1, 0)

        per_sector = self._ss // 4
        difat = list(struct.unpack_from(f"<{_HEADER_DIFAT}I", head, 76))
        sector = difat_start
        for _ in range(n_difat):
            if sector in (ENDOFCHAIN, FREESECT):
                break
            block = np.frombuffer(self._sector(sector), dtype="<u4")
            difat.extend(block[:per_sector - 1].tolist())
            sector = int(block[per_sector - 1])
        fat_sectors = [s for s in difat if s != FREESECT][:n_fat]
        if len(fat_sectors) != n_fat:
            raise CompoundFileError("tabela de alocação incompleta")
        self._fat = np.concatenate(
            [np.frombuffer(self._sector(s), dtype="<u4") for s in fat_sectors]
        ).tolist() if fat_sectors else []

        directory = self._read_sectors(self._chain(dir_start))
        self._streams, self._root = {}, None
        for pos in range(0, len(directory) - 127, 128):
            entry = directory[pos:pos + 128]
            name_len = struct.unpack_from("<H", entry, 64)[0]
            kind = entry[66]
            start, size = struct.unpack_from("<IQ", entry, 116)
            if major == 3:
                size &= 0xFFFFFFFF
            if kind == _ROOT and self._root is None:
                self._root = (start, size)
            elif kind == _STREAM and 2 <= name_len <= 64:
                name = bytes(entry[:name_len - 2]).decode("utf-16-le", "replace")
                self._streams.setdefault(name, (start, size))
        if self._root is None:
            raise CompoundFileError("diretório sem entrada raiz")

        self._minifat = []
        if n_mini and mini_start not in (ENDOFCHAIN, FREESECT):
            self._minifat = np.frombuffer(self._read_sectors(self._chain(mini_start)), dtype="<u4").tolist()
        self._ministream = None
        self._chains = {}

    def _sector(self, number):
        if not 0 <= number < self._n_sectors:
            raise CompoundFileError("o arquivo está truncado ou corrompido")
        self._file.seek((number + 1) * self._ss)
        data = self._file.read(self._ss)
        if len(data) != self._ss:
            raise CompoundFileError("o arquivo está truncado ou corrompido")
        return data

    def _chain(self, start):
        """Sector numbers of a stream, in order."""
        chain, sector, fat = [], start, self._fat
        while sector != ENDOFCHAIN:
            if not 0 <= sector < len(fat) or sector >= self._n_sectors or len(chain) > len(fat):
                raise CompoundFileError("o arquivo está truncado ou corrompido")
            chain.append(sector)
            sector = fat[sector]
        return np.asarray(chain, dtype=np.int64)

    def _read_sectors(self, sectors):
        """Bytes of the given sectors, read with one disk access per contiguous run."""
        sectors = np.asarray(sectors, dtype=np.int64)
        buffer = bytearray(sectors.size * self._ss)
        if sectors.size == 0:
            return memoryview(buffer)
        view = memoryview(buffer)
        breaks = np.flatnonzero(np.diff(sectors) != 1) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [sectors.size]))
        for a, b in zip(starts.tolist(), ends.tolist()):
            self._file.seek((int(sectors[a]) + 1) * self._ss)
            if self._file.readinto(view[a * self._ss:b * self._ss]) != (b - a) * self._ss:
                raise CompoundFileError("o arquivo está truncado ou corrompido")
        return view

    # ---- streams ---------------------------------------------------------
    def has_stream(self, name):
        return name in self._streams

    def stream_size(self, name):
        return self._streams[name][1]

    def read(self, name, offset=0, size=None):
        """Bytes [offset, offset + size) of the stream `name` (a writable buffer)."""
        if name not in self._streams:
            raise CompoundFileError(f"o arquivo não tem o bloco '{name}'")
        start, total = self._streams[name]
        if size is None:
            size = total - offset
        if offset < 0 or size < 0 or offset + size > total:
            raise CompoundFileError(f"o bloco '{name}' é menor do que o esperado")
        if size == 0:
            return memoryview(bytearray())
        if total < self._cutoff:
            return self._read_mini(start, total)[offset:offset + size]
        if name not in self._chains:
            self._chains[name] = self._chain(start)
        chain = self._chains[name]
        first, last = offset // self._ss, (offset + size - 1) // self._ss
        if last >= chain.size:
            raise CompoundFileError(f"o bloco '{name}' é menor do que o esperado")
        skip = offset - first * self._ss
        return self._read_sectors(chain[first:last + 1])[skip:skip + size]

    def _read_mini(self, start, total):
        """Small streams live in 64-byte pieces inside a stream owned by the root entry."""
        if self._ministream is None:
            root_start, root_size = self._root
            self._ministream = self._read_sectors(self._chain(root_start))[:root_size]
        out, sector, steps = bytearray(), start, 0
        while sector != ENDOFCHAIN and len(out) < total:
            if not 0 <= sector < len(self._minifat) or steps > len(self._minifat):
                raise CompoundFileError("o arquivo está truncado ou corrompido")
            piece = self._ministream[sector * self._ms:(sector + 1) * self._ms]
            if len(piece) != self._ms:
                raise CompoundFileError("o arquivo está truncado ou corrompido")
            out += piece
            sector = self._minifat[sector]
            steps += 1
        if len(out) < total:
            raise CompoundFileError("o arquivo está truncado ou corrompido")
        return memoryview(out)[:total]
