"""Minimal OLE2 / CFB reader for classic SolidWorks files."""

from __future__ import annotations

import struct

OLE_MAGIC = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])
END = 0xFFFFFFFE


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _i32(data: bytes, off: int) -> int:
    return struct.unpack_from("<i", data, off)[0]


def is_ole2(data: bytes) -> bool:
    return len(data) >= 8 and data[:8] == OLE_MAGIC


def _utf16z(buf: bytes, max_chars: int) -> str:
    out: list[str] = []
    n = min(max_chars, len(buf) // 2)
    for i in range(n):
        c = struct.unpack_from("<H", buf, i * 2)[0]
        if c == 0:
            break
        out.append(chr(c))
    return "".join(out)


class _DirEntry:
    __slots__ = ("name", "type", "child", "left", "right", "start", "size")

    def __init__(self, name: str, typ: int, left: int, right: int, child: int, start: int, size: int) -> None:
        self.name = name
        self.type = typ
        self.left = left
        self.right = right
        self.child = child
        self.start = start
        self.size = size


def parse_ole2(data: bytes) -> dict[str, bytes]:
    streams: dict[str, bytes] = {}
    if not is_ole2(data) or len(data) < 512:
        return streams

    sector_shift = _u16(data, 0x1E) or 9
    sector_size = 1 << sector_shift
    mini_shift = _u16(data, 0x20) or 6
    mini_size = 1 << mini_shift
    mini_cutoff = _u32(data, 0x38) or 4096
    dir_start = _u32(data, 0x30)
    mini_fat_start = _u32(data, 0x3C)
    difat_start = _u32(data, 0x44)
    difat_count = _u32(data, 0x48)

    def sector(i: int) -> bytes:
        off = 512 + i * sector_size
        return data[off : off + sector_size]

    fat: list[int] = []

    def push_fat_sector(sec: int) -> None:
        if sec >= END:
            return
        bytes_ = sector(sec)
        for i in range(0, sector_size, 4):
            fat.append(struct.unpack_from("<I", bytes_, i)[0])

    for i in range(109):
        s = _u32(data, 0x4C + i * 4)
        if s < END:
            push_fat_sector(s)
    dif = difat_start
    n = 0
    while n < difat_count and dif < END:
        bytes_ = sector(dif)
        entries = sector_size // 4 - 1
        for i in range(entries):
            s = struct.unpack_from("<I", bytes_, i * 4)[0]
            if s < END:
                push_fat_sector(s)
        dif = struct.unpack_from("<I", bytes_, entries * 4)[0]
        n += 1

    def read_chain(start: int, size: int, use_mini: bool, mini_fat: list[int], mini_stream: bytes) -> bytes:
        out = bytearray(size)
        written = 0
        sec = start
        steps = 0
        while sec < END and written < size and steps < 1_000_000:
            steps += 1
            if use_mini:
                off = sec * mini_size
                chunk = mini_stream[off : off + mini_size]
                ncopy = min(mini_size, size - written)
                out[written : written + ncopy] = chunk[:ncopy]
                written += ncopy
                sec = mini_fat[sec] if sec < len(mini_fat) else END
            else:
                chunk = sector(sec)
                ncopy = min(sector_size, size - written)
                out[written : written + ncopy] = chunk[:ncopy]
                written += ncopy
                sec = fat[sec] if sec < len(fat) else END
        return bytes(out)

    dir_sectors: list[bytes] = []
    dsec = dir_start
    hops = 0
    while dsec < END and hops < 10_000:
        dir_sectors.append(sector(dsec))
        dsec = fat[dsec] if dsec < len(fat) else END
        hops += 1
    dir_buf = b"".join(dir_sectors)
    entries: list[_DirEntry] = []
    for i in range(0, len(dir_buf) - 127, 128):
        name_len = struct.unpack_from("<H", dir_buf, i + 64)[0]
        name = _utf16z(dir_buf[i : i + 64], name_len // 2)
        entries.append(
            _DirEntry(
                name=name,
                typ=dir_buf[i + 66],
                left=_i32(dir_buf, i + 68),
                right=_i32(dir_buf, i + 72),
                child=_i32(dir_buf, i + 76),
                start=_u32(dir_buf, i + 116),
                size=_u32(dir_buf, i + 120),
            )
        )

    mini_stream = b""
    mini_fat: list[int] = []
    root = entries[0] if entries else None
    if root and root.size > 0:
        mini_stream = read_chain(root.start, root.size, False, [], b"")
        m = mini_fat_start
        mh = 0
        while m < END and mh < 10_000:
            bytes_ = sector(m)
            for i in range(0, sector_size, 4):
                mini_fat.append(struct.unpack_from("<I", bytes_, i)[0])
            m = fat[m] if m < len(fat) else END
            mh += 1

    def walk(idx: int, prefix: str) -> None:
        if idx < 0 or idx >= len(entries):
            return
        e = entries[idx]
        if e.left >= 0:
            walk(e.left, prefix)
        if e.right >= 0:
            walk(e.right, prefix)
        path = f"{prefix}/{e.name}" if prefix else e.name
        if e.type == 1 and e.child >= 0:
            walk(e.child, path)
        elif e.type == 2 and e.size > 0 and e.name:
            use_mini = e.size < mini_cutoff
            raw = read_chain(e.start, e.size, use_mini, mini_fat, mini_stream)
            streams[path.replace("Root Entry/", "", 1)] = raw
        if e.type == 5 and e.child >= 0:
            walk(e.child, "")

    if entries:
        walk(0, "")
    else:
        for e in entries:
            if e.type == 2 and e.size > 0 and e.name:
                use_mini = e.size < mini_cutoff
                streams[e.name] = read_chain(e.start, e.size, use_mini, mini_fat, mini_stream)
    return streams
