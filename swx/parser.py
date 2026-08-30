"""SolidWorks container parser: OLE2 (classic) and modern SWX."""

from __future__ import annotations

from .inflate import inflate_raw, maybe_decompress_stream
from .ole2 import is_ole2, parse_ole2

MARKER = bytes([0x14, 0x00, 0x06, 0x00, 0x08, 0x00])
CHUNK_HEADER = 0x1E
INLINE_F1 = 65536


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def _rol_byte(b: int, shift: int) -> int:
    shift &= 7
    if shift == 0:
        return b
    return ((b << shift) | (b >> (8 - shift))) & 0xFF


def _rol_decode(raw: bytes, key: int) -> str:
    return "".join(chr(_rol_byte(b, key)) for b in raw)


def _is_valid_name(name: str) -> bool:
    if not name:
        return False
    return all(0x20 <= ord(c) < 0x80 for c in name)


def parse_modern_swx(data: bytes) -> dict[str, bytes]:
    streams: dict[str, bytes] = {}
    if len(data) < 8:
        return streams
    key = data[7]
    search = 0
    while True:
        marker_pos = data.find(MARKER, search)
        if marker_pos < 0:
            break
        if marker_pos < 4:
            search = marker_pos + 1
            continue
        si = marker_pos - 4
        if si + CHUNK_HEADER > len(data):
            search = marker_pos + 1
            continue
        f1 = _u32(data, si + 0x0E)
        csz = _u32(data, si + 0x12)
        nsz = _u32(data, si + 0x1A)
        if nsz > 512 or csz > 64 * 1024 * 1024:
            search = marker_pos + 1
            continue
        name_start = si + CHUNK_HEADER
        name_end = name_start + nsz
        if name_end > len(data):
            search = marker_pos + 1
            continue
        name = _rol_decode(data[name_start:name_end], key)
        if not _is_valid_name(name):
            search = marker_pos + 1
            continue
        is_inline = f1 >= INLINE_F1
        if is_inline and csz > 0:
            data_end = name_end + csz
            if data_end <= len(data):
                inflated = inflate_raw(data[name_end:data_end])
                if inflated is not None and name not in streams:
                    streams[name] = inflated
                search = data_end
                continue
        elif is_inline and csz == 0:
            streams.setdefault(name, b"")
        search = marker_pos + 6
    return streams


def parse_solidworks_file(data: bytes) -> dict[str, bytes]:
    if is_ole2(data):
        raw = parse_ole2(data)
        out: dict[str, bytes] = {}
        for name, blob in raw.items():
            out[name.replace("\\", "/")] = maybe_decompress_stream(name, blob)
        return out
    return parse_modern_swx(data)


def find_display_list_stream(streams: dict[str, bytes]) -> bytes | None:
    preferred = [
        "Contents/DisplayLists",
        "Contents/DisplayLists__ZLB",
        "DisplayLists",
        "DisplayLists__ZLB",
    ]
    for key in preferred:
        hit = streams.get(key)
        if hit and len(hit) > 64:
            return hit
    for name, blob in streams.items():
        if "displaylist" in name.lower() and len(blob) > 64:
            return blob
    largest: bytes | None = None
    for name, blob in streams.items():
        low = name.lower()
        if any(tok in low for tok in ("preview", "png", "xml", "props")):
            continue
        if largest is None or len(blob) > len(largest):
            largest = blob
    return largest
