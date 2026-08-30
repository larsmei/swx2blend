"""Raw deflate / SolidWorks ZLB helpers."""

from __future__ import annotations

import zlib

__all__ = ["inflate_raw", "inflate_zlb", "maybe_decompress_stream", "index_of_seq", "index_of_ascii", "index_of_utf16"]


def inflate_raw(data: bytes | memoryview | bytearray) -> bytes | None:
    if len(data) == 0:
        return b""
    buf = bytes(data)
    try:
        return zlib.decompress(buf, -15)
    except zlib.error:
        try:
            return zlib.decompress(buf)
        except zlib.error:
            return None


def inflate_zlb(block: bytes | memoryview | bytearray) -> bytes | None:
    if len(block) < 24:
        return None
    csz = int.from_bytes(block[20:24], "little")
    if 24 + csz > len(block):
        return None
    return inflate_raw(block[24 : 24 + csz])


def maybe_decompress_stream(name: str, data: bytes) -> bytes:
    lower = name.lower()
    if lower.endswith("__zlb") or lower.endswith("_zlb"):
        return inflate_zlb(data) or data
    if lower.endswith("__zip") or lower.endswith("_zip"):
        return inflate_raw(data) or data
    return data


def index_of_seq(data: bytes | memoryview | bytearray, needle: bytes, from_: int = 0) -> int:
    if not needle:
        return from_
    return bytes(data).find(needle, from_)


def index_of_ascii(data: bytes | memoryview | bytearray, text: str, from_: int = 0) -> int:
    return index_of_seq(data, text.encode("ascii"), from_)


def index_of_utf16(data: bytes | memoryview | bytearray, text: str, from_: int = 0) -> int:
    needle = text.encode("utf-16le")
    return index_of_seq(data, needle, from_)
