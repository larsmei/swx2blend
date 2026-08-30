"""Display-list triangle-strip tessellation from SolidWorks streams."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .assembly import (
    Aabb,
    AssemblyInstance,
    assembly_from_streams,
    find_instance_name_markers,
    invert_rigid,
    match_instance,
    part_name_from_instance_path,
)
from .geometry import (
    DEFAULT_CAD_COLOR,
    ExtractedMesh,
    bounding_box,
    merge_meshes,
    paint_mesh,
    split_connected_bodies,
    stamp_mesh,
    transform_mesh,
)
from .inflate import index_of_ascii, index_of_seq

FLT_HDR = bytes([0x0C, 0, 0, 0, 0x64, 0, 0, 0, 0x02, 0, 0, 0])
INT_HDR = bytes([0x04, 0, 0, 0, 0x08, 0, 0, 0, 0x02, 0, 0, 0])


@dataclass
class SwAppearance:
    color: tuple[float, float, float]
    metalness: float
    roughness: float
    name: str
    off: int | None = None


@dataclass
class FaceTess:
    positions: np.ndarray
    normals: np.ndarray | None
    strips: list[int]
    next: int
    off: int


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _hdr_at(data: bytes, off: int, hdr: bytes) -> bool:
    return off + len(hdr) <= len(data) and data[off : off + len(hdr)] == hdr


def _find_ascii(data: bytes, text: str, from_: int = 0) -> list[int]:
    hits: list[int] = []
    needle = text.encode("ascii")
    i = from_
    while True:
        hit = data.find(needle, i)
        if hit < 0:
            break
        hits.append(hit)
        i = hit + 1
    return hits


def _is_finite_vec(x: float, y: float, z: float) -> bool:
    return math.isfinite(x) and math.isfinite(y) and math.isfinite(z)


def _is_unit_normal(x: float, y: float, z: float) -> bool:
    m = math.hypot(x, y, z)
    return m > 0.5 and abs(m - 1) < 0.12


def _read_float3_block(data: bytes, off: int) -> tuple[int, np.ndarray, int] | None:
    if not _hdr_at(data, off, FLT_HDR) or off + 16 > len(data):
        return None
    count = _u32(data, off + 12)
    data_off = off + 16
    if count < 1 or count > 4_000_000 or data_off + count * 12 > len(data):
        return None
    src = data[data_off : data_off + count * 12]
    values = np.frombuffer(src, dtype="<f4").copy()
    last = values.size - 3
    if last >= 0 and (
        not math.isfinite(float(values[0]))
        or not math.isfinite(float(values[2]))
        or not math.isfinite(float(values[last]))
        or not math.isfinite(float(values[last + 2]))
    ):
        return None
    return count, values, data_off + count * 12


def _looks_like_normals(values: np.ndarray) -> bool:
    n = values.size // 3
    if n < 1:
        return False
    probe = min(n, 12)
    hits = 0
    for i in range(probe):
        if _is_unit_normal(float(values[i * 3]), float(values[i * 3 + 1]), float(values[i * 3 + 2])):
            hits += 1
    return hits >= max(3, probe - 1) or (n <= 4 and hits == n)


def _try_parse_face(data: bytes, int_off: int) -> FaceTess | None:
    if not _hdr_at(data, int_off, INT_HDR) or int_off + 16 > len(data):
        return None
    nstrips = _u32(data, int_off + 12)
    if nstrips < 1 or nstrips > 500_000:
        return None
    len_off = int_off + 16
    if len_off + nstrips * 4 > len(data):
        return None
    strips: list[int] = []
    total = 0
    for i in range(nstrips):
        n = _u32(data, len_off + i * 4)
        if n < 1 or n > 4_000_000:
            return None
        strips.append(n)
        total += n
    if total < 3 or total > 4_000_000:
        return None
    flt_off = len_off + nstrips * 4
    pad_limit = flt_off + 8
    while flt_off <= pad_limit and flt_off + 16 <= len(data) and not _hdr_at(data, flt_off, FLT_HDR):
        flt_off += 1
    verts = _read_float3_block(data, flt_off)
    if verts is None or verts[0] != total:
        return None
    count, values, next_off = verts
    if _looks_like_normals(values):
        return None
    normals = None
    nrm = _read_float3_block(data, next_off)
    if nrm is not None and nrm[0] == count and _looks_like_normals(nrm[1]):
        normals = nrm[1]
        next_off = nrm[2]
    return FaceTess(positions=values, normals=normals, strips=strips, next=next_off, off=int_off)


def _triangle_area(p: np.ndarray, ia: int, ib: int, ic: int) -> float:
    ax = float(p[ib * 3] - p[ia * 3])
    ay = float(p[ib * 3 + 1] - p[ia * 3 + 1])
    az = float(p[ib * 3 + 2] - p[ia * 3 + 2])
    bx = float(p[ic * 3] - p[ia * 3])
    by = float(p[ic * 3 + 1] - p[ia * 3 + 1])
    bz = float(p[ic * 3 + 2] - p[ia * 3 + 2])
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * math.hypot(cx, cy, cz)


def _push_tri(
    out: np.ndarray,
    w: list[int],
    p: np.ndarray,
    n: np.ndarray | None,
    ia: int,
    ib: int,
    ic: int,
) -> None:
    if ia == ib or ib == ic or ia == ic:
        return
    if _triangle_area(p, ia, ib, ic) < 1e-18:
        return
    a, b, c = ia, ib, ic
    if n is not None and n.size == p.size:
        ax = float(p[b * 3] - p[a * 3])
        ay = float(p[b * 3 + 1] - p[a * 3 + 1])
        az = float(p[b * 3 + 2] - p[a * 3 + 2])
        bx = float(p[c * 3] - p[a * 3])
        by = float(p[c * 3 + 1] - p[a * 3 + 1])
        bz = float(p[c * 3 + 2] - p[a * 3 + 2])
        gx = ay * bz - az * by
        gy = az * bx - ax * bz
        gz = ax * by - ay * bx
        nx = float(n[a * 3] + n[b * 3] + n[c * 3])
        ny = float(n[a * 3 + 1] + n[b * 3 + 1] + n[c * 3 + 1])
        nz = float(n[a * 3 + 2] + n[b * 3 + 2] + n[c * 3 + 2])
        if gx * nx + gy * ny + gz * nz < 0:
            b, c = c, b
    i = w[0]
    out[i] = a
    out[i + 1] = b
    out[i + 2] = c
    w[0] = i + 3


def triangulate_strips(positions: np.ndarray, normals: np.ndarray | None, strips: list[int]) -> np.ndarray:
    cap = sum((sl - 2) * 3 for sl in strips if sl >= 3)
    out = np.empty(cap, dtype=np.uint32)
    w = [0]
    vi = 0
    for sl in strips:
        if sl < 3:
            vi += sl
            continue
        for t in range(sl - 2):
            if t % 2 == 0:
                _push_tri(out, w, positions, normals, vi + t, vi + t + 1, vi + t + 2)
            else:
                _push_tri(out, w, positions, normals, vi + t + 1, vi + t, vi + t + 2)
        vi += sl
    return out if w[0] == cap else out[: w[0]]


def triangulate_face(positions: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    n = positions.size // 3
    if n < 3:
        return np.zeros(0, dtype=np.uint32)
    if n == 3:
        return np.array([0, 1, 2], dtype=np.uint32)
    if n == 4:
        splits = (
            (0, 1, 2, 1, 3, 2),
            (0, 1, 3, 0, 3, 2),
            (0, 1, 2, 0, 2, 3),
        )
        best = splits[0]
        best_score = math.inf
        for s in splits:
            a1 = _triangle_area(positions, s[0], s[1], s[2])
            a2 = _triangle_area(positions, s[3], s[4], s[5])
            if a1 < 1e-18 or a2 < 1e-18:
                continue
            score = abs(a1 - a2) / (a1 + a2)
            if score < best_score:
                best_score = score
                best = s
        return np.array(best, dtype=np.uint32)
    fan = np.empty((n - 2) * 3, dtype=np.uint32)
    for i in range(1, n - 1):
        fan[(i - 1) * 3 : i * 3] = (0, i, i + 1)
    return fan


def collect_strip_faces(data: bytes, from_: int = 0, until: int | None = None) -> list[FaceTess]:
    if until is None:
        until = len(data)
    faces: list[FaceTess] = []
    search = from_
    while search < until:
        int_off = index_of_seq(data, INT_HDR, search)
        if int_off < 0 or int_off >= until:
            break
        face = _try_parse_face(data, int_off)
        if face:
            faces.append(face)
            search = face.next
        else:
            search = int_off + 4
    return faces


@dataclass
class _RawFace:
    positions: np.ndarray
    normals: np.ndarray | None
    off: int


def collect_raw_vertex_blocks(data: bytes, from_: int = 0, until: int | None = None) -> list[_RawFace]:
    if until is None:
        until = len(data)
    faces: list[_RawFace] = []
    search = from_
    while search < until:
        off = index_of_seq(data, FLT_HDR, search)
        if off < 0 or off >= until:
            break
        block = _read_float3_block(data, off)
        if block is None or block[0] < 3 or _looks_like_normals(block[1]):
            search = off + 4
            continue
        count, values, next_off = block
        normals = None
        nrm = _read_float3_block(data, next_off)
        if nrm is not None and nrm[0] == count and _looks_like_normals(nrm[1]):
            normals = nrm[1]
        faces.append(_RawFace(positions=values, normals=normals, off=off))
        search = next_off
    return faces


def _merge_faces(
    faces: Sequence[tuple[np.ndarray, np.ndarray | None, np.ndarray, tuple[float, float, float] | None]],
    name: str,
) -> ExtractedMesh | None:
    vert_count = 0
    index_total = 0
    has_c = False
    for pos, _n, idx, color in faces:
        vert_count += pos.size // 3
        index_total += idx.size
        if color:
            has_c = True
    if vert_count < 3 or index_total < 3:
        return None
    positions = np.empty(vert_count * 3, dtype=np.float32)
    normals = np.zeros(vert_count * 3, dtype=np.float32)
    indices = np.empty(index_total, dtype=np.uint32)
    colors = np.empty(vert_count * 3, dtype=np.float32) if has_c else None
    v_base = 0
    i_base = 0
    has_nrm = False
    mesh_color: tuple[float, float, float] | None = None
    for pos, nrm, idx, color in faces:
        n = pos.size // 3
        positions[v_base * 3 : (v_base + n) * 3] = pos
        if nrm is not None:
            normals[v_base * 3 : (v_base + n) * 3] = nrm
            has_nrm = True
        if colors is not None:
            c = color or DEFAULT_CAD_COLOR
            if color and mesh_color is None:
                mesh_color = color
            colors[v_base * 3 : (v_base + n) * 3 : 3] = c[0]
            colors[v_base * 3 + 1 : (v_base + n) * 3 : 3] = c[1]
            colors[v_base * 3 + 2 : (v_base + n) * 3 : 3] = c[2]
        indices[i_base : i_base + idx.size] = idx + v_base
        i_base += idx.size
        v_base += n
    return ExtractedMesh(
        positions=positions,
        normals=normals if has_nrm else None,
        indices=indices,
        name=name,
        color=mesh_color,
        colors=colors,
    )


def _appearance_for_off(off: int, apps: list[SwAppearance]) -> SwAppearance | None:
    if not apps:
        return None
    located = [a for a in apps if a.off is not None]
    pool = located or apps
    prev: SwAppearance | None = None
    nxt: SwAppearance | None = None
    for a in pool:
        at = a.off or 0
        if at <= off:
            prev = a
        else:
            nxt = a
            break
    if nxt and nxt.off is not None and nxt.off - off < 80_000:
        return nxt
    return prev or nxt


def faces_to_mesh(
    data: bytes,
    from_: int,
    until: int,
    name: str,
    appearances: list[SwAppearance] | None = None,
) -> ExtractedMesh | None:
    appearances = appearances or []

    def decorate(
        faces: Sequence[FaceTess | _RawFace],
    ) -> list[tuple[np.ndarray, np.ndarray | None, int, tuple[float, float, float] | None]]:
        out = []
        for face in faces:
            look = _appearance_for_off(face.off, appearances)
            out.append((face.positions, face.normals, face.off, look.color if look else None))
        return out

    strip_faces = collect_strip_faces(data, from_, until)
    if strip_faces:
        decorated = decorate(strip_faces)
        built = []
        for (pos, nrm, _off, color), src in zip(decorated, strip_faces):
            built.append((pos, nrm, triangulate_strips(src.positions, src.normals, src.strips), color))
        mesh = _merge_faces(built, name)
        if mesh and mesh.indices.size >= 3:
            look = _appearance_for_off(from_, appearances) or _appearance_for_off(until, appearances)
            if look:
                mesh.metalness = look.metalness
                mesh.roughness = look.roughness
            return mesh

    raw = collect_raw_vertex_blocks(data, from_, until)
    if not raw:
        return None
    decorated_raw = decorate(raw)
    built = []
    for (pos, nrm, _off, color), src in zip(decorated_raw, raw):
        n = src.positions.size // 3
        strip_idx = triangulate_strips(src.positions, src.normals, [n])
        degenerates = max(0, n - 2 - strip_idx.size // 3)
        indices = strip_idx
        if n >= 4 and degenerates > (n - 2) * 0.35:
            planar = triangulate_face(src.positions, src.normals)
            if planar.size > strip_idx.size:
                indices = planar
        built.append((pos, nrm, indices, color))
    mesh = _merge_faces(built, name)
    if mesh:
        look = _appearance_for_off(from_, appearances)
        if look:
            mesh.metalness = look.metalness
            mesh.roughness = look.roughness
    return mesh


def _clamp01(n: float) -> float:
    return min(1.0, max(0.0, n))


def _parse_utf16z(data: bytes, off: int, max_chars: int = 64) -> str:
    out: list[str] = []
    for i in range(max_chars):
        p = off + i * 2
        if p + 1 >= len(data):
            break
        c = data[p] | (data[p + 1] << 8)
        if c == 0 or c < 32 or c > 126:
            break
        out.append(chr(c))
    return "".join(out)


def parse_visual_properties(data: bytes) -> list[SwAppearance]:
    hits = _find_ascii(data, "moVisualProperties_c")
    out: list[SwAppearance] = []
    for hit in hits:
        start = hit + len("moVisualProperties_c")
        end = min(len(data), start + 240)
        color: tuple[float, float, float] | None = None
        name = ""
        for i in range(start, min(start + 16, end) - 3):
            if data[i + 3] == 0 and (data[i] > 8 or data[i + 1] > 8 or data[i + 2] > 8):
                r, g, b = data[i] / 255.0, data[i + 1] / 255.0, data[i + 2] / 255.0
                if r + g + b > 0.04:
                    color = (r, g, b)
                    break
        i = start
        while i + 4 <= end:
            if data[i] == 0xFF and data[i + 1] == 0xFE and data[i + 2] == 0xFF:
                nchars = data[i + 3]
                if 3 <= nchars <= 40:
                    s = _parse_utf16z(data, i + 4, nchars)
                    if len(s) >= 3 and not name:
                        name = s
            i += 1
        specularity = 0.35
        shininess = 0.4
        i = start
        while i + 32 <= end:
            a, d, s, sh = struct.unpack_from("<dddd", data, i)
            if 0 <= a <= 1.01 and 0 <= d <= 1.01 and 0 <= s <= 1.01 and 0 <= sh <= 1.01 and (a > 0.2 or d > 0.2):
                specularity = s
                shininess = sh
                break
            i += 8
        if not color:
            continue
        out.append(
            SwAppearance(
                color=color,
                name=name,
                off=hit,
                metalness=_clamp01(0.18 + specularity * 0.7),
                roughness=_clamp01(0.12 + (1 - shininess) * 0.7),
            )
        )
    return out


def _look_from_name(name: str, rgb: tuple[float, float, float]) -> SwAppearance:
    n = name.lower()
    color = rgb
    metalness = 0.08
    roughness = 0.46
    if "glass" in n:
        if rgb[0] > 0.82 and rgb[1] > 0.82 and rgb[2] > 0.82:
            color = (0.40, 0.68, 0.90)
        metalness = 0.04
        roughness = 0.08
    elif any(tok in n for tok in ("zinc", "chrome", "steel", "brass", "copper", "aluminium", "aluminum", "iron", "metal")):
        metalness = 0.78
        roughness = 0.30
    elif "black" in n:
        metalness = 0.04
        roughness = 0.55
    return SwAppearance(color=color, metalness=metalness, roughness=roughness, name=name)


def parse_face_appearances(data: bytes) -> list[SwAppearance]:
    out: list[SwAppearance] = []
    i = 0
    limit = len(data) - 10
    while i <= limit:
        hit = data.find(0xFF, i)
        if hit < 0 or hit > limit:
            break
        if hit + 2 >= len(data) or data[hit + 1] != 0xFE or data[hit + 2] != 0xFF:
            i = hit + 1
            continue
        i = hit
        nchars = data[i + 3]
        if nchars < 4 or nchars > 48:
            i = hit + 1
            continue
        name_chars: list[str] = []
        ok = True
        for k in range(nchars):
            p = i + 4 + k * 2
            if p + 1 >= len(data):
                ok = False
                break
            c = data[p] | (data[p + 1] << 8)
            if c < 32 or c > 126:
                ok = False
                break
            name_chars.append(chr(c))
        if not ok:
            i = hit + 1
            continue
        name = "".join(name_chars)
        if not name or not name[0].isalpha():
            i = hit + 1
            continue
        if "\\" in name or ".p2m" in name or "@" in name:
            i = hit + 1
            continue
        after = i + 4 + nchars * 2
        if after + 5 >= len(data):
            i = hit + 1
            continue
        tag = data[after]
        if tag not in (2, 3, 5):
            i = hit + 1
            continue
        r, g, b, a = data[after + 1], data[after + 2], data[after + 3], data[after + 4]
        if a != 0 or r + g + b < 12:
            i = hit + 1
            continue
        look = _look_from_name(name, (r / 255.0, g / 255.0, b / 255.0))
        look.off = i
        out.append(look)
        i = after
    return out


def _is_orthonormal(m00: float, m01: float, m02: float, m10: float, m11: float, m12: float, m20: float, m21: float, m22: float) -> bool:
    c0 = math.hypot(m00, m10, m20)
    c1 = math.hypot(m01, m11, m21)
    c2 = math.hypot(m02, m12, m22)
    if abs(c0 - 1) > 0.08 or abs(c1 - 1) > 0.08 or abs(c2 - 1) > 0.08:
        return False
    d01 = m00 * m01 + m10 * m11 + m20 * m21
    d02 = m00 * m02 + m10 * m12 + m20 * m22
    d12 = m01 * m02 + m11 * m12 + m21 * m22
    return abs(d01) < 0.08 and abs(d02) < 0.08 and abs(d12) < 0.08


def _identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _is_identity(m: list[float]) -> bool:
    ident = _identity()
    return all(abs(m[i] - ident[i]) <= 1e-6 for i in range(16))


def _try_affine_at(data: bytes, off: int, as_double: bool) -> list[float] | None:
    stride = 8 if as_double else 4
    need = stride * 16
    if off < 0 or off + need > len(data):
        return None
    fmt = "<" + ("d" if as_double else "f") * 16
    try:
        v = list(struct.unpack_from(fmt, data, off))
    except struct.error:
        return None
    if not all(math.isfinite(x) for x in v):
        return None
    col = (
        _is_orthonormal(v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10])
        and abs(v[3]) < 1e-3
        and abs(v[7]) < 1e-3
        and abs(v[11]) < 1e-3
        and abs(v[15] - 1) < 0.08
    )
    if col:
        return v
    row = (
        _is_orthonormal(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10])
        and abs(v[12]) < 1e-3
        and abs(v[13]) < 1e-3
        and abs(v[14]) < 1e-3
        and abs(v[15] - 1) < 0.08
    )
    if row:
        return [
            v[0],
            v[4],
            v[8],
            0.0,
            v[1],
            v[5],
            v[9],
            0.0,
            v[2],
            v[6],
            v[10],
            0.0,
            v[3],
            v[7],
            v[11],
            1.0,
        ]
    return None


def find_affine_before(data: bytes, body_off: int) -> list[float] | None:
    from_ = max(0, body_off - 512)
    off = body_off - 16
    while off >= from_:
        d = _try_affine_at(data, off, True)
        if d and not _is_identity(d):
            return d
        f = _try_affine_at(data, off, False)
        if f and not _is_identity(f):
            return f
        off -= 4
    return None


def _aabb_close(a: Aabb, b: Aabb, abs_tol: float = 0.0012) -> bool:
    for i in range(3):
        if abs(a.min[i] - b.min[i]) > abs_tol:
            return False
        if abs(a.max[i] - b.max[i]) > abs_tol:
            return False
    return True


def _aabb_score(a: Aabb, b: Aabb) -> float:
    e = 0.0
    for i in range(3):
        e += abs(a.min[i] - b.min[i])
        e += abs(a.max[i] - b.max[i])
    return e


def _np_to_aabb(mn: np.ndarray, mx: np.ndarray) -> Aabb:
    return Aabb(min=(float(mn[0]), float(mn[1]), float(mn[2])), max=(float(mx[0]), float(mx[1]), float(mx[2])))


def _transform_aabb(box: Aabb, m: Sequence[float]) -> Aabb:
    xs = (box.min[0], box.max[0])
    ys = (box.min[1], box.max[1])
    zs = (box.min[2], box.max[2])
    mn = [math.inf, math.inf, math.inf]
    mx = [-math.inf, -math.inf, -math.inf]
    for x in xs:
        for y in ys:
            for z in zs:
                px = m[0] * x + m[4] * y + m[8] * z + m[12]
                py = m[1] * x + m[5] * y + m[9] * z + m[13]
                pz = m[2] * x + m[6] * y + m[10] * z + m[14]
                if px < mn[0]:
                    mn[0] = px
                if py < mn[1]:
                    mn[1] = py
                if pz < mn[2]:
                    mn[2] = pz
                if px > mx[0]:
                    mx[0] = px
                if py > mx[1]:
                    mx[1] = py
                if pz > mx[2]:
                    mx[2] = pz
    return Aabb(min=(mn[0], mn[1], mn[2]), max=(mx[0], mx[1], mx[2]))


def _recover_part_space(
    mesh: ExtractedMesh,
    inst: AssemblyInstance | None,
    part_bounds: dict[str, Aabb],
) -> ExtractedMesh:
    if not inst:
        return mesh
    local = part_bounds.get(inst.name)
    if not local:
        return mesh
    mn, mx = bounding_box(mesh.positions)
    bb = _np_to_aabb(mn, mx)
    world = _transform_aabb(local, inst.transform)
    if _aabb_score(bb, world) + 0.002 < _aabb_score(bb, local):
        return transform_mesh(mesh, invert_rigid(inst.transform))
    return mesh


def extract_by_instance_names(
    display_lists: bytes,
    instances: list[AssemblyInstance],
    appearances: list[SwAppearance],
    part_bounds: dict[str, Aabb],
) -> ExtractedMesh | None:
    markers = find_instance_name_markers(display_lists)
    if not markers:
        return None
    blobs: list[tuple[str, str, ExtractedMesh, AssemblyInstance | None]] = []
    for i, marker in enumerate(markers):
        start = marker.off
        end = markers[i + 1].off if i + 1 < len(markers) else len(display_lists)
        if end - start < 80:
            continue
        mesh = faces_to_mesh(display_lists, start, end, marker.path, appearances)
        if not mesh or mesh.indices.size < 3:
            continue
        inst = match_instance(marker.path, instances)
        name = inst.name if inst else part_name_from_instance_path(marker.path)
        mesh.name = name
        blobs.append((name, inst.path if inst else marker.path, mesh, inst))
    if not blobs:
        return None

    proto_by_name: dict[str, ExtractedMesh] = {}
    unused = {i.name for i in instances}

    def take_proto(name: str, mesh: ExtractedMesh, inst: AssemblyInstance | None) -> None:
        if not name or name in proto_by_name:
            return
        part = _recover_part_space(mesh, inst, part_bounds)
        part.name = name
        proto_by_name[name] = part
        unused.discard(name)

    def match_body_to_part(body: ExtractedMesh, prefer: list[str]) -> str | None:
        mn, mx = bounding_box(body.positions)
        bb = _np_to_aabb(mn, mx)

        def score_names(names: Sequence[str]) -> str | None:
            best_name: str | None = None
            best_score = math.inf
            for name in names:
                if name not in unused:
                    continue
                xml = part_bounds.get(name)
                if not xml or not _aabb_close(bb, xml):
                    continue
                score = _aabb_score(bb, xml)
                if score < best_score:
                    best_score = score
                    best_name = name
            return best_name

        return score_names(prefer) or score_names(list(unused))

    if part_bounds:
        for name, _path, mesh, inst in blobs:
            xml = part_bounds.get(name)
            if xml and name in unused and name not in proto_by_name:
                mn, mx = bounding_box(mesh.positions)
                if _aabb_close(_np_to_aabb(mn, mx), xml):
                    take_proto(name, mesh, inst)
                    continue
            bodies = split_connected_bodies(mesh)
            if not bodies:
                continue
            leftover: list[ExtractedMesh] = []
            for body in bodies:
                hit = match_body_to_part(body, [name])
                if hit:
                    take_proto(hit, body, inst)
                else:
                    leftover.append(body)
            if len(leftover) == 1 and name in unused and name not in proto_by_name:
                take_proto(name, leftover[0], inst)
                leftover = []
            if leftover:
                merged = leftover[0] if len(leftover) == 1 else merge_meshes(leftover)
                if merged:
                    hit = match_body_to_part(merged, [name, *unused])
                    if hit:
                        take_proto(hit, merged, inst)
                    elif name in unused and name not in proto_by_name:
                        take_proto(name, merged, inst)

    for name, _path, mesh, inst in blobs:
        if name not in proto_by_name:
            take_proto(name, mesh, inst)

    grouped: dict[str, list[AssemblyInstance]] = {}
    for inst in instances:
        grouped.setdefault(inst.name, []).append(inst)

    parts: list[ExtractedMesh] = []
    for name, group in grouped.items():
        proto = proto_by_name.get(name)
        if not proto:
            continue
        stamped = stamp_mesh(proto, [inst.transform for inst in group])
        stamped.name = name
        parts.append(stamped)
    if not parts:
        for _name, path, mesh, inst in blobs:
            hit = inst or match_instance(path, instances)
            parts.append(transform_mesh(mesh, hit.transform) if hit else mesh)
    return merge_meshes(parts) if parts else None


def extract_display_tessellation(
    display_lists: bytes,
    name: str = "SolidWorks",
    appearances: list[SwAppearance] | None = None,
    instance_xforms: list[list[float]] | None = None,
    instances: list[AssemblyInstance] | None = None,
    part_bounds: dict[str, Aabb] | None = None,
) -> ExtractedMesh | None:
    appearances = appearances or []
    instance_xforms = instance_xforms or []
    instances = instances or []
    part_bounds = part_bounds or {}
    local = parse_visual_properties(display_lists) + parse_face_appearances(display_lists)
    local.sort(key=lambda a: a.off or 0)
    looks = local or appearances
    if instances:
        placed = extract_by_instance_names(display_lists, instances, looks, part_bounds)
        if placed and placed.indices.size >= 3:
            return placed

    body_hits = _find_ascii(display_lists, "uoTempBodyTessData_c")
    ranges: list[tuple[int, int, list[float] | None]] = []
    if body_hits:
        for i, start in enumerate(body_hits):
            end = body_hits[i + 1] if i + 1 < len(body_hits) else len(display_lists)
            xform = find_affine_before(display_lists, start) or (instance_xforms[i] if i < len(instance_xforms) else None)
            ranges.append((start, end, xform))
    else:
        ranges.append((0, len(display_lists), instance_xforms[0] if instance_xforms else None))

    parts: list[ExtractedMesh] = []
    for i, (start, end, xform) in enumerate(ranges):
        mesh = faces_to_mesh(display_lists, start, end, name, looks)
        if not mesh or mesh.indices.size < 3:
            continue
        if xform and not _is_identity(xform):
            mesh = transform_mesh(mesh, xform)
        if mesh.colors is None:
            look = looks[i] if i < len(looks) else (looks[0] if looks else None)
            if look:
                mesh = paint_mesh(mesh, look.color)
                mesh.metalness = look.metalness
                mesh.roughness = look.roughness
                if look.name:
                    mesh.name = look.name
        parts.append(mesh)
    return merge_meshes(parts)


def extract_best_tessellation(streams: dict[str, bytes], name: str) -> ExtractedMesh | None:
    appearances: list[SwAppearance] = []
    for key, blob in streams.items():
        if len(blob) < 40 or len(blob) > 12_000_000:
            continue
        if "displaylist" in key.lower():
            continue
        appearances.extend(parse_visual_properties(blob))
        if len(blob) <= 4_000_000:
            appearances.extend(parse_face_appearances(blob))
    instances, part_bounds = assembly_from_streams(streams)
    instance_xforms = [i.transform for i in instances]

    preferred: list[bytes] = []
    rest: list[bytes] = []
    for key, blob in streams.items():
        if len(blob) < 200:
            continue
        low = key.lower()
        if any(tok in low for tok in ("previewpng", "preview", ".png", "custom.xml", "app.xml", "core.xml", "[content_types]", "_rels")):
            if "displaylist" not in low:
                continue
        if "displaylist" in low:
            preferred.append(blob)
        else:
            rest.append(blob)
    rest.sort(key=len, reverse=True)

    collected: list[ExtractedMesh] = []
    for buf in preferred:
        mesh = extract_display_tessellation(buf, name, appearances, instance_xforms, instances, part_bounds)
        if mesh and mesh.indices.size >= 3:
            collected.append(mesh)
    if collected:
        return merge_meshes(collected)
    best: ExtractedMesh | None = None
    for buf in rest:
        mesh = extract_display_tessellation(buf, name, appearances, instance_xforms, instances, part_bounds)
        if not mesh or mesh.indices.size < 3:
            continue
        if best is None or mesh.indices.size > best.indices.size:
            best = mesh
    return best
