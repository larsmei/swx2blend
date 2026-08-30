"""Mesh helpers: normals, transforms, weld, body split, merge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

DEFAULT_CAD_COLOR = (0.718, 0.745, 0.784)


@dataclass
class ExtractedMesh:
    positions: np.ndarray
    normals: np.ndarray | None
    indices: np.ndarray
    name: str
    color: tuple[float, float, float] | None = None
    colors: np.ndarray | None = None
    metalness: float | None = None
    roughness: float | None = None
    uvs: np.ndarray | None = None
    tex_index: np.ndarray | None = None
    textures: list | None = None
    materials: list | None = None


def compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    pos = positions.reshape(-1, 3)
    nrm = normals.reshape(-1, 3)
    idx = indices.reshape(-1, 3)
    a = pos[idx[:, 0]]
    b = pos[idx[:, 1]]
    c = pos[idx[:, 2]]
    cross = np.cross(b - a, c - a)
    np.add.at(nrm, idx[:, 0], cross)
    np.add.at(nrm, idx[:, 1], cross)
    np.add.at(nrm, idx[:, 2], cross)
    lengths = np.linalg.norm(nrm, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    nrm /= lengths
    return normals


def bounding_box(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = positions.reshape(-1, 3)
    if p.size == 0:
        inf = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        return inf, -inf
    return p.min(axis=0), p.max(axis=0)


def scale_positions(positions: np.ndarray, s: float) -> None:
    if s != 1.0:
        positions *= s


def auto_scale_to_meters(positions: np.ndarray) -> tuple[str, float]:
    mn, mx = bounding_box(positions)
    diag = float(np.linalg.norm(mx - mn))
    if not np.isfinite(diag) or diag == 0:
        return "m", 1.0
    if diag > 40:
        scale_positions(positions, 0.001)
        return "mm→m", 0.001
    if diag < 1e-4:
        scale_positions(positions, 1000)
        return "km→m", 1000
    return "m", 1.0


def paint_mesh(mesh: ExtractedMesh, color: tuple[float, float, float]) -> ExtractedMesh:
    n = mesh.positions.size // 3
    colors = np.empty(n * 3, dtype=np.float32)
    colors[0::3] = color[0]
    colors[1::3] = color[1]
    colors[2::3] = color[2]
    return replace(mesh, color=color, colors=colors)


def is_identity4(m: Sequence[float], eps: float = 1e-10) -> bool:
    if len(m) < 12:
        return False
    return (
        abs(m[0] - 1) <= eps
        and abs(m[5] - 1) <= eps
        and abs(m[10] - 1) <= eps
        and abs(m[1]) <= eps
        and abs(m[2]) <= eps
        and abs(m[4]) <= eps
        and abs(m[6]) <= eps
        and abs(m[8]) <= eps
        and abs(m[9]) <= eps
        and abs(m[12] if len(m) > 12 else 0) <= eps
        and abs(m[13] if len(m) > 13 else 0) <= eps
        and abs(m[14] if len(m) > 14 else 0) <= eps
    )


def _det3(m: Sequence[float]) -> float:
    return (
        m[0] * (m[5] * m[10] - m[6] * m[9])
        - m[4] * (m[1] * m[10] - m[2] * m[9])
        + m[8] * (m[1] * m[6] - m[2] * m[5])
    )


def transform_mesh(mesh: ExtractedMesh, m: Sequence[float]) -> ExtractedMesh:
    if len(m) < 16 or is_identity4(m):
        return mesh
    pos = mesh.positions.reshape(-1, 3)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    out = np.empty_like(mesh.positions)
    out[0::3] = m[0] * x + m[4] * y + m[8] * z + m[12]
    out[1::3] = m[1] * x + m[5] * y + m[9] * z + m[13]
    out[2::3] = m[2] * x + m[6] * y + m[10] * z + m[14]
    normals = mesh.normals
    if normals is not None and normals.size == out.size:
        n = normals.reshape(-1, 3)
        nx = m[0] * n[:, 0] + m[4] * n[:, 1] + m[8] * n[:, 2]
        ny = m[1] * n[:, 0] + m[5] * n[:, 1] + m[9] * n[:, 2]
        nz = m[2] * n[:, 0] + m[6] * n[:, 1] + m[10] * n[:, 2]
        nrm = np.empty_like(normals)
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        length = np.maximum(length, 1e-12)
        nrm[0::3] = nx / length
        nrm[1::3] = ny / length
        nrm[2::3] = nz / length
        normals = nrm
    indices = mesh.indices
    if _det3(m) < 0 and indices.size >= 3:
        flipped = indices.copy()
        flipped[1::3] = indices[2::3]
        flipped[2::3] = indices[1::3]
        indices = flipped
    return replace(mesh, positions=out, normals=normals, indices=indices)


def stamp_mesh(mesh: ExtractedMesh, transforms: Sequence[Sequence[float]]) -> ExtractedMesh:
    if not transforms:
        return mesh
    if len(transforms) == 1:
        return transform_mesh(mesh, transforms[0])
    n_v = mesh.positions.size // 3
    n_i = mesh.indices.size
    k = len(transforms)
    if n_v < 1 or n_i < 3:
        return mesh
    src_p = mesh.positions
    src_n = mesh.normals if mesh.normals is not None and mesh.normals.size == src_p.size else None
    src_c = mesh.colors if mesh.colors is not None and mesh.colors.size == src_p.size else None
    src_uv = mesh.uvs if mesh.uvs is not None and mesh.uvs.size == n_v * 2 else None
    src_tex = mesh.tex_index if mesh.tex_index is not None and mesh.tex_index.size == n_v else None
    src_i = mesh.indices
    positions = np.empty(n_v * 3 * k, dtype=np.float32)
    normals = np.empty(n_v * 3 * k, dtype=np.float32) if src_n is not None else None
    colors = np.empty(n_v * 3 * k, dtype=np.float32) if src_c is not None else None
    uvs = np.empty(n_v * 2 * k, dtype=np.float32) if src_uv is not None else None
    tex_index = np.empty(n_v * k, dtype=np.uint16) if src_tex is not None else None
    indices = np.empty(n_i * k, dtype=np.uint32)
    for t, m in enumerate(transforms):
        vo = t * n_v
        po = vo * 3
        if len(m) >= 16 and not is_identity4(m):
            x = src_p[0::3]
            y = src_p[1::3]
            z = src_p[2::3]
            positions[po : po + n_v * 3 : 3] = m[0] * x + m[4] * y + m[8] * z + m[12]
            positions[po + 1 : po + n_v * 3 : 3] = m[1] * x + m[5] * y + m[9] * z + m[13]
            positions[po + 2 : po + n_v * 3 : 3] = m[2] * x + m[6] * y + m[10] * z + m[14]
            if normals is not None and src_n is not None:
                nx = m[0] * src_n[0::3] + m[4] * src_n[1::3] + m[8] * src_n[2::3]
                ny = m[1] * src_n[0::3] + m[5] * src_n[1::3] + m[9] * src_n[2::3]
                nz = m[2] * src_n[0::3] + m[6] * src_n[1::3] + m[10] * src_n[2::3]
                length = np.sqrt(nx * nx + ny * ny + nz * nz)
                length = np.maximum(length, 1e-12)
                normals[po : po + n_v * 3 : 3] = nx / length
                normals[po + 1 : po + n_v * 3 : 3] = ny / length
                normals[po + 2 : po + n_v * 3 : 3] = nz / length
        else:
            positions[po : po + n_v * 3] = src_p
            if normals is not None and src_n is not None:
                normals[po : po + n_v * 3] = src_n
        if colors is not None and src_c is not None:
            colors[po : po + n_v * 3] = src_c
        if uvs is not None and src_uv is not None:
            uvs[vo * 2 : (vo + n_v) * 2] = src_uv
        if tex_index is not None and src_tex is not None:
            tex_index[vo : vo + n_v] = src_tex
        io = t * n_i
        det = 1.0 if len(m) < 16 else _det3(m)
        if det < 0:
            indices[io : io + n_i : 3] = src_i[0::3] + vo
            indices[io + 1 : io + n_i : 3] = src_i[2::3] + vo
            indices[io + 2 : io + n_i : 3] = src_i[1::3] + vo
        elif vo == 0:
            indices[io : io + n_i] = src_i
        else:
            indices[io : io + n_i] = src_i + vo
    return replace(
        mesh,
        positions=positions,
        normals=normals,
        indices=indices,
        colors=colors,
        uvs=uvs,
        tex_index=tex_index,
    )


def merge_meshes(meshes: Sequence[ExtractedMesh]) -> ExtractedMesh | None:
    usable = [m for m in meshes if m.indices.size >= 3 and m.positions.size >= 9]
    if not usable:
        return None
    if len(usable) == 1:
        only = usable[0]
        if only.colors is None and only.color:
            return paint_mesh(only, only.color)
        return only
    v_count = sum(m.positions.size // 3 for m in usable)
    i_count = sum(m.indices.size for m in usable)
    has_n = any(m.normals is not None and m.normals.size == m.positions.size for m in usable)
    has_c = any(m.colors is not None or m.color for m in usable)
    has_uv = any(m.uvs is not None and m.uvs.size == (m.positions.size // 3) * 2 for m in usable)
    has_tex = any(m.tex_index is not None and m.tex_index.size == m.positions.size // 3 for m in usable)
    positions = np.empty(v_count * 3, dtype=np.float32)
    normals = np.zeros(v_count * 3, dtype=np.float32) if has_n else None
    colors = np.empty(v_count * 3, dtype=np.float32) if has_c else None
    uvs = np.zeros(v_count * 2, dtype=np.float32) if has_uv else None
    tex_index = np.full(v_count, 0xFFFF, dtype=np.uint16) if has_tex else None
    indices = np.empty(i_count, dtype=np.uint32)
    vo = 0
    io = 0
    for m in usable:
        n = m.positions.size // 3
        positions[vo * 3 : (vo + n) * 3] = m.positions
        if normals is not None and m.normals is not None and m.normals.size == m.positions.size:
            normals[vo * 3 : (vo + n) * 3] = m.normals
        if colors is not None:
            if m.colors is not None and m.colors.size == m.positions.size:
                colors[vo * 3 : (vo + n) * 3] = m.colors
            else:
                c = m.color or DEFAULT_CAD_COLOR
                colors[vo * 3 : (vo + n) * 3 : 3] = c[0]
                colors[vo * 3 + 1 : (vo + n) * 3 : 3] = c[1]
                colors[vo * 3 + 2 : (vo + n) * 3 : 3] = c[2]
        if uvs is not None and m.uvs is not None and m.uvs.size == n * 2:
            uvs[vo * 2 : (vo + n) * 2] = m.uvs
        if tex_index is not None:
            if m.tex_index is not None and m.tex_index.size == n:
                tex_index[vo : vo + n] = m.tex_index
            else:
                tex_index[vo : vo + n] = 0xFFFF
        indices[io : io + m.indices.size] = m.indices + vo
        io += m.indices.size
        vo += n
    metal = next((m.metalness for m in usable if m.metalness is not None), None)
    rough = next((m.roughness for m in usable if m.roughness is not None), None)
    textures = next((m.textures for m in usable if m.textures), None)
    materials = next((m.materials for m in usable if m.materials), None)
    return ExtractedMesh(
        positions=positions,
        normals=normals,
        indices=indices[:io],
        name=usable[0].name,
        color=usable[0].color,
        colors=colors,
        metalness=metal,
        roughness=rough,
        uvs=uvs,
        tex_index=tex_index,
        textures=textures,
        materials=materials,
    )


def weld_mesh(mesh: ExtractedMesh, tolerance: float | None = None) -> ExtractedMesh:
    mn, mx = bounding_box(mesh.positions)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    eps = tolerance if tolerance is not None else diag * 1e-6
    inv = 1.0 / max(eps, 1e-12)
    keep_color = mesh.colors is not None and mesh.colors.size == mesh.positions.size
    keep_uv = mesh.uvs is not None and mesh.uvs.size == (mesh.positions.size // 3) * 2
    keep_tex = mesh.tex_index is not None and mesh.tex_index.size == mesh.positions.size // 3
    table: dict[tuple[int, ...], int] = {}
    pos: list[float] = []
    nrm: list[float] = []
    col: list[float] = []
    uv: list[float] = []
    tex: list[int] = []
    remap = np.empty(mesh.positions.size // 3, dtype=np.uint32)
    n = remap.size
    for i in range(n):
        x = float(mesh.positions[i * 3])
        y = float(mesh.positions[i * 3 + 1])
        z = float(mesh.positions[i * 3 + 2])
        if keep_color and mesh.colors is not None:
            rgb = (float(mesh.colors[i * 3]), float(mesh.colors[i * 3 + 1]), float(mesh.colors[i * 3 + 2]))
        else:
            rgb = mesh.color or DEFAULT_CAD_COLOR
        tex_id = int(mesh.tex_index[i]) if keep_tex and mesh.tex_index is not None else 0
        if keep_color:
            key = (
                round(x * inv),
                round(y * inv),
                round(z * inv),
                round(rgb[0] * 32),
                round(rgb[1] * 32),
                round(rgb[2] * 32),
                tex_id,
            )
        else:
            key = (round(x * inv), round(y * inv), round(z * inv), tex_id)
        ident = table.get(key)
        if ident is None:
            ident = len(pos) // 3
            table[key] = ident
            pos.extend((x, y, z))
            if mesh.normals is not None:
                nrm.extend(
                    (
                        float(mesh.normals[i * 3]),
                        float(mesh.normals[i * 3 + 1]),
                        float(mesh.normals[i * 3 + 2]),
                    )
                )
            if keep_color:
                col.extend(rgb)
            if keep_uv and mesh.uvs is not None:
                uv.extend((float(mesh.uvs[i * 2]), float(mesh.uvs[i * 2 + 1])))
            if keep_tex and mesh.tex_index is not None:
                tex.append(int(mesh.tex_index[i]))
        remap[i] = ident
    out_i: list[int] = []
    for i in range(0, mesh.indices.size, 3):
        a = int(remap[int(mesh.indices[i])])
        b = int(remap[int(mesh.indices[i + 1])])
        c = int(remap[int(mesh.indices[i + 2])])
        if a == b or b == c or a == c:
            continue
        out_i.extend((a, b, c))
    return ExtractedMesh(
        positions=np.asarray(pos, dtype=np.float32),
        normals=np.asarray(nrm, dtype=np.float32) if mesh.normals is not None else None,
        indices=np.asarray(out_i, dtype=np.uint32),
        name=mesh.name,
        color=mesh.color,
        colors=np.asarray(col, dtype=np.float32) if keep_color else mesh.colors,
        metalness=mesh.metalness,
        roughness=mesh.roughness,
        uvs=np.asarray(uv, dtype=np.float32) if keep_uv else mesh.uvs,
        tex_index=np.asarray(tex, dtype=np.uint16) if keep_tex else mesh.tex_index,
        textures=mesh.textures,
        materials=mesh.materials,
    )


def split_connected_bodies(mesh: ExtractedMesh) -> list[ExtractedMesh]:
    if mesh.indices.size < 3:
        return []
    v_n = mesh.positions.size // 3
    if v_n < 3:
        return []
    mn, mx = bounding_box(mesh.positions)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    inv = 1.0 / max(diag * 1e-6, 1e-12)
    parent = np.arange(v_n, dtype=np.int32)

    def find(a: int) -> int:
        x = a
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def unite(a: int, b: int) -> None:
        a = find(a)
        b = find(b)
        if a != b:
            parent[b] = a

    spatial: dict[tuple[int, int, int], int] = {}
    for i in range(v_n):
        key = (
            round(float(mesh.positions[i * 3]) * inv),
            round(float(mesh.positions[i * 3 + 1]) * inv),
            round(float(mesh.positions[i * 3 + 2]) * inv),
        )
        prev = spatial.get(key)
        if prev is not None:
            unite(prev, i)
        else:
            spatial[key] = i
    for i in range(0, mesh.indices.size, 3):
        unite(int(mesh.indices[i]), int(mesh.indices[i + 1]))
        unite(int(mesh.indices[i + 1]), int(mesh.indices[i + 2]))
    groups: dict[int, list[int]] = {}
    for i in range(0, mesh.indices.size, 3):
        r = find(int(mesh.indices[i]))
        groups.setdefault(r, []).extend(
            (int(mesh.indices[i]), int(mesh.indices[i + 1]), int(mesh.indices[i + 2]))
        )
    has_n = mesh.normals is not None and mesh.normals.size == mesh.positions.size
    has_c = mesh.colors is not None and mesh.colors.size == mesh.positions.size
    has_uv = mesh.uvs is not None and mesh.uvs.size == (mesh.positions.size // 3) * 2
    has_tex = mesh.tex_index is not None and mesh.tex_index.size == mesh.positions.size // 3
    bodies: list[ExtractedMesh] = []
    for idx in groups.values():
        used: dict[int, int] = {}
        pos: list[float] = []
        nrm: list[float] = []
        col: list[float] = []
        uv: list[float] = []
        tex: list[int] = []
        remap: list[int] = []
        for old in idx:
            ident = used.get(old)
            if ident is None:
                ident = len(pos) // 3
                used[old] = ident
                pos.extend(
                    (
                        float(mesh.positions[old * 3]),
                        float(mesh.positions[old * 3 + 1]),
                        float(mesh.positions[old * 3 + 2]),
                    )
                )
                if has_n and mesh.normals is not None:
                    nrm.extend(
                        (
                            float(mesh.normals[old * 3]),
                            float(mesh.normals[old * 3 + 1]),
                            float(mesh.normals[old * 3 + 2]),
                        )
                    )
                if has_c and mesh.colors is not None:
                    col.extend(
                        (
                            float(mesh.colors[old * 3]),
                            float(mesh.colors[old * 3 + 1]),
                            float(mesh.colors[old * 3 + 2]),
                        )
                    )
                if has_uv and mesh.uvs is not None:
                    uv.extend((float(mesh.uvs[old * 2]), float(mesh.uvs[old * 2 + 1])))
                if has_tex and mesh.tex_index is not None:
                    tex.append(int(mesh.tex_index[old]))
            remap.append(ident)
        if len(remap) < 3:
            continue
        bodies.append(
            ExtractedMesh(
                positions=np.asarray(pos, dtype=np.float32),
                normals=np.asarray(nrm, dtype=np.float32) if has_n else None,
                indices=np.asarray(remap, dtype=np.uint32),
                name=mesh.name,
                color=mesh.color,
                colors=np.asarray(col, dtype=np.float32) if has_c else None,
                metalness=mesh.metalness,
                roughness=mesh.roughness,
                uvs=np.asarray(uv, dtype=np.float32) if has_uv else None,
                tex_index=np.asarray(tex, dtype=np.uint16) if has_tex else None,
            )
        )
    bodies.sort(key=lambda b: b.indices.size, reverse=True)
    return bodies
