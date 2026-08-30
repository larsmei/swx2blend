"""Convert a SolidWorks file into a triangle mesh (no bpy required)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assembly import SceneNode, scale_tree_translations
from .geometry import bounding_box, compute_normals, paint_mesh, scale_positions, weld_mesh
from .parser import parse_solidworks_file
from .tessellation import extract_scene
from .textures import NONE, CadPbr, CadTexture, fill_tex_index, intern_pbr

MAX_TRIANGLES = 4_000_000


@dataclass
class ConvertPart:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None
    uvs: np.ndarray | None
    tex_index: np.ndarray | None
    metalness: float
    roughness: float


@dataclass
class ConvertResult:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    colors: np.ndarray | None
    metalness: float
    roughness: float
    source_triangles: int
    output_triangles: int
    source_vertices: int
    output_vertices: int
    units: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    notes: list[str]
    name: str
    uvs: np.ndarray | None = None
    tex_index: np.ndarray | None = None
    textures: list[CadTexture] = field(default_factory=list)
    materials: list[CadPbr] = field(default_factory=list)
    parts: dict[str, ConvertPart] = field(default_factory=dict)
    tree: SceneNode | None = None
    instance_count: int = 0
    part_count: int = 0


def _prep_mesh(mesh, weld: bool, scale: float):
    if mesh.colors is None and mesh.color:
        mesh = paint_mesh(mesh, mesh.color)
    if weld:
        mesh = weld_mesh(mesh)
    if scale != 1.0:
        scale_positions(mesh.positions, scale)
    normals = mesh.normals if mesh.normals is not None and mesh.normals.size == mesh.positions.size else compute_normals(
        mesh.positions, mesh.indices
    )
    return mesh, normals


def _n_colors(colors: np.ndarray | None) -> int:
    if colors is None or colors.size < 3:
        return 0
    keys = set()
    for i in range(0, colors.size, 3):
        keys.add(
            ((round(float(colors[i]) * 24) & 31) << 10)
            | ((round(float(colors[i + 1]) * 24) & 31) << 5)
            | (round(float(colors[i + 2]) * 24) & 31)
        )
        if len(keys) >= 64:
            break
    return len(keys)


def _tint_of(part: ConvertPart) -> tuple[float, float, float]:
    if part.colors is not None and part.colors.size >= 3:
        return (float(part.colors[0]), float(part.colors[1]), float(part.colors[2]))
    return (0.718, 0.745, 0.784)


def _ensure_part_materials(parts: dict[str, ConvertPart], materials: list[CadPbr]) -> None:
    """Every part gets a named Principled look so Blender can keep the hierarchy."""
    n_mat = len(materials)
    for part in parts.values():
        n = part.positions.size // 3
        tex = part.tex_index
        valid = False
        if tex is not None and n_mat > 0 and tex.size == n:
            valid = int(np.count_nonzero(tex < n_mat)) > n * 0.1
        if valid:
            continue
        pbr = CadPbr(
            id=f"part:{part.name}",
            name=part.name,
            metalness=part.metalness,
            roughness=part.roughness,
        )
        idx = intern_pbr(materials, pbr)
        if tex is None or tex.size != n:
            part.tex_index = fill_tex_index(n, idx)
        else:
            next_idx = np.array(tex, copy=True)
            next_idx[(next_idx >= n_mat) | (next_idx == NONE)] = idx
            part.tex_index = next_idx


def convert_solidworks(data: bytes, name: str = "SolidWorks", weld: bool = True) -> ConvertResult:
    notes: list[str] = []
    streams = parse_solidworks_file(data)
    if not streams:
        raise ValueError("Die SOLIDWORKS-Datei konnte nicht geöffnet werden (unbekanntes Containerformat).")
    scene = extract_scene(streams, name)
    mesh = scene.merged
    if mesh is None or mesh.indices.size < 3:
        raise ValueError(
            "In der Datei steckt keine auslesbare Anzeige-Tessellation. "
            "Speichern Sie das Teil in SOLIDWORKS zusätzlich als STEP."
        )
    notes.append(f"Anzeige-Netz direkt aus der SOLIDWORKS-Datei ({len(streams)} Streams).")
    notes.append("Das ist das Viewport-Netz (Triangle-Strips je Fläche), keine exakte B-Rep-Neuvermaschung.")
    if scene.instances:
        unique = len({i.name for i in scene.instances})
        mirrored = sum(1 for i in scene.instances if i.mirrored)
        extra = []
        patterned = len(scene.instances) - unique
        if patterned > 0:
            extra.append(f"{patterned} Musterkopien")
        if mirrored:
            extra.append(f"{mirrored} Spiegelungen")
        suffix = f", {', '.join(extra)}" if extra else ""
        notes.append(f"Baugruppe: {len(scene.instances)} Platzierungen ({unique} Teile{suffix}).")
        if scene.tree:
            notes.append("Baugruppenbaum aus CompInstance-XML — Prototypen bleiben im Teile-Raum.")

    mn0, mx0 = bounding_box(mesh.positions)
    diag0 = float(np.linalg.norm(mx0 - mn0))
    if not np.isfinite(diag0) or diag0 == 0:
        scale = 1.0
        units = "m"
    elif diag0 > 40:
        scale = 0.001
        units = "mm→m"
    elif diag0 < 1e-4:
        scale = 1000
        units = "km→m"
    else:
        scale = 1.0
        units = "m"

    src_tris = mesh.indices.size // 3
    src_verts = mesh.positions.size // 3
    mesh, normals = _prep_mesh(mesh, weld, scale)
    if src_tris == 0:
        raise ValueError("Die Tessellation lieferte keine Dreiecke.")
    if mesh.indices.size // 3 > MAX_TRIANGLES:
        mesh.indices = mesh.indices[: MAX_TRIANGLES * 3]
        notes.append("Hart auf 4 Mio. Dreiecke gekappt.")

    parts: dict[str, ConvertPart] = {}
    merged_id = id(scene.merged.positions) if scene.merged is not None else None
    for part_name, proto in scene.prototypes.items():
        same = merged_id is not None and id(proto.positions) == merged_id
        if same:
            pmesh, pnrm = mesh, normals
        else:
            pmesh, pnrm = _prep_mesh(proto, weld, scale)
        n_c = _n_colors(pmesh.colors)
        metal = pmesh.metalness if pmesh.metalness is not None else (0.08 if n_c >= 1 else 0.42)
        rough = pmesh.roughness if pmesh.roughness is not None else (0.48 if n_c >= 1 else 0.38)
        parts[part_name] = ConvertPart(
            name=part_name,
            positions=pmesh.positions,
            normals=pnrm,
            indices=pmesh.indices,
            colors=pmesh.colors,
            uvs=pmesh.uvs,
            tex_index=pmesh.tex_index,
            metalness=float(metal),
            roughness=float(rough),
        )
    if not parts:
        parts[mesh.name or name] = ConvertPart(
            name=mesh.name or name,
            positions=mesh.positions,
            normals=normals,
            indices=mesh.indices,
            colors=mesh.colors,
            uvs=mesh.uvs,
            tex_index=mesh.tex_index,
            metalness=float(mesh.metalness if mesh.metalness is not None else 0.42),
            roughness=float(mesh.roughness if mesh.roughness is not None else 0.38),
        )

    tree = scene.tree
    if tree is not None and scale != 1.0:
        scale_tree_translations(tree, scale)

    n_colors = _n_colors(mesh.colors)
    if n_colors >= 2:
        notes.append(f"{n_colors} Farben aus dem CAD-Modell auf das Netz übertragen.")
    textures = scene.textures
    materials = scene.materials
    before_looks = len(materials)
    _ensure_part_materials(parts, materials)
    if materials:
        n_alb = sum(1 for t in textures if t.role == "albedo")
        n_nrm = sum(1 for t in textures if t.role == "normal")
        bits = [f"{len(materials)} PBR-Looks"]
        if n_alb:
            bits.append(f"{n_alb} Diffuse")
        if n_nrm:
            bits.append(f"{n_nrm} Normal")
        if any(t.role == "roughness" for t in textures):
            bits.append("Rauheit")
        notes.append("PBR-Materialien: " + ", ".join(bits) + ".")
        if len(materials) > before_looks:
            notes.append("Fehlende Appearances als Principled-Looks je Teil ergänzt.")
    elif n_colors >= 1:
        notes.append("Appearances ohne zuordenbare Textur — Farbe und Glanz übernommen.")

    metal = mesh.metalness if mesh.metalness is not None else (0.08 if n_colors >= 1 else 0.42)
    rough = mesh.roughness if mesh.roughness is not None else (0.48 if n_colors >= 1 else 0.38)
    mn, mx = bounding_box(mesh.positions)
    return ConvertResult(
        positions=mesh.positions,
        normals=normals,
        indices=mesh.indices,
        colors=mesh.colors,
        metalness=float(metal),
        roughness=float(rough),
        source_triangles=src_tris,
        output_triangles=mesh.indices.size // 3,
        source_vertices=src_verts,
        output_vertices=mesh.positions.size // 3,
        units=units,
        bbox_min=(float(mn[0]), float(mn[1]), float(mn[2])),
        bbox_max=(float(mx[0]), float(mx[1]), float(mx[2])),
        notes=notes,
        name=mesh.name or name,
        uvs=mesh.uvs,
        tex_index=mesh.tex_index,
        textures=textures,
        materials=materials,
        parts=parts,
        tree=tree,
        instance_count=len(scene.instances),
        part_count=len(parts),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="swx2blend — SolidWorks tessellation dump")
    parser.add_argument("file", help=".sldprt / .sldasm")
    args = parser.parse_args(argv)
    path = Path(args.file)
    result = convert_solidworks(path.read_bytes(), path.stem)
    print(
        f"{path.name}: {result.output_triangles} tris, {result.output_vertices} verts, "
        f"{result.part_count} Teile, {result.instance_count} Instanzen, "
        f"{len(result.materials)} Materialien, {len(result.textures)} Texturen, units={result.units}"
    )
    for note in result.notes:
        print(f"  {note}")
    if result.tree:
        print(f"  Baum: {result.tree.name} ({len(result.tree.children)} Kinder)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
