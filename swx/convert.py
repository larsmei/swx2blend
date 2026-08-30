"""Convert a SolidWorks file into a triangle mesh (no bpy required)."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import auto_scale_to_meters, bounding_box, compute_normals, paint_mesh, weld_mesh
from .parser import parse_solidworks_file
from .tessellation import extract_best_tessellation

MAX_TRIANGLES = 4_000_000


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


def convert_solidworks(data: bytes, name: str = "SolidWorks", weld: bool = True) -> ConvertResult:
    notes: list[str] = []
    streams = parse_solidworks_file(data)
    if not streams:
        raise ValueError("Die SOLIDWORKS-Datei konnte nicht geöffnet werden (unbekanntes Containerformat).")
    mesh = extract_best_tessellation(streams, name)
    if mesh is None or mesh.indices.size < 3:
        raise ValueError(
            "In der Datei steckt keine auslesbare Anzeige-Tessellation. "
            "Speichern Sie das Teil in SOLIDWORKS zusätzlich als STEP."
        )
    notes.append(f"Anzeige-Netz direkt aus der SOLIDWORKS-Datei ({len(streams)} Streams).")
    notes.append("Das ist das Viewport-Netz (Triangle-Strips je Fläche), keine exakte B-Rep-Neuvermaschung.")
    if mesh.colors is None and mesh.color:
        mesh = paint_mesh(mesh, mesh.color)
    if weld:
        mesh = weld_mesh(mesh)
    src_tris = mesh.indices.size // 3
    src_verts = mesh.positions.size // 3
    if src_tris == 0:
        raise ValueError("Die Tessellation lieferte keine Dreiecke.")
    units, _scale = auto_scale_to_meters(mesh.positions)
    if src_tris > MAX_TRIANGLES:
        mesh.indices = mesh.indices[: MAX_TRIANGLES * 3]
        notes.append("Hart auf 4 Mio. Dreiecke gekappt.")
    normals = mesh.normals if mesh.normals is not None and mesh.normals.size == mesh.positions.size else compute_normals(
        mesh.positions, mesh.indices
    )
    mn, mx = bounding_box(mesh.positions)
    n_colors = 0
    if mesh.colors is not None and mesh.colors.size >= 3:
        keys = set()
        for i in range(0, mesh.colors.size, 3):
            keys.add(
                ((round(float(mesh.colors[i]) * 24) & 31) << 10)
                | ((round(float(mesh.colors[i + 1]) * 24) & 31) << 5)
                | (round(float(mesh.colors[i + 2]) * 24) & 31)
            )
            if len(keys) >= 64:
                break
        n_colors = len(keys)
        if n_colors >= 2:
            notes.append(f"{n_colors} Farben aus dem CAD-Modell auf das Netz übertragen.")
    metal = mesh.metalness if mesh.metalness is not None else (0.08 if n_colors >= 1 else 0.42)
    rough = mesh.roughness if mesh.roughness is not None else (0.48 if n_colors >= 1 else 0.38)
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
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="swx2blend — SolidWorks tessellation dump")
    parser.add_argument("file", help=".sldprt / .sldasm")
    args = parser.parse_args(argv)
    path = Path(args.file)
    result = convert_solidworks(path.read_bytes(), path.stem)
    print(f"{path.name}: {result.output_triangles} tris, {result.output_vertices} verts, units={result.units}")
    for note in result.notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
