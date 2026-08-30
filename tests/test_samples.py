"""Parse the shipped SolidWorks samples (run from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swx.convert import convert_solidworks  # noqa: E402


def _sample(name: str) -> Path | None:
    for path in (
        Path("/workspace/public/samples") / name,
        ROOT.parents[1] / "public" / "samples" / name,
        ROOT / "tests" / name,
    ):
        if path.exists():
            return path
    return None


def test_teil() -> None:
    path = _sample("teil.sldprt")
    if not path:
        print("skip teil.sldprt")
        return
    result = convert_solidworks(path.read_bytes(), path.stem)
    assert result.output_triangles >= 100, result.output_triangles
    assert result.part_count >= 1
    assert result.materials, "part should carry at least one appearance"
    assert result.textures, "brushed steel should emit a procedural map"
    part = next(iter(result.parts.values()))
    assert part.uvs is not None and part.uvs.size >= 6
    print(
        f"teil: {result.output_triangles} tris, {result.part_count} parts, "
        f"{len(result.materials)} mats, {len(result.textures)} tex"
    )


def test_baugruppe() -> None:
    path = _sample("baugruppe.sldasm")
    if not path:
        print("skip baugruppe.sldasm")
        return
    result = convert_solidworks(path.read_bytes(), path.stem)
    assert result.output_triangles >= 1000, result.output_triangles
    assert result.tree is not None, "assembly should keep CompInstance tree"
    assert result.part_count >= 2, result.parts.keys()
    assert result.instance_count >= 2, result.instance_count
    assert result.tree.children, "root should have children"
    assert any(child.children for child in result.tree.children), "nested subassemblies"
    assert result.materials, "each part should have a principled look"
    nested = sum(1 for child in result.tree.children if child.children)
    print(
        f"baugruppe: {result.output_triangles} tris, {result.part_count} parts, "
        f"{result.instance_count} instances, {len(result.materials)} mats, "
        f"tree={result.tree.name} children={len(result.tree.children)} nested={nested}"
    )
    for child in result.tree.children[:8]:
        print(f"  {child.name} part={child.part} kids={len(child.children)}")


if __name__ == "__main__":
    test_teil()
    test_baugruppe()
    print("ok")
