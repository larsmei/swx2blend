"""Parse the shipped SolidWorks samples (run from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swx.convert import convert_solidworks  # noqa: E402


def test_teil() -> None:
    path = Path("/workspace/public/samples/teil.sldprt")
    if not path.exists():
        path = ROOT / "tests" / "teil.sldprt"
    if not path.exists():
        print("skip teil.sldprt")
        return
    result = convert_solidworks(path.read_bytes(), path.stem)
    assert result.output_triangles >= 100, result.output_triangles
    print(f"teil: {result.output_triangles} tris")


def test_baugruppe() -> None:
    path = Path("/workspace/public/samples/baugruppe.sldasm")
    if not path.exists():
        path = ROOT / "tests" / "baugruppe.sldasm"
    if not path.exists():
        print("skip baugruppe.sldasm")
        return
    result = convert_solidworks(path.read_bytes(), path.stem)
    assert result.output_triangles >= 1000, result.output_triangles
    print(f"baugruppe: {result.output_triangles} tris")


if __name__ == "__main__":
    test_teil()
    test_baugruppe()
    print("ok")
