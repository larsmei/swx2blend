"""Assembly CompInstance XML + DisplayList instance-path markers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

IDENTITY_4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@dataclass
class Aabb:
    min: tuple[float, float, float]
    max: tuple[float, float, float]


@dataclass
class AssemblyRef:
    name: str
    number: str
    model_ref: str
    transform: list[float]
    hidden: bool


@dataclass
class AssemblyModel:
    id: str
    name: str
    refs: list[AssemblyRef] = field(default_factory=list)
    bbox: Aabb | None = None


@dataclass
class AssemblyInstance:
    path: str
    name: str
    transform: list[float]
    mirrored: bool


@dataclass
class NameMarker:
    off: int
    path: str


def multiply4(a: Sequence[float], b: Sequence[float]) -> list[float]:
    r = [0.0] * 16
    for c in range(4):
        for row in range(4):
            r[c * 4 + row] = (
                a[row] * b[c * 4]
                + a[4 + row] * b[c * 4 + 1]
                + a[8 + row] * b[c * 4 + 2]
                + a[12 + row] * b[c * 4 + 3]
            )
    return r


def det3of4(m: Sequence[float]) -> float:
    return (
        m[0] * (m[5] * m[10] - m[6] * m[9])
        - m[4] * (m[1] * m[10] - m[2] * m[9])
        + m[8] * (m[1] * m[6] - m[2] * m[5])
    )


def invert_rigid(m: Sequence[float]) -> list[float]:
    t0, t1, t2 = m[12], m[13], m[14]
    return [
        m[0],
        m[4],
        m[8],
        0.0,
        m[1],
        m[5],
        m[9],
        0.0,
        m[2],
        m[6],
        m[10],
        0.0,
        -(m[0] * t0 + m[1] * t1 + m[2] * t2),
        -(m[4] * t0 + m[5] * t1 + m[6] * t2),
        -(m[8] * t0 + m[9] * t1 + m[10] * t2),
        1.0,
    ]


def parse_sw_transform(text: str | None) -> list[float]:
    if not text:
        return IDENTITY_4[:]
    nums = [float(tok) for tok in re.split(r"[\s,;]+", text.strip()) if tok]
    nums = [n for n in nums if n == n]  # finite
    if len(nums) == 16:
        return nums
    if len(nums) == 12:
        return [
            nums[0],
            nums[4],
            nums[8],
            0.0,
            nums[1],
            nums[5],
            nums[9],
            0.0,
            nums[2],
            nums[6],
            nums[10],
            0.0,
            nums[3],
            nums[7],
            nums[11],
            1.0,
        ]
    return IDENTITY_4[:]


def _parse_bbox_attr(text: str | None) -> Aabb | None:
    if not text:
        return None
    n = [float(tok) for tok in re.split(r"[\s,;]+", text.strip()) if tok]
    n = [v for v in n if v == v]
    if len(n) < 6:
        return None
    return Aabb(min=(n[0], n[1], n[2]), max=(n[3], n[4], n[5]))


_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:]*)="([^"]*)"')


def _attrs(tag: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(tag)}


def _parse_refs(inner: str) -> list[AssemblyRef]:
    refs: list[AssemblyRef] = []
    for m in re.finditer(r"<swReference\b([^>]*?)/?>", inner):
        a = _attrs(m.group(1))
        refs.append(
            AssemblyRef(
                name=a.get("swName") or "",
                number=a.get("swReferenceNumber") or "1",
                model_ref=a.get("swModelRef") or "",
                transform=parse_sw_transform(a.get("swTransform")),
                hidden=a.get("swHidden") == "YES" or a.get("swSuppressed") == "YES",
            )
        )
    return refs


def parse_comp_instance_tree(xml: str) -> list[AssemblyModel]:
    models: list[AssemblyModel] = []
    for m in re.finditer(r"<swModel\b([^>]*?)(/>|>([\s\S]*?)</swModel>)", xml):
        a = _attrs(m.group(1))
        models.append(
            AssemblyModel(
                id=a.get("id") or "",
                name=a.get("swName") or "",
                refs=_parse_refs(m.group(3) or "") if m.group(3) else [],
                bbox=_parse_bbox_attr(a.get("swBoundingBox")),
            )
        )
    return models


def part_bounds_from_models(models: list[AssemblyModel]) -> dict[str, Aabb]:
    out: dict[str, Aabb] = {}
    for model in models:
        if model.bbox and not model.refs and model.name:
            out[model.name] = model.bbox
    return out


def _find_root(models: list[AssemblyModel]) -> AssemblyModel | None:
    if not models:
        return None
    referenced: set[str] = set()
    for m in models:
        for r in m.refs:
            if r.model_ref:
                referenced.add(r.model_ref)
    roots = [m for m in models if m.refs and m.id not in referenced]
    pool = roots or [m for m in models if m.refs]
    if not pool:
        return None
    return sorted(pool, key=lambda m: len(m.refs), reverse=True)[0]


def flatten_assembly_instances(models: list[AssemblyModel]) -> list[AssemblyInstance]:
    by_id = {m.id: m for m in models}
    root = _find_root(models)
    if not root:
        return []
    leaves: list[AssemblyInstance] = []

    def walk(model: AssemblyModel, parent: list[float], parent_path: str, depth: int) -> None:
        if depth > 24:
            return
        for ref in model.refs:
            if ref.hidden:
                continue
            world = multiply4(parent, ref.transform)
            inst = f"{ref.name}-{ref.number}@{model.name}"
            path = f"{parent_path}/{inst}" if parent_path else inst
            child = by_id.get(ref.model_ref)
            if child and child.refs:
                walk(child, world, path, depth + 1)
            else:
                leaves.append(AssemblyInstance(path=path, name=ref.name, transform=world, mirrored=det3of4(world) < 0))

    walk(root, IDENTITY_4[:], "", 0)
    return leaves


def assembly_from_streams(streams: dict[str, bytes]) -> tuple[list[AssemblyInstance], dict[str, Aabb]]:
    for key, blob in streams.items():
        if "compinstance" not in key.lower():
            continue
        xml = blob.decode("utf-8", errors="replace")
        if "swTransform=" not in xml and "swtransform=" not in xml.lower():
            continue
        models = parse_comp_instance_tree(xml)
        return flatten_assembly_instances(models), part_bounds_from_models(models)
    return [], {}


def canonicalize_instance_path(path: str) -> str:
    return "/".join(re.sub(r"\^[^-@]+", "", seg) for seg in path.split("/"))


def is_instance_path(path: str) -> bool:
    if "@" not in path or not (5 <= len(path) <= 240):
        return False
    if not path[0].isalnum():
        return False
    if "://" in path or "\\" in path:
        return False
    return bool(re.search(r"-\d+@", canonicalize_instance_path(path)))


def _read_utf16_run(data: bytes, off: int, max_chars: int = 240) -> str:
    out: list[str] = []
    for i in range(max_chars):
        p = off + i * 2
        if p + 1 >= len(data):
            break
        c = data[p] | (data[p + 1] << 8)
        if c < 32 or c > 126:
            break
        out.append(chr(c))
    return "".join(out)


def find_instance_name_markers(data: bytes) -> list[NameMarker]:
    out: list[NameMarker] = []
    seen: set[int] = set()
    i = 0
    last = len(data) - 1
    while i < last:
        at = data.find(0x40, i)
        if at < 0 or at >= last:
            break
        if data[at + 1] != 0:
            i = at + 1
            continue
        start = at
        while start >= 2:
            c = data[start - 2] | (data[start - 1] << 8)
            if c < 32 or c > 126:
                break
            start -= 2
        if start not in seen:
            path = _read_utf16_run(data, start)
            if is_instance_path(path):
                seen.add(start)
                out.append(NameMarker(off=start, path=path))
        i = at + 2
    out.sort(key=lambda m: m.off)
    return out


def part_name_from_instance_path(path: str) -> str:
    last = canonicalize_instance_path(path).split("/")[-1] or path
    return re.sub(r"-\d+@.*$", "", last) or last


def match_instance(path: str, instances: list[AssemblyInstance]) -> AssemblyInstance | None:
    canon = canonicalize_instance_path(path)
    for inst in instances:
        if inst.path == path or inst.path == canon:
            return inst
    last = canon.split("/")[-1]
    by_last = [
        inst
        for inst in instances
        if inst.path.split("/")[-1] == last or inst.path == last or inst.path.endswith("/" + last)
    ]
    if len(by_last) == 1:
        return by_last[0]
    if len(by_last) > 1:
        slash = canon.rfind("/")
        parent = canon[:slash] if slash >= 0 else ""
        if parent:
            same = [
                inst
                for inst in by_last
                if inst.path.startswith(parent + "/") or canonicalize_instance_path(inst.path).startswith(parent + "/")
            ]
        else:
            same = [inst for inst in by_last if "/" not in inst.path]
        if len(same) == 1:
            return same[0]
    return None
