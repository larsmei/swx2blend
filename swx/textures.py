"""SolidWorks appearances, raster extraction, box UVs, procedural PBR maps."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .inflate import index_of_seq, index_of_utf16

Vec3 = tuple[float, float, float]
Rgb = tuple[float, float, float]
TexRole = Literal["albedo", "normal", "roughness", "metalness"]

NONE = 0xFFFF
PROC_SIZE = 256


@dataclass
class CadTexture:
    id: str
    name: str
    width: int
    height: int
    rgba: np.ndarray | None
    role: TexRole = "albedo"
    encoded: bytes | None = None
    mime: str | None = None


@dataclass
class CadPbr:
    id: str
    name: str
    metalness: float
    roughness: float
    albedo: int | None = None
    normal: int | None = None
    roughness_map: int | None = None
    metalness_map: int | None = None
    normal_scale: float | None = None


@dataclass
class CadMapping:
    scale_u: float = 0.1
    scale_v: float = 0.1
    rotation_deg: float = 0.0
    origin: Vec3 | None = None
    axis_u: Vec3 | None = None
    axis_v: Vec3 | None = None
    map_type: int | None = None


@dataclass
class TextureLook:
    path: str
    scale_u: float
    scale_v: float
    rotation_deg: float
    off: int = 0
    bump_amount: float | None = None
    origin: Vec3 | None = None
    axis_u: Vec3 | None = None
    axis_v: Vec3 | None = None
    map_type: int | None = None
    pbr: CadPbr | None = None


@dataclass
class PbrProfile:
    kind: str
    albedo: bool
    bump: bool
    metalness: float
    roughness: float


def _clamp01(n: float) -> float:
    return 0.0 if n < 0 else 1.0 if n > 1 else n


def _clamp_byte(n: float) -> int:
    return max(0, min(255, int(n)))


def sw_bump_amount(value: float | None, fallback: float = 0.3) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        return _clamp01(fallback)
    if 1.5 < value <= 100:
        return _clamp01(value / 100)
    return _clamp01(value)


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("\\", "/"))


def pbr_profile(path_or_name: str) -> PbrProfile | None:
    n = path_or_name.lower().replace("\\", "/")
    if re.search(r"color\.p2m|defaultplastic|clearglass|clear glass|glass/gloss|blacklowgloss|low gloss plastic", n) and not re.search(
        r"frost|texture|brushed|knurl|carbon|bump", n
    ):
        return None
    if re.search(r"brushed|zinc|steel|chrome|aluminium|aluminum|iron|metal/|copper|brass|gold|titanium", n):
        return PbrProfile("brushed", False, True, 0.82, 0.3)
    if re.search(r"carbon|kevlar|weave", n):
        return PbrProfile("carbon", True, True, 0.18, 0.42)
    if re.search(r"wood|oak|pine|walnut|maple|teak", n):
        return PbrProfile("wood", True, True, 0.02, 0.62)
    if re.search(r"checker|chequer|checkered", n):
        return PbrProfile("checker", True, False, 0.04, 0.5)
    if re.search(r"knurl|diamond plate|tread", n):
        return PbrProfile("knurl", False, True, 0.72, 0.4)
    if re.search(r"cast|sand|roughcast", n):
        return PbrProfile("cast", False, True, 0.62, 0.5)
    if re.search(r"rubber|nitrile|elastomer", n):
        return PbrProfile("rubber", False, True, 0.02, 0.78)
    if "leather" in n:
        return PbrProfile("leather", True, True, 0.02, 0.58)
    if re.search(r"fabric|cloth|carpet", n):
        return PbrProfile("fabric", True, True, 0.01, 0.82)
    if re.search(r"concrete|stone|granite|marble", n):
        return PbrProfile("stone", True, True, 0.04, 0.72)
    if re.search(r"plastic/(textured|rough|medium gloss)", n):
        return PbrProfile("speckle", True, True, 0.04, 0.55)
    return None


def classify_raster_role(name: str) -> TexRole:
    n = name.lower().replace("\\", "/")
    if re.search(r"bump|normal|nrm|height|disp|displacement", n):
        return "normal"
    if re.search(r"rough|rgh|gloss", n) and "glass" not in n:
        return "roughness"
    if re.search(r"metal|metallic|mtl", n) and not re.search(r"sheet|part", n):
        return "metalness"
    return "albedo"


def _hash2(x: float, y: float) -> float:
    n = math.sin(x * 127.1 + y * 311.7) * 43758.5453
    return n - math.floor(n)


def _sample_lum(height: np.ndarray, size: int, x: int, y: int) -> float:
    return float(height[((y % size) + size) % size, ((x % size) + size) % size])


def bump_to_normal(height: np.ndarray, size: int, strength: float) -> np.ndarray:
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            h_l = _sample_lum(height, size, x - 1, y)
            h_r = _sample_lum(height, size, x + 1, y)
            h_d = _sample_lum(height, size, x, y - 1)
            h_u = _sample_lum(height, size, x, y + 1)
            nx = (h_l - h_r) * strength
            ny = (h_d - h_u) * strength
            nz = 1.0
            length = math.hypot(nx, ny, nz) or 1.0
            rgba[y, x, 0] = _clamp_byte((nx / length * 0.5 + 0.5) * 255)
            rgba[y, x, 1] = _clamp_byte((ny / length * 0.5 + 0.5) * 255)
            rgba[y, x, 2] = _clamp_byte((nz / length * 0.5 + 0.5) * 255)
            rgba[y, x, 3] = 255
    return rgba.reshape(-1)


def _height_field(kind: str, size: int) -> np.ndarray:
    h = np.empty((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            u, v = x / size, y / size
            z = 0.5
            if kind == "brushed":
                streak = _hash2(math.floor(x * 0.055), y * 0.016)
                grain = _hash2(x * 0.85, math.floor(y / 2) * 2.4)
                z = 0.5 + (streak - 0.5) * 0.28 + (grain - 0.5) * 0.1
            elif kind == "carbon":
                cx, cy = x // 10, y // 10
                z = 0.46 if ((cx + cy) & 1) == 0 else 0.54
                z += (_hash2(x, y) - 0.5) * 0.04
            elif kind == "wood":
                dx, dy = u - 0.5, (v - 0.5) * 0.35
                rad = math.hypot(dx, dy) * 18 + _hash2(x * 0.2, y * 0.2) * 0.8
                z = 0.5 + 0.1 * math.sin(rad * 6.2) + (_hash2(x, y) - 0.5) * 0.04
            elif kind == "knurl":
                z = 0.5 + 0.16 * math.sin(u * 42) * math.sin(v * 42)
            elif kind == "cast":
                z = 0.46 + _hash2(x * 0.7, y * 0.7) * 0.16
            elif kind == "rubber":
                z = 0.48 + _hash2(x * 1.3, y * 1.1) * 0.08
            elif kind == "leather":
                z = 0.48 + _hash2(x // 3, y // 3) * 0.12
            elif kind == "fabric":
                z = 0.5 + 0.06 * math.sin(x * 0.9) + 0.06 * math.sin(y * 0.9)
            elif kind == "stone":
                z = 0.48 + _hash2(x * 0.2, y * 0.2) * 0.1 + _hash2(x, y) * 0.05
            elif kind == "speckle":
                z = 0.5 + _hash2(x, y) * 0.08
            else:
                z = 0.5 + (_hash2(x, y) - 0.5) * 0.1
            h[y, x] = _clamp01(z)
    return h


def _albedo_from_kind(kind: str, name: str, tint: Rgb) -> CadTexture:
    size = PROC_SIZE
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            u, v = x / size, y / size
            r, g, b = tint[0] * 255, tint[1] * 255, tint[2] * 255
            if kind == "carbon":
                weave = 0.16 if (((x // 10) + (y // 10)) & 1) == 0 else 0.42
                r = g = weave * 48
                b = weave * 56
            elif kind == "wood":
                dx, dy = u - 0.5, (v - 0.5) * 0.35
                rad = math.hypot(dx, dy) * 18 + _hash2(x * 0.2, y * 0.2) * 0.8
                lum = 0.55 + 0.22 * math.sin(rad * 6.2) + _hash2(x, y) * 0.08
                r, g, b = lum * 210, lum * 140, lum * 70
            elif kind == "checker":
                on = ((int(u * 8) + int(v * 8)) & 1) == 0
                r = g = b = 48 if on else 220
            elif kind == "leather":
                lum = 0.45 + _hash2(x // 3, y // 3) * 0.28
                r, g, b = lum * 160, lum * 95, lum * 55
            elif kind == "fabric":
                lum = 0.55 + 0.12 * math.sin(x * 0.9) + 0.12 * math.sin(y * 0.9)
                r, g, b = tint[0] * lum * 255, tint[1] * lum * 255, tint[2] * lum * 255
            elif kind == "stone":
                lum = 0.5 + _hash2(x * 0.2, y * 0.2) * 0.28
                r = g = b = lum * 255
            elif kind == "speckle":
                spec = 0.78 + _hash2(x, y) * 0.22
                r, g, b = tint[0] * spec * 255, tint[1] * spec * 255, tint[2] * spec * 255
            rgba[y, x] = (_clamp_byte(r), _clamp_byte(g), _clamp_byte(b), 255)
    return CadTexture(
        id=f"proc:{kind}:{name}:albedo",
        name=f"{name} albedo",
        width=size,
        height=size,
        rgba=rgba.reshape(-1),
        role="albedo",
    )


def _roughness_from_height(height: np.ndarray, size: int) -> np.ndarray:
    rgba = np.empty((size * size, 4), dtype=np.uint8)
    flat = height.reshape(-1)
    for i, h in enumerate(flat):
        mod = _clamp01(0.72 + (0.5 - float(h)) * 0.42)
        g = _clamp_byte(mod * 255)
        rgba[i] = (0, g, 255, 255)
    return rgba.reshape(-1)


def intern_texture(catalog: list[CadTexture], tex: CadTexture) -> int:
    for i, t in enumerate(catalog):
        if t.id == tex.id:
            return i
    catalog.append(tex)
    return len(catalog) - 1


def intern_pbr(materials: list[CadPbr], mat: CadPbr) -> int:
    for i, m in enumerate(materials):
        if m.id == mat.id:
            return i
    materials.append(mat)
    return len(materials) - 1


def make_pbr_from_profile(
    profile: PbrProfile,
    name: str,
    tint: Rgb,
    catalog: list[CadTexture],
    bump_amount: float = 0.3,
) -> CadPbr:
    amount = sw_bump_amount(bump_amount, 0.3 if profile.bump else 0)
    mat = CadPbr(
        id=f"pbr:{profile.kind}:{name}:b{round(amount * 100)}",
        name=name,
        metalness=profile.metalness,
        roughness=profile.roughness,
        normal_scale=amount,
    )
    if profile.albedo:
        mat.albedo = intern_texture(catalog, _albedo_from_kind(profile.kind, name, tint))
    if profile.bump and amount > 0.02:
        height = _height_field(profile.kind, PROC_SIZE)
        mat.normal = intern_texture(
            catalog,
            CadTexture(
                id=f"proc:{profile.kind}:{name}:normal",
                name=f"{name} bump",
                width=PROC_SIZE,
                height=PROC_SIZE,
                rgba=bump_to_normal(height, PROC_SIZE, 1),
                role="normal",
            ),
        )
        mat.roughness_map = intern_texture(
            catalog,
            CadTexture(
                id=f"proc:{profile.kind}:{name}:rough",
                name=f"{name} rough",
                width=PROC_SIZE,
                height=PROC_SIZE,
                rgba=_roughness_from_height(height, PROC_SIZE),
                role="roughness",
            ),
        )
    return mat


def raster_to_pbr(
    image: CadTexture,
    name: str,
    catalog: list[CadTexture],
    hinted_role: TexRole | None = None,
    bump_amount: float = 0.3,
) -> CadPbr:
    role = hinted_role or classify_raster_role(image.name or name)
    tex = CadTexture(
        id=image.id,
        name=image.name or name,
        width=image.width,
        height=image.height,
        rgba=image.rgba,
        role=role,
        encoded=image.encoded,
        mime=image.mime,
    )
    idx = intern_texture(catalog, tex)
    amount = sw_bump_amount(bump_amount, 0.3 if role == "normal" else 1)
    mat = CadPbr(
        id=f"pbr:img:{tex.id}:{role}:b{round(amount * 100)}",
        name=name,
        metalness=0.55 if role == "normal" else 0.08,
        roughness=0.38 if role == "normal" else 0.5,
        normal_scale=amount if role == "normal" else 1.0,
    )
    if role == "normal":
        mat.normal = idx
    elif role == "roughness":
        mat.roughness_map = idx
    elif role == "metalness":
        mat.metalness_map = idx
    else:
        mat.albedo = idx
    return mat


def _png_end(data: bytes, start: int) -> int:
    iend = b"\x49\x45\x4e\x44\xae\x42\x60\x82"
    hit = index_of_seq(data, iend, start + 16)
    return -1 if hit < 0 else hit + len(iend)


def _jpeg_end(data: bytes, start: int) -> int:
    i = start + 20
    last = len(data) - 1
    while i < last:
        hit = data.find(0xFF, i)
        if hit < 0 or hit >= last:
            return -1
        if data[hit + 1] == 0xD9:
            return hit + 2
        i = hit + 1
    return -1


def extract_raster_images(streams: dict[str, bytes]) -> list[CadTexture]:
    out: list[CadTexture] = []
    for key, blob in streams.items():
        low = key.lower()
        if "preview" in low:
            continue
        if "displaylist" in low:
            continue
        i = 0
        limit = len(blob) - 24
        while i < limit:
            png_at = blob.find(0x89, i)
            jpg_at = blob.find(0xFF, i)
            if png_at < 0 and jpg_at < 0:
                break
            use_png = png_at >= 0 and (jpg_at < 0 or png_at <= jpg_at)
            if use_png:
                if png_at + 3 < len(blob) and blob[png_at + 1 : png_at + 4] == b"PNG":
                    end = _png_end(blob, png_at)
                    if end > png_at + 80:
                        name = key.split("/")[-1]
                        out.append(
                            CadTexture(
                                id=f"img:{key}:{png_at}",
                                name=name,
                                width=0,
                                height=0,
                                rgba=None,
                                encoded=blob[png_at:end],
                                mime="image/png",
                                role=classify_raster_role(name),
                            )
                        )
                        i = end
                        continue
                i = png_at + 1
                continue
            if jpg_at >= 0 and jpg_at + 2 < len(blob) and blob[jpg_at + 1] == 0xD8 and blob[jpg_at + 2] == 0xFF and len(blob) - jpg_at > 400:
                end = _jpeg_end(blob, jpg_at)
                if end > jpg_at + 400:
                    name = key.split("/")[-1]
                    out.append(
                        CadTexture(
                            id=f"jpg:{key}:{jpg_at}",
                            name=name,
                            width=0,
                            height=0,
                            rgba=None,
                            encoded=blob[jpg_at:end],
                            mime="image/jpeg",
                            role=classify_raster_role(name),
                        )
                    )
                    i = end
                    continue
            i = (jpg_at if jpg_at >= 0 else i) + 1
    return out


def decode_texture(tex: CadTexture) -> CadTexture:
    if tex.width > 0 and tex.height > 0 and tex.rgba is not None:
        return tex
    raw = tex.encoded
    if not raw:
        return tex
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(BytesIO(raw)).convert("RGBA")
        arr = np.asarray(im, dtype=np.uint8)
        tex.width, tex.height = im.size
        tex.rgba = arr.reshape(-1)
    except Exception:
        pass
    return tex


def is_mapping_size(n: float) -> bool:
    a = abs(n)
    return math.isfinite(n) and 1e-4 <= a < 5 and abs(a - 1) > 1e-3


def mapping_length(metres: float, pos_diag: float) -> float:
    if not math.isfinite(metres) or metres == 0:
        return metres if math.isfinite(metres) else 0.0
    sign = -1 if metres < 0 else 1
    s = abs(metres)
    mesh_in_mm = math.isfinite(pos_diag) and pos_diag > 40
    if mesh_in_mm and s < 5:
        s *= 1000
    elif not mesh_in_mm and 5 <= s < 4000:
        s /= 1000
    return sign * s


def mapping_scale(value: float, pos_diag: float) -> float:
    if not math.isfinite(value) or value == 0:
        return mapping_length(0.1, pos_diag) or 0.1
    s = mapping_length(value, pos_diag)
    return 0.1 if s == 0 else s


def _unit_axis(v: Vec3 | None) -> Vec3 | None:
    if not v:
        return None
    length = math.hypot(v[0], v[1], v[2])
    if length < 0.5 or length > 1.5:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def _read_utf16z(data: bytes, off: int, max_chars: int = 240) -> tuple[str, int]:
    text: list[str] = []
    i = off
    for _ in range(max_chars):
        if i + 1 >= len(data):
            break
        c = data[i] | (data[i + 1] << 8)
        i += 2
        if c == 0 or c < 32 or c > 126:
            break
        text.append(chr(c))
    return "".join(text), i


def _is_near_90(ang: float) -> bool:
    return math.isfinite(ang) and abs(abs(ang) - 90) < 0.51


def _plausible_tile(n: float) -> bool:
    return is_mapping_size(n)


def _plausible_origin(n: float) -> bool:
    return math.isfinite(n) and abs(n) < 20


def _skip_mfc_string(data: bytes, off: int) -> int:
    if off + 4 > len(data):
        return off
    if data[off] != 0xFF or data[off + 1] != 0xFE or data[off + 2] != 0xFF:
        return off
    n = data[off + 3]
    nxt = off + 4 + n * 2
    return nxt if nxt <= len(data) else off


def _mapping_from_body(data: bytes, p: int) -> CadMapping | None:
    if p + 76 > len(data):
        return None
    width, height = struct.unpack_from("<ff", data, p)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = struct.unpack_from("<fffffffff", data, p + 24)
    ang, sx, sy, sz = struct.unpack_from("<ffff", data, p + 60)
    has90 = _is_near_90(ang)
    has_ident = abs(r00 - 1) < 0.08 and abs(r11 - 1) < 0.08 and abs(abs(r22) - 1) < 0.08
    has_rot = _unit_axis((r00, r01, r02)) and _unit_axis((r10, r11, r12))
    if not has90 and not has_ident and not has_rot:
        return None
    u = abs(width) if _plausible_tile(width) else 0.0
    v = abs(height) if _plausible_tile(height) else 0.0
    if not u and _plausible_tile(sy):
        u = abs(sy)
    if not v and _plausible_tile(sz):
        v = abs(sz)
    if not u and _plausible_tile(sx):
        u = abs(sx)
    if not v and _plausible_tile(sx):
        v = abs(sx)
    if not u and v:
        u = v
    if not v and u:
        v = u
    if not u:
        u = 0.1
    if not v:
        v = u
    flip_u = sx < 0 and not _plausible_tile(sx)
    flip_v = (sy < 0 and not _plausible_tile(sy)) or (sz < 0 and not _plausible_tile(sz))
    rotation = 0.0
    if math.isfinite(ang) and 0.5 <= abs(ang) <= 360 and not _is_near_90(ang):
        rotation = ang
    axis_u = _unit_axis((r00, r01, r02))
    axis_v = _unit_axis((r10, r11, r12))
    origin = None
    third = (r20, r21, r22)
    if not _unit_axis(third) and all(_plausible_origin(x) for x in third):
        origin = third
    return CadMapping(
        scale_u=-u if flip_u else u,
        scale_v=-v if flip_v else v,
        rotation_deg=rotation,
        axis_u=axis_u,
        axis_v=axis_v,
        origin=origin,
    )


def _parse_mapping_block(data: bytes, nxt: int) -> CadMapping:
    fallback = CadMapping()
    if nxt + 110 > len(data):
        return fallback
    p = nxt
    while p < len(data) and data[p] == 0xFF:
        p += 1
    if p < len(data) and data[p] == 0x00:
        p += 1
    if p + 16 <= len(data):
        blend, ior = struct.unpack_from("<ff", data, p)
        if 0 <= blend <= 1.01 and 0.5 <= ior <= 3:
            p += 8
            p = _skip_mfc_string(data, p)
    aligned = _mapping_from_body(data, p)
    if aligned:
        return aligned
    for delta in range(-16, 52, 4):
        q = p + delta
        if q < 0 or q + 64 > len(data):
            continue
        ang = struct.unpack_from("<f", data, q + 60)[0]
        if not _is_near_90(ang):
            continue
        hit = _mapping_from_body(data, q)
        if hit:
            return hit
    return fallback


def parse_texture_looks(data: bytes) -> list[TextureLook]:
    out: list[TextureLook] = []
    needles = ["<SystemTexture>", ".p2m", ".jpg", ".jpeg", ".png", ".bmp", ".tif"]
    seen: set[int] = set()
    for needle in needles:
        from_ = 0
        while from_ < len(data):
            hit = index_of_utf16(data, needle, from_)
            if hit < 0:
                break
            from_ = hit + 2
            start = hit
            if needle != "<SystemTexture>":
                while start >= 2:
                    c = data[start - 2] | (data[start - 1] << 8)
                    if c < 32 or c > 126:
                        break
                    start -= 2
                    if hit - start > 400:
                        break
            text, z_next = _read_utf16z(data, start)
            if len(text) < 5 or start in seen:
                continue
            if not re.search(r"\.(p2m|jpe?g|png|bmp|tiff?)$", text, re.I) and "<SystemTexture>" not in text:
                continue
            seen.add(start)
            nxt = z_next
            mfc_at = start - 4
            if mfc_at >= 0 and data[mfc_at] == 0xFF and data[mfc_at + 1] == 0xFE and data[mfc_at + 2] == 0xFF:
                n = data[mfc_at + 3]
                if text.__len__() <= n <= 240:
                    nxt = start + n * 2
            elif nxt >= 2 and data[nxt - 2] == 0xFF and data[nxt - 1] == 0x00:
                nxt -= 2
            mapped = _parse_mapping_block(data, nxt)
            out.append(
                TextureLook(
                    off=start,
                    path=text,
                    scale_u=mapped.scale_u,
                    scale_v=mapped.scale_v,
                    rotation_deg=mapped.rotation_deg,
                    origin=mapped.origin,
                    axis_u=mapped.axis_u,
                    axis_v=mapped.axis_v,
                    map_type=mapped.map_type,
                )
            )
    out.sort(key=lambda t: t.off)
    return out


def project_box_uvs(
    positions: np.ndarray,
    normals: np.ndarray | None,
    scale_u: float,
    scale_v: float,
    rotation_deg: float,
    mapping: CadMapping | None = None,
) -> np.ndarray:
    n = positions.size // 3
    uvs = np.empty(n * 2, dtype=np.float32)
    rad = rotation_deg * math.pi / 180
    c, s = math.cos(rad), math.sin(rad)
    su, sv = scale_u or 0.1, scale_v or 0.1
    ox = mapping.origin[0] if mapping and mapping.origin else 0.0
    oy = mapping.origin[1] if mapping and mapping.origin else 0.0
    oz = mapping.origin[2] if mapping and mapping.origin else 0.0
    axis_u = _unit_axis(mapping.axis_u) if mapping else None
    axis_v = _unit_axis(mapping.axis_v) if mapping else None
    have_axes = bool(axis_u and axis_v)
    ux = uy = uz = 0.0
    vx = vy = vz = 0.0
    wx = wy = wz = 0.0
    if axis_u and axis_v:
        ux, uy, uz = axis_u
        vx, vy, vz = axis_v
        wx = uy * vz - uz * vy
        wy = uz * vx - ux * vz
        wz = ux * vy - uy * vx
        w_len = math.hypot(wx, wy, wz) or 1.0
        wx, wy, wz = wx / w_len, wy / w_len, wz / w_len
    planar = have_axes and (mapping.map_type or 0) >= 2 if mapping else False
    for i in range(n):
        px = float(positions[i * 3]) - ox
        py = float(positions[i * 3 + 1]) - oy
        pz = float(positions[i * 3 + 2]) - oz
        nx = ny = 0.0
        nz = 1.0
        if normals is not None and normals.size == positions.size:
            nx, ny, nz = float(normals[i * 3]), float(normals[i * 3 + 1]), float(normals[i * 3 + 2])
        if have_axes:
            pu = px * ux + py * uy + pz * uz
            pv = px * vx + py * vy + pz * vz
            if planar:
                u, v = pu / su, pv / sv
            else:
                pw = px * wx + py * wy + pz * wz
                nu = nx * ux + ny * uy + nz * uz
                nv = nx * vx + ny * vy + nz * vz
                nw = nx * wx + ny * wy + nz * wz
                au, av, aw = abs(nu), abs(nv), abs(nw)
                if aw >= au and aw >= av:
                    u, v = pu / su, pv / sv
                elif av >= au:
                    u, v = pu / su, pw / sv
                else:
                    u, v = pv / su, pw / sv
        else:
            ax, ay, az = abs(nx), abs(ny), abs(nz)
            if az >= ax and az >= ay:
                u, v = px / su, py / sv
            elif ay >= ax:
                u, v = px / su, pz / sv
            else:
                u, v = py / su, pz / sv
        uvs[i * 2] = u * c - v * s
        uvs[i * 2 + 1] = u * s + v * c
    return uvs


def fill_tex_index(count: int, index: int) -> np.ndarray:
    a = np.empty(count, dtype=np.uint16)
    a.fill(NONE if index < 0 else index)
    return a


@dataclass
class BindLook:
    name: str
    color: Rgb | None = None
    metalness: float | None = None
    roughness: float | None = None
    off: int | None = None
    path: str | None = None
    scale_u: float | None = None
    scale_v: float | None = None
    rotation_deg: float | None = None
    bump_amount: float | None = None
    origin: Vec3 | None = None
    axis_u: Vec3 | None = None
    axis_v: Vec3 | None = None
    map_type: int | None = None
    pbr: CadPbr | None = None


def bind_appearance_textures(
    looks: list[BindLook],
    mapped: list[TextureLook],
    images: list[CadTexture],
    catalog: list[CadTexture],
    materials: list[CadPbr],
) -> None:
    by_base = {img.name.lower(): img for img in images}

    def attach(look: BindLook, path: str, mapped_look: TextureLook | None) -> None:
        look.path = path
        if mapped_look:
            if look.scale_u is None:
                look.scale_u = mapped_look.scale_u
            if look.scale_v is None:
                look.scale_v = mapped_look.scale_v
            if look.rotation_deg is None:
                look.rotation_deg = mapped_look.rotation_deg
            if look.bump_amount is None:
                look.bump_amount = mapped_look.bump_amount
            if not look.axis_u and mapped_look.axis_u:
                look.axis_u = mapped_look.axis_u
            if not look.axis_v and mapped_look.axis_v:
                look.axis_v = mapped_look.axis_v
            if not look.origin and mapped_look.origin:
                look.origin = mapped_look.origin
            if look.map_type is None and mapped_look.map_type is not None:
                look.map_type = mapped_look.map_type
        file_name = path.replace("\\", "/").split("/")[-1].lower()
        image = by_base.get(file_name)
        if image is None and file_name:
            stem = re.sub(r"\.[^.]+$", "", file_name)
            image = next((im for im in images if stem and stem in im.name.lower()), None)
        profile = pbr_profile(path) or pbr_profile(look.name or "")
        tint: Rgb = look.color or (0.72, 0.74, 0.78)
        bump = sw_bump_amount(look.bump_amount, 0.3 if profile and profile.bump else 0)
        pbr: CadPbr | None = None
        encoded_len = len(image.encoded) if image and image.encoded else (image.rgba.size if image and image.rgba is not None else 0)
        if image and encoded_len > 80:
            role = classify_raster_role(image.name + " " + path + " " + (look.name or ""))
            if profile and not profile.albedo and profile.bump:
                role = "normal"
            pbr = raster_to_pbr(image, look.name or file_name, catalog, role, bump)
            if profile:
                pbr.metalness = profile.metalness
                pbr.roughness = profile.roughness
                pbr.normal_scale = bump
                if not profile.albedo:
                    pbr.albedo = None
                if profile.bump and pbr.normal is None and bump > 0.02:
                    kit = make_pbr_from_profile(profile, look.name or file_name, tint, catalog, bump)
                    pbr.normal = kit.normal
                    pbr.roughness_map = kit.roughness_map
                    pbr.normal_scale = kit.normal_scale
        elif profile:
            pbr = make_pbr_from_profile(profile, look.name or file_name or profile.kind, tint, catalog, bump)
        if pbr:
            if look.metalness is not None and profile and profile.kind == "brushed":
                pbr.metalness = max(pbr.metalness, look.metalness)
            intern_pbr(materials, pbr)
            look.pbr = pbr

    for look in looks:
        look_norm = _norm_name(look.name)
        nearby = next((m for m in mapped if look.off is not None and abs(m.off - look.off) < 800), None)
        if nearby is None:
            for m in mapped:
                base = _norm_name((m.path.replace("\\", "/").split("/")[-1]).replace(".p2m", ""))
                if len(base) > 3 and (base in look_norm or look_norm in base):
                    nearby = m
                    break
        if nearby is None and len(look_norm) > 4:
            nearby = next((m for m in mapped if look_norm in _norm_name(m.path)), None)
        attach(look, nearby.path if nearby else (look.path or look.name), nearby)

    named: dict[str, CadPbr] = {}
    by_color: dict[str, CadPbr] = {}

    def color_key(c: Rgb | None) -> str:
        if not c:
            return ""
        return f"{round(c[0] * 10)}|{round(c[1] * 10)}|{round(c[2] * 10)}"

    for look in looks:
        if not look.pbr:
            continue
        if look.name:
            named[_norm_name(look.name)] = look.pbr
        k = color_key(look.color)
        if k:
            by_color[k] = look.pbr
    for look in looks:
        if look.pbr:
            continue
        n = _norm_name(look.name) if look.name else ""
        look.pbr = (named.get(n) if n else None) or by_color.get(color_key(look.color))
