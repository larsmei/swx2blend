# SPDX-License-Identifier: MIT
"""swx2blend — import SolidWorks .sldprt / .sldasm into Blender."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import bpy
from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import Context, Operator
from bpy_extras.io_utils import ImportHelper, axis_conversion, orientation_helper
from mathutils import Matrix

from .swx.convert import ConvertPart, ConvertResult, convert_solidworks
from .swx.assembly import SceneNode
from .swx.textures import CadPbr, CadTexture, NONE

bl_info = {
    "name": "swx2blend",
    "author": "larsmei",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > SolidWorks (.sldprt/.sldasm)",
    "description": "Import SolidWorks parts and assemblies with materials and hierarchy",
    "doc_url": "https://github.com/larsmei/swx2blend",
    "category": "Import-Export",
}


def _sw_matrix(m) -> Matrix:
    return Matrix(
        (
            (m[0], m[4], m[8], m[12]),
            (m[1], m[5], m[9], m[13]),
            (m[2], m[6], m[10], m[14]),
            (m[3], m[7], m[11], m[15]),
        )
    )


def _image_from_texture(tex: CadTexture, cache: dict[str, bpy.types.Image]) -> bpy.types.Image | None:
    hit = cache.get(tex.id)
    if hit:
        return hit
    img = None
    raw = tex.encoded
    if raw:
        suffix = ".png"
        if (tex.mime or "").endswith("jpeg") or raw[:2] == b"\xff\xd8":
            suffix = ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="swx2blend_")
        try:
            os.write(fd, raw)
            os.close(fd)
            fd = -1
            img = bpy.data.images.load(path)
            img.pack()
            img.name = (tex.name or tex.id)[:63]
        except Exception:
            img = None
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(path)
            except OSError:
                pass
    if img is None and tex.rgba is not None and tex.width > 0 and tex.height > 0:
        import numpy as np

        img = bpy.data.images.new(tex.name or tex.id, tex.width, tex.height, alpha=True)
        arr = np.asarray(tex.rgba, dtype=np.uint8).reshape(tex.height, tex.width, 4)
        arr = np.flipud(arr).astype(np.float32) / 255.0
        img.pixels.foreach_set(arr.reshape(-1))
        img.pack()
    if img is None:
        return None
    img.colorspace_settings.name = "sRGB" if tex.role == "albedo" else "Non-Color"
    cache[tex.id] = img
    return img


def _slot_image(result: ConvertResult, idx: int | None, cache: dict[str, bpy.types.Image]) -> bpy.types.Image | None:
    if idx is None or idx < 0 or idx >= len(result.textures):
        return None
    return _image_from_texture(result.textures[idx], cache)


def _principled_material(
    pbr: CadPbr | None,
    name: str,
    tint: tuple[float, float, float],
    result: ConvertResult,
    images: dict[str, bpy.types.Image],
    metalness: float,
    roughness: float,
    vertex_color: bool,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name[:63] or "SolidWorks")
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    out = tree.nodes.get("Material Output")
    if not bsdf:
        return mat
    bsdf.location = (0, 0)
    color = (*tint, 1.0)
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = color
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(pbr.metalness if pbr else metalness)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(pbr.roughness if pbr else roughness)

    tex_coord = None

    def uv_socket():
        nonlocal tex_coord
        if tex_coord is None:
            tex_coord = tree.nodes.new("ShaderNodeTexCoord")
            tex_coord.location = (-720, 80)
        return tex_coord.outputs["UV"]

    def image_node(img: bpy.types.Image, loc):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.location = loc
        tree.links.new(uv_socket(), node.inputs["Vector"])
        return node

    albedo = _slot_image(result, pbr.albedo if pbr else None, images)
    if albedo:
        node = image_node(albedo, (-420, 280))
        tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    elif vertex_color:
        col_attr = tree.nodes.new("ShaderNodeVertexColor")
        col_attr.layer_name = "Col"
        col_attr.location = (-420, 280)
        tree.links.new(col_attr.outputs["Color"], bsdf.inputs["Base Color"])

    normal_img = _slot_image(result, pbr.normal if pbr else None, images)
    if normal_img and "Normal" in bsdf.inputs:
        node = image_node(normal_img, (-420, -40))
        node.image.colorspace_settings.name = "Non-Color"
        nrm = tree.nodes.new("ShaderNodeNormalMap")
        nrm.location = (-180, -40)
        scale = pbr.normal_scale if pbr and pbr.normal_scale is not None else 1.0
        if "Strength" in nrm.inputs:
            nrm.inputs["Strength"].default_value = float(scale)
        tree.links.new(node.outputs["Color"], nrm.inputs["Color"])
        tree.links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])

    rough_img = _slot_image(result, pbr.roughness_map if pbr else None, images)
    if rough_img and "Roughness" in bsdf.inputs:
        node = image_node(rough_img, (-420, -260))
        node.image.colorspace_settings.name = "Non-Color"
        sep = tree.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-180, -260)
        tree.links.new(node.outputs["Color"], sep.inputs["Color"])
        tree.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])

    metal_img = _slot_image(result, pbr.metalness_map if pbr else None, images)
    if metal_img and "Metallic" in bsdf.inputs:
        node = image_node(metal_img, (-420, -460))
        node.image.colorspace_settings.name = "Non-Color"
        sep = tree.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-180, -460)
        tree.links.new(node.outputs["Color"], sep.inputs["Color"])
        tree.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    if out:
        out.location = (280, 0)
    return mat


def _fallback_tint(part: ConvertPart) -> tuple[float, float, float]:
    if part.colors is not None and part.colors.size >= 3:
        return (float(part.colors[0]), float(part.colors[1]), float(part.colors[2]))
    return (0.718, 0.745, 0.784)


def _mesh_from_part(
    part: ConvertPart,
    object_name: str,
    result: ConvertResult,
    images: dict[str, bpy.types.Image],
    import_materials: bool,
) -> bpy.types.Mesh:
    import numpy as np

    n_verts = part.positions.size // 3
    n_tris = part.indices.size // 3
    mesh = bpy.data.meshes.new(object_name)
    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", np.asarray(part.positions, dtype=np.float32))
    mesh.loops.add(n_tris * 3)
    mesh.loops.foreach_set("vertex_index", np.asarray(part.indices, dtype=np.int32))
    mesh.polygons.add(n_tris)
    loop_start = np.arange(0, n_tris * 3, 3, dtype=np.int32)
    loop_total = np.full(n_tris, 3, dtype=np.int32)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    mesh.update()
    mesh.validate()
    if part.normals is not None and part.normals.size == part.positions.size:
        mesh.shade_smooth()
        try:
            nrm = np.asarray(part.normals, dtype=np.float32).reshape(-1, 3)
            mesh.normals_split_custom_set_from_vertices(nrm.tolist())
        except Exception:
            pass
    has_color = part.colors is not None and part.colors.size == part.positions.size
    if has_color:
        attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
        cols = np.ones((n_verts, 4), dtype=np.float32)
        rgb = np.asarray(part.colors, dtype=np.float32).reshape(-1, 3)
        cols[:, :3] = rgb
        attr.data.foreach_set("color", cols.reshape(-1))
    if part.uvs is not None and part.uvs.size == n_verts * 2:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        loop_uvs = np.empty(n_tris * 3 * 2, dtype=np.float32)
        idx = np.asarray(part.indices, dtype=np.int64)
        uv = np.asarray(part.uvs, dtype=np.float32).reshape(-1, 2)
        loop_uvs[0::2] = uv[idx, 0]
        loop_uvs[1::2] = uv[idx, 1]
        uv_layer.data.foreach_set("uv", loop_uvs)

    tint = _fallback_tint(part)
    used: list[int] = []
    slot_of: dict[int, int] = {}
    tex = part.tex_index if part.tex_index is not None and part.tex_index.size == n_verts else None
    if import_materials and tex is not None and result.materials:
        for i in range(n_tris):
            t = int(tex[int(part.indices[i * 3])])
            if t == NONE or t < 0 or t >= len(result.materials):
                t = -1
            if t not in slot_of:
                slot_of[t] = len(used)
                used.append(t)
        if not used:
            used = [-1]
            slot_of[-1] = 0
        for cad_idx in used:
            pbr = result.materials[cad_idx] if cad_idx >= 0 else None
            label = pbr.name if pbr and pbr.name else object_name
            mesh.materials.append(
                _principled_material(
                    pbr,
                    label,
                    tint,
                    result,
                    images,
                    part.metalness,
                    part.roughness,
                    has_color and (pbr is None or pbr.albedo is None),
                )
            )
        poly_mat = np.zeros(n_tris, dtype=np.int32)
        for i in range(n_tris):
            t = int(tex[int(part.indices[i * 3])])
            if t == NONE or t < 0 or t >= len(result.materials):
                t = -1
            poly_mat[i] = slot_of.get(t, 0)
        mesh.polygons.foreach_set("material_index", poly_mat)
    else:
        mesh.materials.append(
            _principled_material(
                result.materials[0] if import_materials and result.materials else None,
                object_name,
                tint,
                result,
                images,
                part.metalness,
                part.roughness,
                has_color and (
                    not import_materials
                    or not result.materials
                    or result.materials[0].albedo is None
                ),
            )
        )
    return mesh


def _unique_object(name: str, data):
    obj = bpy.data.objects.new(name, data)
    return obj


def _build_tree(
    node: SceneNode,
    parent,
    collection,
    meshes: dict[str, bpy.types.Mesh],
    result: ConvertResult,
    images: dict[str, bpy.types.Image],
    import_materials: bool,
) -> None:
    if node.part and node.part in result.parts:
        mesh = meshes.get(node.part)
        if mesh is None:
            mesh = _mesh_from_part(result.parts[node.part], node.part, result, images, import_materials)
            meshes[node.part] = mesh
        obj = _unique_object(node.name, mesh)
    else:
        obj = _unique_object(node.name, None)
        obj.empty_display_type = "PLAIN_AXES"
        obj.empty_display_size = 0.02
    collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
        obj.matrix_parent_inverse.identity()
    obj.matrix_local = _sw_matrix(node.local_transform)
    for child in node.children:
        _build_tree(child, obj, collection, meshes, result, images, import_materials)


def _import_result(
    context: Context,
    result: ConvertResult,
    object_name: str,
    global_matrix: Matrix,
    keep_hierarchy: bool,
    import_materials: bool,
):
    images: dict[str, bpy.types.Image] = {}
    collection = context.collection
    if keep_hierarchy and result.tree and result.parts:
        root = _unique_object(object_name, None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.04
        collection.objects.link(root)
        root.matrix_world = global_matrix
        meshes: dict[str, bpy.types.Mesh] = {}
        for child in result.tree.children:
            _build_tree(child, root, collection, meshes, result, images, import_materials)
        if not result.tree.children and result.parts:
            # assembly with a single unnamed child — instance the parts under root
            for part in result.parts.values():
                mesh = _mesh_from_part(part, part.name, result, images, import_materials)
                obj = _unique_object(part.name, mesh)
                obj.parent = root
                collection.objects.link(obj)
        context.view_layer.objects.active = root
        root.select_set(True)
        return root
    part = next(iter(result.parts.values()), None)
    if part is None:
        part = ConvertPart(
            name=object_name,
            positions=result.positions,
            normals=result.normals,
            indices=result.indices,
            colors=result.colors,
            uvs=result.uvs,
            tex_index=result.tex_index,
            metalness=result.metalness,
            roughness=result.roughness,
        )
    mesh = _mesh_from_part(part, object_name, result, images, import_materials)
    obj = _unique_object(object_name, mesh)
    obj.matrix_world = global_matrix
    collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


@orientation_helper(axis_forward="Y", axis_up="Z")
class IMPORT_SCENE_OT_solidworks(Operator, ImportHelper):
    """Import a SolidWorks part or assembly as a mesh."""

    bl_idname = "import_scene.solidworks"
    bl_label = "Import SolidWorks"
    bl_options = {"PRESET", "UNDO"}

    filename_ext = ".sldprt"
    filter_glob: StringProperty(default="*.sldprt;*.sldasm;*.SLDPRT;*.SLDASM", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})
    weld: BoolProperty(
        name="Weld vertices",
        description="Merge coincident vertices after tessellation",
        default=True,
    )
    keep_hierarchy: BoolProperty(
        name="Keep assembly structure",
        description="Nested empties for subassemblies, linked mesh datablocks per part",
        default=True,
    )
    import_materials: BoolProperty(
        name="Import materials",
        description="Principled BSDF from SOLIDWORKS appearances, packed images and procedural maps",
        default=True,
    )

    def execute(self, context: Context):
        paths: list[Path] = []
        if self.files and self.directory:
            for entry in self.files:
                paths.append(Path(self.directory) / entry.name)
        elif self.filepath:
            paths.append(Path(self.filepath))
        if not paths:
            self.report({"ERROR"}, "Keine Datei ausgewählt")
            return {"CANCELLED"}

        global_matrix = axis_conversion(
            from_forward="Y",
            from_up="Z",
            to_forward=self.axis_forward,
            to_up=self.axis_up,
        ).to_4x4()

        imported = 0
        for path in paths:
            if not path.is_file():
                self.report({"WARNING"}, f"Datei nicht gefunden: {path}")
                continue
            try:
                result = convert_solidworks(path.read_bytes(), path.stem, weld=self.weld)
            except Exception as exc:
                self.report({"ERROR"}, f"{path.name}: {exc}")
                continue
            _import_result(
                context,
                result,
                path.stem,
                global_matrix,
                self.keep_hierarchy,
                self.import_materials,
            )
            for note in result.notes:
                self.report({"INFO"}, note)
            extra = ""
            if result.part_count:
                extra += f", {result.part_count} Teile"
            if result.instance_count:
                extra += f", {result.instance_count} Instanzen"
            if result.materials:
                extra += f", {len(result.materials)} Materialien"
            self.report(
                {"INFO"},
                f"{path.name}: {result.output_triangles} Dreiecke, {result.output_vertices} Vertices ({result.units}){extra}",
            )
            imported += 1
        return {"FINISHED"} if imported else {"CANCELLED"}


def menu_func_import(self, _context):
    self.layout.operator(IMPORT_SCENE_OT_solidworks.bl_idname, text="SolidWorks (.sldprt/.sldasm)")


classes = (IMPORT_SCENE_OT_solidworks,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
