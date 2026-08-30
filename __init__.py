# SPDX-License-Identifier: MIT
"""swx2blend — import SolidWorks .sldprt / .sldasm into Blender."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import Context, Operator
from bpy_extras.io_utils import ImportHelper, axis_conversion, orientation_helper

from .swx.convert import convert_solidworks

bl_info = {
    "name": "swx2blend",
    "author": "larsmei",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > SolidWorks (.sldprt/.sldasm)",
    "description": "Import SolidWorks parts and assemblies directly — no SolidWorks install required",
    "doc_url": "https://github.com/larsmei/swx2blend",
    "category": "Import-Export",
}


def _mesh_from_result(result, object_name: str):
    import numpy as np

    n_verts = result.positions.size // 3
    n_tris = result.indices.size // 3
    mesh = bpy.data.meshes.new(object_name)
    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", np.asarray(result.positions, dtype=np.float32))
    mesh.loops.add(n_tris * 3)
    mesh.loops.foreach_set("vertex_index", np.asarray(result.indices, dtype=np.int32))
    mesh.polygons.add(n_tris)
    loop_start = np.arange(0, n_tris * 3, 3, dtype=np.int32)
    loop_total = np.full(n_tris, 3, dtype=np.int32)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    mesh.update()
    mesh.validate()
    if result.normals is not None and result.normals.size == result.positions.size:
        mesh.shade_smooth()
        try:
            nrm = np.asarray(result.normals, dtype=np.float32).reshape(-1, 3)
            mesh.normals_split_custom_set_from_vertices(nrm.tolist())
        except Exception:
            pass
    if result.colors is not None and result.colors.size == result.positions.size:
        attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
        cols = np.ones((n_verts, 4), dtype=np.float32)
        rgb = np.asarray(result.colors, dtype=np.float32).reshape(-1, 3)
        cols[:, :3] = rgb
        attr.data.foreach_set("color", cols.reshape(-1))
    mat = bpy.data.materials.new(object_name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    if bsdf:
        color = (0.718, 0.745, 0.784, 1.0)
        if result.colors is not None and result.colors.size >= 3:
            color = (float(result.colors[0]), float(result.colors[1]), float(result.colors[2]), 1.0)
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = float(result.metalness)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = float(result.roughness)
        if result.colors is not None:
            col_attr = tree.nodes.new("ShaderNodeVertexColor")
            col_attr.layer_name = "Col"
            col_attr.location = (-280, 200)
            tree.links.new(col_attr.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)
    return mesh


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
            mesh = _mesh_from_result(result, path.stem)
            obj = bpy.data.objects.new(path.stem, mesh)
            obj.matrix_world = global_matrix
            context.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)
            for note in result.notes:
                self.report({"INFO"}, note)
            self.report(
                {"INFO"},
                f"{path.name}: {result.output_triangles} Dreiecke, {result.output_vertices} Vertices ({result.units})",
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
