#!/usr/bin/env python3
import bpy
import sys
from pathlib import Path

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = []

if len(argv) != 2:
    print("Usage: blender --background --python convert_glb_to_obj.py -- input.glb output.obj")
    sys.exit(1)

src_glb = Path(argv[0]).expanduser().resolve()
dst_obj = Path(argv[1]).expanduser().resolve()
dst_dir = dst_obj.parent
dst_dir.mkdir(parents=True, exist_ok=True)

# Debug info
print(f"Source GLB: {src_glb}")
print(f"Destination OBJ: {dst_obj}")

if not src_glb.exists():
    print(f"ERROR: Source file not found: {src_glb}")
    sys.exit(1)

# Clear scene and import GLB
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(src_glb))

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if len(mesh_objs) > 1:
    print(f"Joining {len(mesh_objs)} mesh objects into one...")
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()
    print("Objects joined successfully")


bpy.ops.object.select_all(action='DESELECT')
merged_obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
merged_obj.select_set(True)
bpy.context.view_layer.objects.active = merged_obj

bpy.ops.object.mode_set(mode='EDIT')

bpy.ops.mesh.select_all(action='SELECT')

bpy.ops.mesh.remove_doubles(threshold=0.0001)

bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
bpy.ops.mesh.delete_loose()

bpy.ops.object.mode_set(mode='OBJECT')

print(f"Mesh cleaned: {len(merged_obj.data.vertices)} vertices")

print("\n--- Extracting textures ---")
texture_files = {}
for img in bpy.data.images:
    if img.source == 'FILE' or img.size[0] > 0:  # Skip empty images
        # Create a meaningful filename
        img_name = img.name.replace(':', '_').replace('/', '_').replace('\\', '_')
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'

        target = dst_dir / img_name
        print(f"Saving texture: {img.name} -> {target}")

        # Save the image
        img.filepath_raw = str(target)
        img.file_format = 'PNG'
        img.save()

        texture_files[img.name] = target

# Update materials to reference the exported textures
print("\n--- Updating material texture references ---")
for mat in bpy.data.materials:
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                img_name = node.image.name
                if img_name in texture_files:
                    # Update the image path to the exported file
                    node.image.filepath = str(texture_files[img_name])
                    print(f"Material '{mat.name}' updated with texture: {texture_files[img_name].name}")

# Export OBJ with materials
print("\n--- Exporting OBJ ---")
bpy.ops.wm.obj_export(
    filepath=str(dst_obj),
    export_uv=True,
    export_materials=True,
    export_normals=True,
    export_colors=True,
    export_triangulated_mesh=True,
    export_object_groups=False,
    export_material_groups=False,
    path_mode='RELATIVE',
    apply_modifiers=True,
)

# Print summary
print(f"\n--- Export complete ---")
print(f"OBJ file: {dst_obj}")
print(f"MTL file: {dst_obj.with_suffix('.mtl')}")
print(f"Textures exported: {len(texture_files)}")

print(f"\nDone!")