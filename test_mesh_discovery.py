import os
import glob

# Test the mesh discovery
MESHES_FOLDER = 'meshes'

def get_mesh_files():
    """Get all GLB files from the meshes folder."""
    mesh_files = []
    for ext in ['*.glb', '*.obj', '*.gltf']:
        pattern = os.path.join(MESHES_FOLDER, ext)
        print(f"Looking for pattern: {pattern}")
        found_files = glob.glob(pattern)
        print(f"Found files for {ext}: {found_files}")
        mesh_files.extend(found_files)
    return sorted(mesh_files)

# Check current directory
print("Current directory:", os.getcwd())
print("Meshes folder exists:", os.path.exists(MESHES_FOLDER))

if os.path.exists(MESHES_FOLDER):
    print("Files in meshes folder:", os.listdir(MESHES_FOLDER))

# Test the function
mesh_files = get_mesh_files()
print("Found mesh files:", mesh_files)