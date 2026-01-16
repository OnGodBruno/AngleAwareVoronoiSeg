"""
Multi-mesh calibration and batch segmentation web interface.
Allows calibration of seed points for multiple meshes and batch processing.
"""

import os
import json
import numpy as np
import trimesh
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import zipfile
from werkzeug.utils import secure_filename
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh, export_segment
import glob
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['MESHES_FOLDER'] = 'meshes'
app.config['CONFIG_FOLDER'] = 'config'
CORS(app)

# Create directories if they don't exist
os.makedirs(app.config['MESHES_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONFIG_FOLDER'], exist_ok=True)

# Global storage for mesh data
mesh_data_cache = {}
current_segmentation_results = {}

def get_mesh_files():
    """Get all GLB files from the meshes folder."""
    mesh_files = []
    print(f"DEBUG: Looking in folder: {app.config['MESHES_FOLDER']}")
    print(f"DEBUG: Folder exists: {os.path.exists(app.config['MESHES_FOLDER'])}")
    
    for ext in ['*.glb', '*.obj', '*.gltf']:
        pattern = os.path.join(app.config['MESHES_FOLDER'], ext)
        found = glob.glob(pattern)
        print(f"DEBUG: Pattern {pattern} found: {found}")
        mesh_files.extend(found)
    return sorted(mesh_files)

def load_mesh_data(mesh_path):
    """Load mesh data and cache it."""
    if mesh_path not in mesh_data_cache:
        try:
            print(f"DEBUG: Loading mesh from {mesh_path}")
            
            # Load the file with trimesh
            loaded = trimesh.load(mesh_path, process=False)
            
            # Handle different types of loaded objects
            if hasattr(loaded, 'geometry') and loaded.geometry:
                # It's a scene, extract and combine all geometries
                print("DEBUG: Processing scene with multiple geometries...")
                geometries = list(loaded.geometry.values())
                
                if not geometries:
                    print("ERROR: No geometry found in file")
                    return None
                
                if len(geometries) == 1:
                    mesh = geometries[0]
                else:
                    # Combine multiple geometries into one mesh
                    print(f"DEBUG: Combining {len(geometries)} geometries...")
                    combined_vertices = []
                    combined_faces = []
                    vertex_offset = 0
                    
                    for geom in geometries:
                        if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                            combined_vertices.append(geom.vertices)
                            # Offset face indices for the combined mesh
                            offset_faces = geom.faces + vertex_offset
                            combined_faces.append(offset_faces)
                            vertex_offset += len(geom.vertices)
                    
                    if combined_vertices:
                        # Create combined mesh
                        mesh = trimesh.Trimesh(
                            vertices=np.vstack(combined_vertices),
                            faces=np.vstack(combined_faces)
                        )
                    else:
                        print("ERROR: No valid geometries found")
                        return None
            else:
                # It's already a mesh
                mesh = loaded
            
            # Clean the mesh
            print("DEBUG: Cleaning mesh...")
            mesh.remove_unreferenced_vertices()
            mesh.remove_infinite_values()
            
            print("DEBUG: Building adjacency graph...")
            sparse_matrix, face_centers = build_adjacency_graph(mesh, 100.0)
            
            mesh_data_cache[mesh_path] = {
                'mesh': mesh,
                'sparse_matrix': sparse_matrix,
                'face_centers': face_centers,
                'filename': os.path.basename(mesh_path)
            }
            print(f"DEBUG: Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    return mesh_data_cache[mesh_path]

def get_config_path(mesh_filename):
    """Get the config file path for a mesh."""
    base_name = os.path.splitext(mesh_filename)[0]
    return os.path.join(app.config['CONFIG_FOLDER'], f"{base_name}_config.json")

def save_mesh_config(mesh_filename, seed_points):
    """Save seed point configuration for a mesh."""
    config_path = get_config_path(mesh_filename)
    config = {
        'mesh_filename': mesh_filename,
        'seed_points': seed_points,
        'timestamp': np.datetime64('now').astype(str)
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_mesh_config(mesh_filename):
    """Load seed point configuration for a mesh."""
    config_path = get_config_path(mesh_filename)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    """Serve the main interface page."""
    return render_template('multi_mesh.html')

@app.route('/calibration')
def calibration():
    """Serve the calibration interface page."""
    return render_template('calibration.html')

@app.route('/viewer')
def viewer():
    """Serve the multi-mesh viewer page."""
    return render_template('viewer.html')

@app.route('/api/meshes', methods=['GET'])
def get_meshes():
    """Get list of all available meshes."""
    try:
        mesh_files = get_mesh_files()
        print(f"DEBUG: Found {len(mesh_files)} mesh files: {mesh_files}")
        meshes = []
        
        for mesh_path in mesh_files:
            filename = os.path.basename(mesh_path)
            config = load_mesh_config(filename)
            print(f"DEBUG: Processing {filename}, config exists: {config is not None}")
            
            meshes.append({
                'filename': filename,
                'path': mesh_path,
                'configured': config is not None,
                'seed_count': len(config['seed_points']) if config else 0
            })
        
        print(f"DEBUG: Returning {len(meshes)} meshes")
        return jsonify({
            'success': True,
            'meshes': meshes
        })
    except Exception as e:
        print(f"DEBUG: Error in get_meshes: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/load_mesh/<filename>')
def load_mesh(filename):
    """Load a specific mesh for calibration."""
    try:
        mesh_path = os.path.join(app.config['MESHES_FOLDER'], filename)
        if not os.path.exists(mesh_path):
            return jsonify({'success': False, 'error': 'Mesh file not found'})
        
        print(f"DEBUG: Loading mesh from {mesh_path}")
        
        # Load the file with trimesh
        loaded = trimesh.load(mesh_path, process=False)
        
        # Handle different types of loaded objects
        if hasattr(loaded, 'geometry') and loaded.geometry:
            # It's a scene, extract and combine all geometries
            print("DEBUG: Processing scene with multiple geometries...")
            geometries = list(loaded.geometry.values())
            
            if not geometries:
                print("ERROR: No geometry found in file")
                return jsonify({'success': False, 'error': 'No geometry found in GLB/GLTF file'})
            
            if len(geometries) == 1:
                mesh = geometries[0]
            else:
                # Combine multiple geometries into one mesh
                print(f"DEBUG: Combining {len(geometries)} geometries...")
                combined_vertices = []
                combined_faces = []
                vertex_offset = 0
                
                for geom in geometries:
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        combined_vertices.append(geom.vertices)
                        # Offset face indices for the combined mesh
                        offset_faces = geom.faces + vertex_offset
                        combined_faces.append(offset_faces)
                        vertex_offset += len(geom.vertices)
                
                if combined_vertices:
                    # Create combined mesh
                    mesh = trimesh.Trimesh(
                        vertices=np.vstack(combined_vertices),
                        faces=np.vstack(combined_faces)
                    )
                else:
                    print("ERROR: No valid geometries found")
                    return jsonify({'success': False, 'error': 'No valid geometries found'})
        else:
            # It's already a mesh
            mesh = loaded
        
        # Clean the mesh
        print("DEBUG: Cleaning mesh...")
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return jsonify({'success': False, 'error': 'Mesh has no vertices or faces'})

        # Build adjacency graph - updated to match segment_mesh.py signature
        sparse_matrix, face_centers = build_adjacency_graph(mesh, 100.0)
        
        # Prepare mesh data for Three.js
        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        face_centers = mesh.triangles_center.tolist()
        
        print(f"Converted to lists: vertices={len(vertices)}, faces={len(faces)}, centers={len(face_centers)}")
        
        # Validate the data before sending
        if not vertices or not faces:
            return jsonify({'success': False, 'error': 'Failed to convert mesh data to lists'})
        
        # Load existing configuration if available
        config = load_mesh_config(filename)
        
        return jsonify({
            'success': True,
            'mesh_data': {
                'vertices': vertices,
                'faces': faces,
                'face_centers': face_centers,
                'total_faces': len(mesh.faces),
                'filename': filename
            },
            'config': config
        })
        
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_config', methods=['POST'])
def save_config():
    """Save seed point configuration for a mesh."""
    try:
        data = request.get_json()
        filename = data['filename']
        seed_points = data['seed_points']
        
        save_mesh_config(filename, seed_points)
        
        return jsonify({
            'success': True,
            'message': f'Configuration saved for {filename}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/segment_all', methods=['POST'])
def segment_all():
    """Segment all configured meshes."""
    try:
        mesh_files = get_mesh_files()
        results = {}
        
        for mesh_path in mesh_files:
            filename = os.path.basename(mesh_path)
            config = load_mesh_config(filename)
            
            if not config:
                continue  # Skip unconfigured meshes
                
            mesh_data = load_mesh_data(mesh_path)
            if not mesh_data:
                continue
                
            mesh = mesh_data['mesh']
            sparse_matrix = mesh_data['sparse_matrix']
            face_centers = mesh_data['face_centers']
            
            # Convert seed points to face indices
            seed_faces = [seed['face_index'] if isinstance(seed, dict) else seed for seed in config['seed_points']]
            
            # Run segmentation
            segmentation_result = segment_mesh(
                sparse_matrix, 
                seed_faces
            )
            
            # Convert result to JSON-serializable format
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()
            face_labels = segmentation_result.tolist() if hasattr(segmentation_result, 'tolist') else segmentation_result
            
            results[filename] = {
                'vertices': vertices,
                'faces': faces,
                'face_labels': face_labels,
                'n_segments': len(seed_faces),
                'seed_colors': [seed.get('color', 'red') if isinstance(seed, dict) else 'red' for seed in config['seed_points']],
                'seed_faces': seed_faces
            }
        
        current_segmentation_results.update(results)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/export_segments', methods=['POST'])
def export_segments():
    """Export segmented meshes as individual files."""
    try:
        data = request.get_json()
        filename = data['filename']
        
        if filename not in current_segmentation_results:
            return jsonify({'success': False, 'error': 'No segmentation data found'})
        
        result = current_segmentation_results[filename]
        mesh_path = os.path.join(app.config['MESHES_FOLDER'], filename)
        mesh_data = load_mesh_data(mesh_path)
        mesh = mesh_data['mesh']
        
        # Create temporary directory for exports
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}_segments.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                face_labels = np.array(result['face_labels'])
                unique_labels = np.unique(face_labels)
                
                for label in unique_labels:
                    if label >= 0:  # Skip background/unlabeled faces
                        segment_faces = np.where(face_labels == label)[0]
                        segment_mesh = export_segment(mesh, segment_faces)
                        
                        segment_filename = f"segment_{label}.obj"
                        segment_path = os.path.join(temp_dir, segment_filename)
                        segment_mesh.export(segment_path)
                        zipf.write(segment_path, segment_filename)
            
            return send_file(zip_path, as_attachment=True, download_name=f"{os.path.splitext(filename)[0]}_segments.zip")
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)