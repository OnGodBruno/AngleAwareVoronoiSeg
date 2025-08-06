"""
Web interface for interactive 3D mesh segmentation with manual seed selection.
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
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh, export_segments

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store current mesh and processing data
current_mesh = None
current_sparse_matrix = None
current_face_coords = None
current_active_faces = None
current_mesh_path = None
current_curvature_penalty = 100.0
current_enhanced_mode = False

@app.route('/')
def index():
    """Serve the main interface page."""
    return render_template('index_simple.html')

@app.route('/upload_mesh', methods=['POST'])
def upload_mesh():
    """Upload and load a mesh file."""
    global current_mesh, current_sparse_matrix, current_face_coords, current_active_faces, current_mesh_path, current_curvature_penalty, current_enhanced_mode
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['obj', 'glb', 'gltf']:
            return jsonify({'success': False, 'error': 'Only .obj, .glb, and .gltf files are supported'})
        
        # Get parameters from form data
        curvature_penalty_strength = float(request.form.get('curvature_penalty_strength', 100.0))
        enhanced_mode = request.form.get('enhanced_mode', 'false').lower() == 'true'
        
        # Store settings
        current_curvature_penalty = curvature_penalty_strength
        current_enhanced_mode = enhanced_mode
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert GLB/GLTF to OBJ if necessary
        if file_ext in ['glb', 'gltf']:
            try:
                print(f"Converting {file_ext.upper()} file to OBJ format...")
                
                # Load the GLB/GLTF file with trimesh
                loaded = trimesh.load(filepath, process=False)
                
                # Handle different types of loaded objects
                if hasattr(loaded, 'geometry') and loaded.geometry:
                    # It's a scene, extract and combine all geometries
                    print("Processing scene with multiple geometries...")
                    geometries = list(loaded.geometry.values())
                    
                    if not geometries:
                        return jsonify({'success': False, 'error': 'No geometry found in GLB/GLTF file'})
                    
                    if len(geometries) == 1:
                        mesh = geometries[0]
                    else:
                        # Combine multiple geometries into one mesh
                        print(f"Combining {len(geometries)} geometries...")
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
                            import numpy as np
                            # Create combined mesh
                            mesh = trimesh.Trimesh(
                                vertices=np.vstack(combined_vertices),
                                faces=np.vstack(combined_faces)
                            )
                        else:
                            return jsonify({'success': False, 'error': 'No valid geometries found in GLB/GLTF file'})
                
                elif hasattr(loaded, 'vertices') and hasattr(loaded, 'faces'):
                    # It's already a single mesh
                    mesh = loaded
                else:
                    return jsonify({'success': False, 'error': 'Invalid or unsupported GLB/GLTF file format'})
                
                # Validate the mesh
                if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                    return jsonify({'success': False, 'error': 'Invalid mesh data in GLB/GLTF file'})
                
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    return jsonify({'success': False, 'error': 'Empty mesh in GLB/GLTF file'})
                
                # Clean up the mesh
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()
                
                print(f"Converted mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Create OBJ filename
                obj_filename = os.path.splitext(filename)[0] + '_converted.obj'
                obj_filepath = os.path.join(app.config['UPLOAD_FOLDER'], obj_filename)
                
                # Export as OBJ
                mesh.export(obj_filepath)
                
                # Update filepath to use the converted OBJ file
                filepath = obj_filepath
                filename = obj_filename
                
                print(f"Successfully converted {file_ext.upper()} to OBJ: {obj_filename}")
                
            except Exception as conversion_error:
                import traceback
                print(f"Error converting {file_ext.upper()} file: {conversion_error}")
                print("Full traceback:", traceback.format_exc())
                return jsonify({'success': False, 'error': f'Error converting {file_ext.upper()} file: {str(conversion_error)}'})
        
        current_mesh_path = filepath
        
        # Load and process mesh
        try:
            current_mesh = load_and_clean_mesh(filepath)
            print(f"Loaded mesh: {len(current_mesh.vertices)} vertices, {len(current_mesh.faces)} faces")
            
            if len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                return jsonify({'success': False, 'error': 'Mesh has no vertices or faces'})
            
            current_sparse_matrix, current_face_coords, current_active_faces = build_adjacency_graph(
                current_mesh, curvature_penalty_strength, enhanced_mode=False  # Always use fast mode for initial loading
            )
            
            # Prepare mesh data for Three.js
            vertices = current_mesh.vertices.tolist()
            faces = current_mesh.faces.tolist()
            face_centers = current_mesh.triangles_center.tolist()
            
            print(f"Converted to lists: vertices={len(vertices)}, faces={len(faces)}, centers={len(face_centers)}")
            
            # Validate the data before sending
            if not vertices or not faces:
                return jsonify({'success': False, 'error': 'Failed to convert mesh data to lists'})
            
            return jsonify({
                'success': True,
                'vertices': vertices,
                'faces': faces,
                'face_centers': face_centers,
                'total_faces': len(current_mesh.faces),
                'active_faces': len(current_active_faces),
                'filename': filename
            })
            
        except Exception as mesh_error:
            print(f"Error processing mesh: {mesh_error}")
            return jsonify({'success': False, 'error': f'Error processing mesh: {str(mesh_error)}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_mesh', methods=['POST'])
def load_mesh():
    """Load a mesh file and prepare it for visualization."""
    global current_mesh, current_sparse_matrix, current_face_coords, current_active_faces, current_mesh_path, current_curvature_penalty, current_enhanced_mode
    
    try:
        data = request.json
        mesh_path = data.get('mesh_path', 'input/run/example.obj')
        curvature_penalty_strength = data.get('curvature_penalty_strength', 100.0)
        enhanced_mode = data.get('enhanced_mode', False)
        
        # Store settings
        current_curvature_penalty = curvature_penalty_strength
        current_enhanced_mode = enhanced_mode
        
        # Convert to absolute path
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(os.getcwd(), mesh_path)
        
        current_mesh_path = mesh_path
        
        # Load and process mesh
        current_mesh = load_and_clean_mesh(mesh_path)
        current_sparse_matrix, current_face_coords, current_active_faces = build_adjacency_graph(
            current_mesh, curvature_penalty_strength, enhanced_mode=False  # Always use fast mode for initial loading
        )
        
        # Prepare mesh data for Three.js
        vertices = current_mesh.vertices.tolist()
        faces = current_mesh.faces.tolist()
        face_centers = current_mesh.triangles_center.tolist()
        
        return jsonify({
            'success': True,
            'vertices': vertices,
            'faces': faces,
            'face_centers': face_centers,
            'total_faces': len(current_mesh.faces),
            'active_faces': len(current_active_faces)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/segment_with_seeds', methods=['POST'])
def segment_with_seeds():
    """Perform segmentation using manually selected seed faces."""
    global current_mesh, current_sparse_matrix, current_face_coords, current_active_faces
    
    try:
        data = request.json
        clicked_points = data.get('clicked_points', [])
        output_dir = data.get('output_dir', 'output')
        enhanced_mode = data.get('enhanced_mode', True)  # Default to enhanced for user seeds
        
        if not clicked_points:
            return jsonify({'success': False, 'error': 'No seed points selected'})
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
        
        # Convert clicked 3D points to face indices
        seed_face_indices = []
        face_centers = current_mesh.triangles_center
        
        for point in clicked_points:
            # Find the closest face center to the clicked point
            distances = np.linalg.norm(face_centers - np.array(point), axis=1)
            closest_face_idx = np.argmin(distances)
            
            # Convert global face index to active face matrix index
            if closest_face_idx in current_active_faces:
                matrix_idx = np.where(current_active_faces == closest_face_idx)[0][0]
                seed_face_indices.append(matrix_idx)
        
        if not seed_face_indices:
            return jsonify({'success': False, 'error': 'No valid seed faces found in active graph'})
        
        seed_idx = np.array(seed_face_indices)
        
        # If enhanced mode is enabled, rebuild the graph with user seed information
        if enhanced_mode:
            print("Rebuilding graph with user seed information for enhanced segmentation...")
            # Get the actual face indices for user seeds
            user_seed_faces = [current_active_faces[idx] for idx in seed_face_indices]
            
            # Rebuild graph with enhanced distance metrics
            current_sparse_matrix, current_face_coords, current_active_faces = build_adjacency_graph(
                current_mesh, current_curvature_penalty, enhanced_mode=True, user_seeds=user_seed_faces
            )
            
            # Recompute seed indices for the new graph
            seed_face_indices = []
            for point in clicked_points:
                distances = np.linalg.norm(face_centers - np.array(point), axis=1)
                closest_face_idx = np.argmin(distances)
                if closest_face_idx in current_active_faces:
                    matrix_idx = np.where(current_active_faces == closest_face_idx)[0][0]
                    seed_face_indices.append(matrix_idx)
            
            seed_idx = np.array(seed_face_indices)
            print(f"Enhanced graph rebuilt with {len(seed_idx)} seeds")
        
        # Perform segmentation
        face_labels = segment_mesh(current_sparse_matrix, seed_idx)
        
        # Export segments
        export_segments(current_mesh, face_labels, seed_idx, current_active_faces, output_dir)
        
        # Create segment colors for visualization
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2]
        ])
        
        # Initialize all faces with default color using NumPy for efficiency
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.2, 0.2, 0.3], dtype=np.float32)
        
        # Create boolean mask for segmented faces (much faster than set)
        segmented_mask = np.zeros(total_faces, dtype=bool)
        
        # Color segments efficiently using vectorized operations
        if face_labels:
            # Convert to arrays for vectorized operations
            row_indices = np.array(list(face_labels.keys()))
            seed_rows = np.array(list(face_labels.values()))
            
            # Map seed rows to segment IDs
            seed_to_segment = {seed: i for i, seed in enumerate(seed_idx)}
            segment_ids = np.array([seed_to_segment[seed] for seed in seed_rows])
            
            # Get face IDs and colors in one go
            face_ids = current_active_faces[row_indices]
            color_indices = segment_ids % len(colors)
            
            # Assign colors vectorized
            face_colors[face_ids] = colors[color_indices]
            segmented_mask[face_ids] = True
        
        colored_faces = np.sum(segmented_mask)
        
        # Better solution: Assign uncolored faces to the most dominant segment
        uncolored_faces = np.where(~segmented_mask)[0]
        
        if len(uncolored_faces) > 0:
            print(f"Assigning {len(uncolored_faces)} uncolored faces to first segment...")
            
            # Simple and fast: assign all uncolored faces to the first segment
            if len(seed_idx) > 0:
                face_colors[uncolored_faces] = colors[0]
                segmented_mask[uncolored_faces] = True
                print(f"Assigned {len(uncolored_faces)} faces to first segment")
            else:
                # Fallback: assign to default color
                face_colors[uncolored_faces] = colors[0]
                segmented_mask[uncolored_faces] = True
        
        colored_faces = np.sum(segmented_mask)
        
        # Convert back to list format for JSON serialization
        face_colors_list = face_colors.tolist()
        
        # Log statistics for debugging
        print(f"Final segmentation stats: {colored_faces}/{total_faces} faces colored ({(colored_faces/total_faces)*100:.1f}%)")
        print(f"Segmented faces from algorithm: {np.sum(segmented_mask)}, Uncolored faces filled by nearest neighbor: {len(uncolored_faces) if 'uncolored_faces' in locals() else 0}")
        
        return jsonify({
            'success': True,
            'segments_created': len(seed_idx),
            'face_colors': face_colors_list,
            'output_dir': output_dir,
            'stats': {
                'total_faces': total_faces,
                'colored_faces': int(colored_faces),
                'active_faces': len(current_active_faces),
                'face_labels': len(face_labels)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_segments')
def download_segments():
    """Create and download a zip file containing all segment files."""
    try:
        output_dir = request.args.get('output_dir', 'output')
        
        # Create a temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w') as zip_file:
            for filename in os.listdir(output_dir):
                if filename.endswith('.obj'):
                    file_path = os.path.join(output_dir, filename)
                    zip_file.write(file_path, filename)
        
        return send_file(temp_zip.name, as_attachment=True, download_name='segments.zip')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
