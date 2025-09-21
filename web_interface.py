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
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh, export_segment, select_seeds

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store current mesh and processing data
current_mesh = None
current_sparse_matrix = None
current_face_centers = None
current_mesh_path = None
current_curvature_penalty = 100.0
current_valley_faces = None
current_valley_scores = None

@app.route('/')
def index():
    """Serve the main interface page."""
    return render_template('index.html')

@app.route('/upload_mesh', methods=['POST'])
def upload_mesh():
    """Upload and load a mesh file."""
    global current_mesh, current_sparse_matrix, current_face_centers, current_mesh_path, current_curvature_penalty
    
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
        
        # Store settings
        current_curvature_penalty = curvature_penalty_strength
        
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
            
            # Build adjacency graph - updated to match segment_mesh.py signature
            current_sparse_matrix, current_face_centers = build_adjacency_graph(
                current_mesh, curvature_penalty_strength, user_seeds=None
            )
            
            # Get valley faces for visualization
            from segment_mesh import get_valley_faces
            valley_face_mask, valley_scores = get_valley_faces(current_mesh, angle_threshold_deg=20.0)
            current_valley_faces = valley_face_mask
            current_valley_scores = valley_scores
            
            # Prepare mesh data for Three.js
            vertices = current_mesh.vertices.tolist()
            faces = current_mesh.faces.tolist()
            face_centers = current_mesh.triangles_center.tolist()
            
            # Convert valley data to lists for JSON
            valley_faces_list = current_valley_faces.tolist() if current_valley_faces is not None else []
            valley_scores_list = current_valley_scores.tolist() if current_valley_scores is not None else []
            
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
                'filename': filename,
                'valley_faces': valley_faces_list,
                'valley_scores': valley_scores_list
            })
            
        except Exception as mesh_error:
            print(f"Error processing mesh: {mesh_error}")
            return jsonify({'success': False, 'error': f'Error processing mesh: {str(mesh_error)}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_mesh', methods=['POST'])
def load_mesh():
    """Load a mesh file and prepare it for visualization."""
    global current_mesh, current_sparse_matrix, current_face_centers, current_mesh_path, current_curvature_penalty
    
    try:
        data = request.json
        mesh_path = data.get('mesh_path', 'input/run/example.obj')
        curvature_penalty_strength = data.get('curvature_penalty_strength', 100.0)
        
        # Store settings
        current_curvature_penalty = curvature_penalty_strength
        
        # Convert to absolute path
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(os.getcwd(), mesh_path)
        
        current_mesh_path = mesh_path
        
        # Load and process mesh
        current_mesh = load_and_clean_mesh(mesh_path)
        
        # Build adjacency graph - updated to match segment_mesh.py signature
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, curvature_penalty_strength, user_seeds=None
        )
        
        # Get valley faces for visualization
        from segment_mesh import get_valley_faces
        valley_face_mask, valley_scores = get_valley_faces(current_mesh, angle_threshold_deg=20.0)
        current_valley_faces = valley_face_mask
        current_valley_scores = valley_scores
        
        # Prepare mesh data for Three.js
        vertices = current_mesh.vertices.tolist()
        faces = current_mesh.faces.tolist()
        face_centers = current_mesh.triangles_center.tolist()
        
        # Convert valley data to lists for JSON
        valley_faces_list = current_valley_faces.tolist() if current_valley_faces is not None else []
        valley_scores_list = current_valley_scores.tolist() if current_valley_scores is not None else []
                
        return jsonify({
            'success': True,
            'vertices': vertices,
            'faces': faces,
            'face_centers': face_centers,
            'total_faces': len(current_mesh.faces),
            'valley_faces': valley_faces_list,  # 
            'valley_scores': valley_scores_list #
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/segment_with_colored_seeds', methods=['POST'])
def segment_with_colored_seeds():
    """Perform segmentation using manually selected seed faces with colors, then combine segments by color."""
    global current_mesh, current_sparse_matrix, current_face_centers
    
    try:
        data = request.json
        colored_seeds = data.get('colored_seeds', [])
        output_dir = data.get('output_dir', 'output')
        
        if not colored_seeds:
            return jsonify({'success': False, 'error': 'No seed points selected'})
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
        
        # Convert clicked 3D points to face indices with color information
        seed_face_data = []  # List of {face_idx, color}
        face_centers = current_mesh.triangles_center
        
        # Calculate the same transformation that was applied in the frontend
        vertices = current_mesh.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)
        
        display_scale = 4.0 / max_dim if max_dim > 0 else 1.0
        display_center = -center
        
        print(f"Processing {len(colored_seeds)} colored seeds...")
        
        for seed_data in colored_seeds:
            point = seed_data['position']
            color = seed_data['color']
            
            # Transform clicked point back to original coordinate system
            original_point = (np.array(point) - display_center) / display_scale
            
            # Find the closest face center to the original point
            distances = np.linalg.norm(face_centers - original_point, axis=1)
            closest_face_idx = np.argmin(distances)
            
            seed_face_data.append({
                'face_idx': closest_face_idx,
                'color': color
            })
            
            print(f"Colored seed: {color} at point {point} -> face {closest_face_idx}")
        
        if not seed_face_data:
            return jsonify({'success': False, 'error': 'No valid seed faces found'})
        
        # Extract face indices for segmentation
        seed_face_indices = [seed['face_idx'] for seed in seed_face_data]
        
        # Rebuild graph with user seed information
        print("Rebuilding graph with colored user seeds...")
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=seed_face_indices
        )
        
        # Create seed indices array for the matrix
        seed_idx = np.array(seed_face_indices)
        
        # Perform segmentation
        face_labels = segment_mesh(current_sparse_matrix, seed_idx)
        
        # Now we need to combine segments that have the same color
        print("Combining segments by color...")
        
        # Create color mapping for seeds
        seed_color_map = {}  # face_idx -> color
        for seed_data in seed_face_data:
            seed_color_map[seed_data['face_idx']] = seed_data['color']
        
        # Group segments by color
        color_groups = {}  # color -> list of segment faces
        for face_idx, seed_face in face_labels.items():
            seed_color = seed_color_map.get(seed_face, 'unknown')
            if seed_color not in color_groups:
                color_groups[seed_color] = []
            color_groups[seed_color].append(face_idx)
        
        print(f"Created {len(color_groups)} color groups from {len(seed_idx)} original segments")
        
        # Export combined segments by color
        export_combined_segments_by_color(current_mesh, color_groups, output_dir)
        
        # Create face colors for visualization - use the same color for all faces that belong to the same color group
        color_palette = {
            'red': [1.0, 0.0, 0.0],
            'green': [0.0, 1.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'yellow': [1.0, 1.0, 0.0],
            'magenta': [1.0, 0.0, 1.0],
            'cyan': [0.0, 1.0, 1.0],
            'orange': [1.0, 0.5, 0.0],
            'purple': [0.5, 0.0, 1.0]
        }
        
        # Initialize all faces with default color
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.7, 0.7, 0.7], dtype=np.float32)
        
        # Color faces by their final color group
        for color, face_indices in color_groups.items():
            if color in color_palette:
                color_rgb = color_palette[color]
                for face_idx in face_indices:
                    face_colors[face_idx] = color_rgb
        
        # Convert to list format for JSON serialization
        face_colors_list = face_colors.tolist()
        
        print(f"Colored segmentation completed: {len(seed_idx)} initial segments combined into {len(color_groups)} color groups")
        
        return jsonify({
            'success': True,
            'segments_created': len(seed_idx),
            'combined_segments': len(color_groups),
            'face_colors': face_colors_list,
            'color_groups': list(color_groups.keys()),
            'output_dir': output_dir,
            'stats': {
                'total_faces': total_faces,
                'segmented_faces': len(face_labels),
                'coverage_ratio': len(face_labels) / total_faces if total_faces > 0 else 0.0,
                'color_groups': len(color_groups)
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in colored segmentation: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
    

@app.route('/segment_automatic', methods=['POST'])
def segment_automatic():
    """Perform automatic segmentation using the algorithm's seed selection."""
    global current_mesh, current_sparse_matrix, current_face_centers

    try:
        data = request.json or {}
        n_seeds = int(data.get('n_seeds', 10))
        output_dir = data.get('output_dir', 'output')

        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})

        print(f"Running automatic segmentation with {n_seeds} seeds...")

        # Build adjacency graph for automatic selection
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=None
        )

        # Automatic seed selection (should return face indices)
        seed_idx = select_seeds(current_face_centers, n_seeds)
        seed_idx = np.asarray(seed_idx, dtype=int)
        print(f"Automatically selected seed indices: {seed_idx.tolist()}")

        # Perform segmentation
        face_labels = segment_mesh(current_sparse_matrix, seed_idx)

        # Export segments
        export_segment(current_mesh, face_labels, seed_idx, output_dir)

        # Color palette for visualization
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8]
        ])

        # Initialize face colors (default)
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.7, 0.7, 0.7], dtype=np.float32)

        # Map seed face -> segment index
        seed_to_segment = {int(seed): i for i, seed in enumerate(seed_idx.tolist())}

        # Color segments according to the selected seed mapping
        if face_labels:
            for face_idx, seed_face in face_labels.items():
                if seed_face in seed_to_segment:
                    segment_id = seed_to_segment[seed_face]
                    color_idx = segment_id % len(colors)
                    face_colors[int(face_idx)] = colors[color_idx]

        # JSON-serializable lists
        face_colors_list = face_colors.tolist()
        seed_face_indices = [int(s) for s in seed_idx.tolist()]

        # seed positions: centroids of the selected seed faces (useful for frontend markers)
        try:
            seed_positions = current_mesh.triangles_center[seed_idx].tolist()
        except Exception:
            # fallback: empty list if something unexpected happens
            seed_positions = []

        # seed colors aligned with seed order
        seed_colors = [colors[i % len(colors)].tolist() for i in range(len(seed_face_indices))]

        print(f"Automatic segmentation completed with {len(seed_face_indices)} segments")

        return jsonify({
            'success': True,
            'segments_created': len(seed_face_indices),
            'face_colors': face_colors_list,
            'seed_face_indices': seed_face_indices,
            'seed_positions': seed_positions,
            'seed_colors': seed_colors,
            'output_dir': output_dir,
            'stats': {
                'total_faces': total_faces,
                'segmented_faces': len(face_labels),
                'coverage_ratio': len(face_labels) / total_faces if total_faces > 0 else 0.0
            }
        })

    except Exception as e:
        import traceback
        print(f"Error in automatic segmentation: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


def export_combined_segments_by_color(mesh, color_groups, output_dir):
    """Export segments combined by color."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export combined segments by color
    for color, face_indices in color_groups.items():
        if face_indices:  # Only export if there are faces
            sub = mesh.submesh([np.asarray(face_indices, dtype=np.int64)], append=True)
            filename = f"segment_{color}_combined.obj"
            sub.export(os.path.join(output_dir, filename))
            print(f"Exported {len(face_indices)} faces for color '{color}' to {filename}")


@app.route('/segment_with_seeds', methods=['POST'])
def segment_with_seeds():
    """Perform segmentation using manually selected seed faces."""
    global current_mesh, current_sparse_matrix, current_face_centers
    
    try:
        data = request.json
        clicked_points = data.get('clicked_points', [])
        output_dir = data.get('output_dir', 'output')
        
        if not clicked_points:
            return jsonify({'success': False, 'error': 'No seed points selected'})
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
        
        # Convert clicked 3D points to face indices
        # NOTE: Clicked points are in the transformed display coordinate system
        # We need to transform them back to the original mesh coordinate system
        seed_face_indices = []
        face_centers = current_mesh.triangles_center
        
        # Calculate the same transformation that was applied in the frontend
        # 1. Get mesh bounding box
        vertices = current_mesh.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)
        
        # 2. Calculate the display transformation parameters
        # In frontend: scale = 4 / maxDim, position = -center
        display_scale = 4.0 / max_dim if max_dim > 0 else 1.0
        display_center = -center
        
        print(f"Mesh transformation: center={center}, max_dim={max_dim}, display_scale={display_scale}")
        
        for point in clicked_points:
            # Transform clicked point back to original coordinate system
            # Reverse the display transformation: point_original = (point_display - display_center) / display_scale
            original_point = (np.array(point) - display_center) / display_scale
            
            # Find the closest face center to the original point
            distances = np.linalg.norm(face_centers - original_point, axis=1)
            closest_face_idx = np.argmin(distances)
            seed_face_indices.append(closest_face_idx)
            
            print(f"Clicked point {point} -> original {original_point} -> face {closest_face_idx}")
        
        if not seed_face_indices:
            return jsonify({'success': False, 'error': 'No valid seed faces found'})
        
        # Always use enhanced mode with user seeds to improve segmentation quality
        print("Rebuilding graph with user seed information for enhanced segmentation...")
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=seed_face_indices
        )
        print(f"Enhanced graph rebuilt with {len(seed_face_indices)} user seeds")
        
        # Create seed indices array for the matrix (face indices are already in correct format)
        seed_idx = np.array(seed_face_indices)
        
        # Perform segmentation
        face_labels = segment_mesh(current_sparse_matrix, seed_idx)
        
        # Export segments using the corrected function name
        export_segment(current_mesh, face_labels, seed_idx, output_dir)
        
        # Create segment colors for visualization
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2]
        ])
        
        # Initialize all faces with default color
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.7, 0.7, 0.7], dtype=np.float32)
        
        # Create mapping of seed indices to their colors for the frontend
        seed_colors = []
        seed_to_segment = {seed: i for i, seed in enumerate(seed_idx)}
        
        for i, seed_face in enumerate(seed_idx):
            color_idx = i % len(colors)
            seed_colors.append(colors[color_idx].tolist())
        
        # Color segments
        if face_labels:
            for face_idx, seed_face in face_labels.items():
                if seed_face in seed_to_segment:
                    segment_id = seed_to_segment[seed_face]
                    color_idx = segment_id % len(colors)
                    face_colors[face_idx] = colors[color_idx]
        
        # Convert to list format for JSON serialization
        face_colors_list = face_colors.tolist()
        
        print(f"Segmentation completed with {len(seed_idx)} segments")
        
        return jsonify({
            'success': True,
            'segments_created': len(seed_idx),
            'face_colors': face_colors_list,
            'seed_colors': seed_colors,
            'clicked_points': clicked_points,
            'output_dir': output_dir,
            'stats': {
                'total_faces': total_faces,
                'segmented_faces': len(face_labels),
                'coverage_ratio': len(face_labels) / total_faces if total_faces > 0 else 0.0
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in segmentation: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_segments')
def download_segments():
    """Create and download a zip file containing all segment files."""
    try:
        output_dir = request.args.get('output_dir', 'output')
        
        if not os.path.exists(output_dir):
            return jsonify({'success': False, 'error': 'Output directory does not exist'})
        
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
