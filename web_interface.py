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

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store current mesh and processing data
current_mesh = None
current_sparse_matrix = None
current_face_centers = None
current_mesh_path = None
current_curvature_penalty = 100.0

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
            print(f"Loading mesh with curvature penalty strength: {curvature_penalty_strength}")
            current_sparse_matrix, current_face_centers, curvature_stats = build_adjacency_graph(
                current_mesh, curvature_penalty_strength, user_seeds=None, return_stats=True
            )
            print(f"Curvature penalty applied: {curvature_stats}")
            
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
                'filename': filename,
                'curvature_stats': curvature_stats
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
        print(f"Building adjacency graph with curvature penalty strength: {curvature_penalty_strength}")
        current_sparse_matrix, current_face_centers, curvature_stats = build_adjacency_graph(
            current_mesh, curvature_penalty_strength, user_seeds=None, return_stats=True
        )
        print(f"Curvature penalty applied: {curvature_stats}")
        
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
            'curvature_stats': curvature_stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/auto_place_seeds', methods=['POST'])
def auto_place_seeds():
    """Automatically place seeds using curvature analysis and distance optimization."""
    global current_mesh, current_sparse_matrix, current_face_centers, current_curvature_penalty
    
    try:
        data = request.json
        num_seeds = data.get('num_seeds', 3)
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
        
        if num_seeds < 1 | num_seeds > 20:
            return jsonify({'success': False, 'error': 'Number of seeds must be between 1 and 20'})
        
        print(f"Auto-placing {num_seeds} seeds using optimal algorithm...")
        
        # Use the optimal seed placement algorithm
        seed_positions = auto_place_optimal_seeds(
            current_mesh, 
            current_curvature_penalty, 
            num_seeds
        )
        
        # Convert seed positions to display coordinate system (same transformation as frontend)
        vertices = current_mesh.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)
        
        display_scale = 4.0 / max_dim if max_dim > 0 else 1.0
        display_center = -center
        
        # Transform positions to display coordinates
        display_positions = []
        for pos in seed_positions:
            display_pos = (np.array(pos) + display_center) * display_scale
            display_positions.append(display_pos.tolist())
        
        print(f"Successfully placed {len(display_positions)} optimal seeds")
        
        return jsonify({
            'success': True,
            'seed_positions': display_positions,
            'algorithm_info': f'Curvature-optimized placement with geodesic distance maximization',
            'num_seeds_placed': len(display_positions)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in auto seed placement: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


def auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds):
    """
    Automatically place seeds using improved algorithm:
    1. Find area with low curvature penalty for first seed
    2. Use combination of geodesic and Euclidean distance to ensure good spatial distribution
    3. Add minimum distance constraints to prevent clustering
    """
    print(f"Running improved optimal seed placement for {num_seeds} seeds...")
    
    # Get mesh properties
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    face_adjacency = mesh.face_adjacency
    
    # Calculate mesh scale for distance thresholds
    mesh_bounds = np.max(face_centers, axis=0) - np.min(face_centers, axis=0)
    mesh_scale = np.max(mesh_bounds)
    min_seed_distance = mesh_scale * 0.15  # Minimum 15% of mesh size between seeds
    
    print(f"Mesh scale: {mesh_scale:.3f}, minimum seed distance: {min_seed_distance:.3f}")
    
    # Calculate curvature penalties for all adjacent face pairs
    n1 = face_normals[face_adjacency[:, 0]]
    n2 = face_normals[face_adjacency[:, 1]]
    angles = np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))
    curvature_penalties = np.exp(curvature_penalty_strength * angles)
    
    # Calculate average curvature penalty per face
    face_curvature_scores = np.zeros(len(face_centers))
    face_neighbor_counts = np.zeros(len(face_centers))
    
    for i, (f1, f2) in enumerate(face_adjacency):
        penalty = curvature_penalties[i]
        face_curvature_scores[f1] += penalty
        face_curvature_scores[f2] += penalty
        face_neighbor_counts[f1] += 1
        face_neighbor_counts[f2] += 1
    
    # Average out the curvature scores
    valid_faces = face_neighbor_counts > 0
    face_curvature_scores[valid_faces] /= face_neighbor_counts[valid_faces]
    
    print(f"Calculated curvature scores. Min: {np.min(face_curvature_scores):.3f}, Max: {np.max(face_curvature_scores):.3f}")
    
    # Find faces with low curvature (smooth areas) for first seed
    low_curvature_threshold = np.percentile(face_curvature_scores[valid_faces], 25)  # Bottom 25%
    low_curvature_faces = np.where((face_curvature_scores <= low_curvature_threshold) & valid_faces)[0]
    
    if len(low_curvature_faces) == 0:
        low_curvature_faces = np.where(valid_faces)[0]
    
    # Pick first seed from low curvature area that's also near center of mesh
    mesh_center = np.mean(face_centers, axis=0)
    center_distances = np.linalg.norm(face_centers[low_curvature_faces] - mesh_center, axis=1)
    # Choose from faces that are reasonably central (within 60% of max distance from center)
    max_center_dist = np.max(center_distances)
    central_threshold = max_center_dist * 0.6
    central_low_curv_faces = low_curvature_faces[center_distances <= central_threshold]
    
    if len(central_low_curv_faces) == 0:
        central_low_curv_faces = low_curvature_faces
    
    np.random.seed(42)  # For reproducible results
    first_seed_face = np.random.choice(central_low_curv_faces)
    selected_seeds = [first_seed_face]
    
    print(f"First seed placed at face {first_seed_face} (curvature score: {face_curvature_scores[first_seed_face]:.3f}, distance from center: {np.linalg.norm(face_centers[first_seed_face] - mesh_center):.3f})")
    
    if num_seeds == 1:
        return [face_centers[first_seed_face].tolist()]
    
    # Build adjacency matrix for geodesic distances
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    
    edge_lengths = mesh.edges_unique_length
    avg_edge_length = np.mean(edge_lengths)
    
    # Calculate edge weights (spatial distance + curvature penalty)
    p1 = face_centers[face_adjacency[:, 0]]
    p2 = face_centers[face_adjacency[:, 1]]
    spatial_dist = np.linalg.norm(p2 - p1, axis=1)
    spatial_penalty = 1 + (spatial_dist / avg_edge_length) ** 2
    weights = spatial_penalty * curvature_penalties
    
    # Create symmetric sparse matrix
    n_faces = len(face_centers)
    row = np.concatenate([face_adjacency[:, 0], face_adjacency[:, 1]])
    col = np.concatenate([face_adjacency[:, 1], face_adjacency[:, 0]])
    data = np.concatenate([weights, weights])
    
    sparse_matrix = csr_matrix((data, (row, col)), shape=(n_faces, n_faces))
    
    print(f"Built adjacency matrix: {n_faces} faces, {len(weights)} edges")
    
    # Place remaining seeds iteratively with improved algorithm
    for seed_num in range(1, num_seeds):
        print(f"Placing seed {seed_num + 1}/{num_seeds}...")
        
        # Calculate Euclidean distances from all current seeds
        min_euclidean_dist = np.full(n_faces, np.inf)
        for seed_face in selected_seeds:
            euclidean_dists = np.linalg.norm(face_centers - face_centers[seed_face], axis=1)
            min_euclidean_dist = np.minimum(min_euclidean_dist, euclidean_dists)
        
        # Find faces that are far enough away (minimum distance constraint)
        far_enough_faces = min_euclidean_dist >= min_seed_distance
        
        # If no faces are far enough, reduce the threshold
        if not np.any(far_enough_faces):
            reduced_threshold = min_seed_distance * 0.5
            far_enough_faces = min_euclidean_dist >= reduced_threshold
            print(f"Reduced minimum distance to {reduced_threshold:.3f}")
        
        # Among faces that are far enough, prefer those with low curvature
        candidate_faces = np.where(far_enough_faces & valid_faces)[0]
        
        if len(candidate_faces) == 0:
            print("Warning: No valid candidates found, using all valid faces")
            candidate_faces = np.where(valid_faces)[0]
        
        # Calculate geodesic distances from current seeds to candidate faces
        current_seed_indices = np.array(selected_seeds)
        geodesic_distances = dijkstra(sparse_matrix, indices=current_seed_indices, directed=False)
        
        # For each candidate face, find minimum geodesic distance to any existing seed
        min_geodesic_dist = np.full(n_faces, np.inf)
        for i in range(len(current_seed_indices)):
            geodesic_dist_from_seed = geodesic_distances[i]
            min_geodesic_dist = np.minimum(min_geodesic_dist, geodesic_dist_from_seed)
        
        # Combine geodesic and Euclidean distances with curvature preference
        combined_scores = np.zeros(n_faces)
        for face_idx in candidate_faces:
            if np.isfinite(min_geodesic_dist[face_idx]):
                # Normalize geodesic distance
                geodesic_score = min_geodesic_dist[face_idx]
                
                # Normalize Euclidean distance
                euclidean_score = min_euclidean_dist[face_idx] / mesh_scale
                
                # Inverse curvature score (lower curvature = higher score)
                curvature_score = 1.0 / (1.0 + face_curvature_scores[face_idx])
                
                # Combined score: 60% geodesic, 30% euclidean, 10% curvature
                combined_scores[face_idx] = (0.6 * geodesic_score + 
                                           0.3 * euclidean_score * mesh_scale + 
                                           0.1 * curvature_score * mesh_scale)
            
        # Find the best candidate
        valid_scores = combined_scores[candidate_faces]
        if len(valid_scores) > 0 and np.max(valid_scores) > 0:
            best_local_idx = np.argmax(valid_scores)
            next_seed_face = candidate_faces[best_local_idx]
        else:
            # Fallback: use the face with maximum Euclidean distance
            next_seed_face = np.argmax(min_euclidean_dist)
            print(f"Fallback: using face with max Euclidean distance")
        
        selected_seeds.append(next_seed_face)
        
        euclidean_dist = min_euclidean_dist[next_seed_face]
        geodesic_dist = min_geodesic_dist[next_seed_face] if np.isfinite(min_geodesic_dist[next_seed_face]) else "inf"
        curvature_score = face_curvature_scores[next_seed_face]
        
        print(f"Seed {seed_num + 1} placed at face {next_seed_face}")
        print(f"  Euclidean distance: {euclidean_dist:.3f}")
        print(f"  Geodesic distance: {geodesic_dist}")
        print(f"  Curvature score: {curvature_score:.3f}")
        print(f"  Combined score: {combined_scores[next_seed_face]:.3f}")
    
    # Convert face indices to 3D positions
    seed_positions = []
    for seed_face in selected_seeds:
        position = face_centers[seed_face]
        seed_positions.append(position.tolist())
    
    # Verify final distances between seeds
    print("\nFinal seed separation analysis:")
    for i in range(len(selected_seeds)):
        for j in range(i+1, len(selected_seeds)):
            dist = np.linalg.norm(face_centers[selected_seeds[i]] - face_centers[selected_seeds[j]])
            print(f"  Seeds {i+1}-{j+1}: {dist:.3f}")
    
    print(f"Completed improved seed placement: {len(seed_positions)} seeds with better spatial distribution")
    return seed_positions


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
            current_mesh, current_curvature_penalty, user_seeds=seed_face_indices, mode='segmentation'
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
        data = request.json
        n_seeds = data.get('n_seeds', 10)
        output_dir = data.get('output_dir', 'output')
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
        
        print(f"Running automatic segmentation with {n_seeds} seeds...")
        
        # Build adjacency graph without user seeds for automatic selection
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=None, mode='segmentation'
        )
        
        # Use the automatic seed selection from segment_mesh.py
        seed_idx = select_seeds(current_face_centers, n_seeds)
        print(f"Automatically selected seed indices: {seed_idx}")
        
        # Perform segmentation
        face_labels = segment_mesh(current_sparse_matrix, seed_idx)
        
        # Export segments
        export_segment(current_mesh, face_labels, seed_idx, output_dir)
        
        # Create segment colors for visualization
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8], [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8]
        ])
        
        # Initialize all faces with default color
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.7, 0.7, 0.7], dtype=np.float32)
        
        # Create mapping of seed indices to segments
        seed_to_segment = {seed: i for i, seed in enumerate(seed_idx)}
        
        # Color segments
        for face_idx, seed_face in face_labels.items():
            if seed_face in seed_to_segment:
                segment_id = seed_to_segment[seed_face]
                color_idx = segment_id % len(colors)
                face_colors[face_idx] = colors[color_idx]
        
        # Convert to list format for JSON serialization
        face_colors_list = face_colors.tolist()
        
        print(f"Automatic segmentation completed with {len(seed_idx)} segments")
        
        return jsonify({
            'success': True,
            'segments_created': len(seed_idx),
            'face_colors': face_colors_list,
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
            current_mesh, current_curvature_penalty, user_seeds=seed_face_indices, mode='segmentation'
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

@app.route('/visualize_geodesic_distance', methods=['POST'])
def visualize_geodesic_distance():
    """
    Calculate and return geodesic distances from a clicked point for visualization.
    """
    global current_mesh, current_sparse_matrix

    try:
        data = request.json
        clicked_point = data.get('clicked_point')
        curvature_penalty_strength = data.get('curvature_penalty_strength', 100.0)

        if clicked_point is None:
            return jsonify({'success': False, 'error': 'No clicked point provided'})

        if current_mesh is None or current_sparse_matrix is None:
            return jsonify({'success': False, 'error': 'No mesh loaded or graph not built'})

        # Transform clicked point back to original mesh coordinates
        vertices = current_mesh.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)

        display_scale = 4.0 / max_dim if max_dim > 0 else 1.0
        display_center = -center

        original_point = (np.array(clicked_point) - display_center) / display_scale

        # Find the closest face to the clicked point
        face_centers = current_mesh.triangles_center
        distances_to_faces = np.linalg.norm(face_centers - original_point, axis=1)
        seed_face_idx = np.argmin(distances_to_faces)

        print(f"Visualizing geodesic distance from seed face: {seed_face_idx}")

        # Build graph with curvature penalties
        print("Building graph with curvature penalties...")
        sparse_matrix, _ = build_adjacency_graph(current_mesh, curvature_penalty_strength)

        # Calculate geodesic distances from the seed face
        from scipy.sparse.csgraph import dijkstra
        geodesic_distances = dijkstra(
            csgraph=sparse_matrix,
            directed=False,
            indices=seed_face_idx,
            unweighted=False
        )

        # Handle infinite distances (unreachable faces)
        finite_mask = np.isfinite(geodesic_distances)
        if not np.any(finite_mask):
            return jsonify({'success': False, 'error': 'No finite geodesic distances found.'})

        finite_distances = geodesic_distances[finite_mask]
        min_dist = np.min(finite_distances)
        max_dist = np.max(finite_distances)

        print(f"Geodesic distance range: [{min_dist}, {max_dist}]")

        # Normalize distances to [0, 1] for colormapping
        if max_dist > min_dist:
            normalized_distances = (geodesic_distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros_like(geodesic_distances)

        # Prepare face colors for visualization
        total_faces = len(current_mesh.faces)
        face_colors = np.full((total_faces, 3), [0.5, 0.5, 0.5], dtype=np.float32)  # Default gray for unreachable faces

        # Apply simple green-to-red colormap to finite-distance faces
        colors_reachable = np.zeros((total_faces, 3))

        colors_reachable[finite_mask, 0] = normalized_distances[finite_mask]  # Red channel
        colors_reachable[finite_mask, 1] = 1.0 - normalized_distances[finite_mask]  # Green channel
        colors_reachable[finite_mask, 2] = 0.0  # Blue channel

        face_colors[finite_mask] = colors_reachable[finite_mask]

        return jsonify({
            'success': True,
            'face_colors': face_colors.tolist()
        })

    except Exception as e:
        import traceback
        print(f"Error in geodesic distance visualization: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/visualize_geodesic_with_penalty', methods=['POST'])
def visualize_geodesic_with_penalty():
    """
    Visualize geodesic distances with curvature penalties.
    """
    global current_mesh, current_sparse_matrix, current_face_centers

    try:
        data = request.json
        curvature_penalty_strength = data.get('curvature_penalty_strength', 100.0)

        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})

        # Build adjacency graph with curvature penalties
        sparse_matrix, face_centers = build_adjacency_graph(current_mesh, curvature_penalty_strength)

        # Compute geodesic distances from a random seed
        seed_idx = np.random.choice(face_centers.shape[0])
        distances = np.zeros(face_centers.shape[0])

        for i in range(face_centers.shape[0]):
            distances[i] = sparse_matrix[seed_idx, i]

        # Normalize distances for visualization
        distances = np.nan_to_num(distances, nan=np.max(distances))
        distances = (distances - distances.min()) / (distances.max() - distances.min())

        # Log data for debugging
        print("Face Centers:", face_centers[:10])
        print("Distances:", distances[:10])

        # Prepare data for visualization
        visualization_data = {
            'face_centers': face_centers.tolist(),
            'distances': distances.tolist()
        }

        return jsonify({'success': True, 'visualization_data': visualization_data})

    except Exception as e:
        import traceback
        print(f"Error in geodesic visualization: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/facility_place_seeds', methods=['POST'])
def facility_place_seeds():
    """Use facility placement algorithms to optimally place seeds on the mesh."""
    global current_mesh, current_sparse_matrix, current_face_centers, current_curvature_penalty
    
    try:
        data = request.json
        num_seeds = data.get('num_seeds', 5)
        strategy = data.get('strategy', 'adaptive_hybrid')
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
            
        if not (1 <= num_seeds <= 20):
            return jsonify({'success': False, 'error': 'Number of seeds must be between 1 and 20'})
            
        print(f"Facility placement: {num_seeds} seeds using {strategy} strategy")
        
        # Import the facility placement function
        try:
            from src.services.automatic_placement import automatic_seed_placement
        except ImportError:
            return jsonify({'success': False, 'error': 'Facility placement module not available'})
        
        # Build adjacency graph with penalties (using facility placement mode)
        penalty_matrix, face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=None, mode='facility_placement'
        )
        
        # Get face centers for spatial algorithms
        mesh_face_centers = current_mesh.triangles_center
        
        # Run facility placement algorithm
        optimal_seed_indices = automatic_seed_placement(
            adjacency_graph=penalty_matrix,
            num_seeds=num_seeds,
            face_centers=mesh_face_centers,
            strategy=strategy,
            max_computation_time=30.0,
            verbose=True
        )
        
        print(f"Facility placement found seeds at face indices: {optimal_seed_indices}")
        
        # Convert face indices to 3D positions
        seed_positions = []
        for face_idx in optimal_seed_indices:
            position = mesh_face_centers[face_idx]
            seed_positions.append(position.tolist())
        
        # Transform positions to display coordinate system
        vertices = current_mesh.vertices
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dim = np.max(size)
        
        display_scale = 4.0 / max_dim if max_dim > 0 else 1.0
        display_center = -center
        
        display_positions = []
        for pos in seed_positions:
            display_pos = (np.array(pos) + display_center) * display_scale
            display_positions.append(display_pos.tolist())
        
        # Assign colors to seeds
        color_palette = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
            'orange', 'purple', 'pink', 'lime', 'brown', 'teal',
            'navy', 'maroon', 'olive', 'aqua', 'silver', 'gray',
            'fuchsia', 'indigo'
        ]
        
        colored_seeds = []
        for i, pos in enumerate(display_positions):
            color = color_palette[i % len(color_palette)]
            colored_seeds.append({
                'position': pos,
                'color': color,
                'face_idx': int(optimal_seed_indices[i])
            })
        
        return jsonify({
            'success': True,
            'seeds': colored_seeds,
            'algorithm_info': f'Facility placement using {strategy} strategy',
            'num_seeds_placed': len(colored_seeds),
            'strategy_used': strategy
        })
        
    except Exception as e:
        import traceback
        print(f"Error in facility placement: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/update_penalty', methods=['POST'])
def update_penalty():
    """Update the curvature penalty and rebuild the adjacency graph."""
    global current_mesh, current_sparse_matrix, current_face_centers, current_curvature_penalty
    
    try:
        data = request.json
        new_penalty = float(data.get('curvature_penalty_strength'))
        
        if current_mesh is None:
            return jsonify({'success': False, 'error': 'No mesh loaded'})
            
        current_curvature_penalty = new_penalty
        
        print(f"Updating curvature penalty to: {new_penalty}")
        
        # Rebuild the graph with the new penalty
        current_sparse_matrix, current_face_centers = build_adjacency_graph(
            current_mesh, current_curvature_penalty, user_seeds=None
        )
        
        return jsonify({'success': True, 'message': f'Penalty updated to {new_penalty}'})
        
    except Exception as e:
        import traceback
        print(f"Error updating penalty: {e}")
        print("Full traceback:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
