

import trimesh
import numpy as np
from scipy.sparse import csgraph
import scipy.sparse as sparse
import os
import argparse

import time # Debug

def load_and_clean_mesh(mesh_path, texture_path=None):
    """
    Load and clean a 3D mesh.
    """
    if texture_path and os.path.exists(texture_path):
        mesh = trimesh.load(mesh_path, process=False)

        from PIL import Image
        texture_image = Image.open(texture_path)

        material = trimesh.visual.material.SimpleMaterial(image=texture_image)

        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv,
                material=material
            )

        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
    else:
        mesh = trimesh.load(mesh_path, process=True)
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()

    return mesh

def extract_face_colors(mesh):
    # Vertex colors
    if getattr(mesh.visual, 'vertex_colors', None) is not None and mesh.visual.vertex_colors.size:
        v_rgb = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        face_rgb = v_rgb[mesh.faces].mean(axis=1)

    # UV texture
    elif hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and hasattr(mesh.visual, "material") and hasattr(
            mesh.visual.material, "image"):
        tex = np.asarray(mesh.visual.material.image.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = tex.shape

        face_uvs = mesh.visual.uv[mesh.faces]
        uv_centers = face_uvs.mean(axis=1)

        px = (uv_centers[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
        py = ((1 - uv_centers[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
        face_rgb = tex[py, px]
    else:
        return None

    # Convert to LAB
    from skimage.color import rgb2lab
    face_lab = rgb2lab(face_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    return face_lab


def compute_enhanced_distance_metrics(mesh, adj, user_seeds=None):
    """
    Compute enhanced distance metrics for user-seeded segmentation.
    
    Args:
        mesh: trimesh object
        adj: face adjacency pairs
        user_seeds: list of user-selected face indices (optional)
    
    Returns:
        dict with various distance components
    """
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    edge_lengths = mesh.edges_unique_length
    
    avg_edge_length = np.mean(edge_lengths)
    avg_face_area = np.mean(face_areas)
    
    # Vectorized calculations
    p1 = face_centers[adj[:, 0]]
    p2 = face_centers[adj[:, 1]]
    n1 = face_normals[adj[:, 0]]
    n2 = face_normals[adj[:, 1]]
    a1 = face_areas[adj[:, 0]]
    a2 = face_areas[adj[:, 1]]
    
    # Basic metrics
    spatial_dist = np.linalg.norm(p1 - p2, axis=1)
    normal_angle = np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))
    
    # Enhanced metrics for user-seeded segmentation
    area_ratio = np.abs(a1 - a2) / (a1 + a2 + 1e-8)  # Area similarity
    
    # Curvature-based features
    shape_index_1 = compute_shape_index(mesh, adj[:, 0])
    shape_index_2 = compute_shape_index(mesh, adj[:, 1])
    shape_similarity = np.abs(shape_index_1 - shape_index_2)
    
    # User seed affinity (if user seeds provided)
    seed_affinity = np.ones_like(spatial_dist)
    if user_seeds is not None and len(user_seeds) > 0:
        seed_affinity = compute_user_seed_affinity(adj, user_seeds, face_centers)
    
    return {
        'spatial_dist': spatial_dist,
        'normal_angle': normal_angle,
        'area_ratio': area_ratio,
        'shape_similarity': shape_similarity,
        'seed_affinity': seed_affinity,
        'avg_edge_length': avg_edge_length
    }

def compute_shape_index(mesh, face_indices):
    """
    Compute shape index for faces (simplified, fast version).
    Uses face area as a proxy for curvature to avoid expensive neighbor searches.
    """
    try:
        # Fast approximation: use normalized face area as shape indicator
        face_areas = mesh.area_faces[face_indices]
        avg_area = np.mean(mesh.area_faces)
        
        # Normalize to [0, 1] range where 0 = flat, 1 = highly curved
        shape_proxy = np.abs(face_areas - avg_area) / (avg_area + 1e-8)
        return np.clip(shape_proxy, 0, 2)  # Cap at 2 for extreme cases
        
    except Exception:
        # Fallback to zeros if computation fails
        return np.zeros(len(face_indices))

def compute_user_seed_affinity(adj, user_seeds, face_centers):
    """
    Compute affinity based on proximity to user-selected seed regions.
    Faces closer to seed regions get lower distance penalties.
    Optimized vectorized version.
    """
    if not user_seeds:
        return np.ones(len(adj))
    
    # Convert user seeds to face centers if needed
    seed_positions = []
    for seed in user_seeds:
        if isinstance(seed, (list, tuple)) and len(seed) == 3:
            # Seed is a 3D point, find closest face
            distances = np.linalg.norm(face_centers - np.array(seed), axis=1)
            closest_face = np.argmin(distances)
            seed_positions.append(face_centers[closest_face])
        else:
            # Seed is already a face index
            seed_positions.append(face_centers[seed])
    
    seed_positions = np.array(seed_positions)
    
    # Vectorized computation of edge midpoints
    face1_pos = face_centers[adj[:, 0]]
    face2_pos = face_centers[adj[:, 1]]
    edge_midpoints = (face1_pos + face2_pos) / 2
    
    # Vectorized distance computation to all seeds
    # Shape: (n_edges, n_seeds, 3)
    edge_to_seeds = edge_midpoints[:, np.newaxis, :] - seed_positions[np.newaxis, :, :]
    distances_to_seeds = np.linalg.norm(edge_to_seeds, axis=2)
    
    # Get minimum distance to any seed for each edge
    min_distances = np.min(distances_to_seeds, axis=1)
    
    # Convert to affinity using exponential decay
    affinities = np.exp(-min_distances * 2.0)  # Adjust decay rate as needed
    
    return affinities

def build_adjacency_graph(mesh, curvature_penalty_strength, texture_strength=5, max_normal_angle=np.radians(60), user_seeds=None, enhanced_mode=False):
    """
    Build a face adjacency graph with curvature-aware edge weights.
    
    Args:
        mesh: trimesh object
        curvature_penalty_strength: float, strength of curvature penalty
        texture_strength: float, strength of texture_penalty
        max_normal_angle: float, maximum angle between face normals to consider adjacent
        user_seeds: list, user-selected seed points or face indices
        enhanced_mode: bool, whether to use enhanced distance metrics
    
    Returns:
         sparse_matrix : scipy.sparse.csr_matrix, shape (m, m)
        Weighted adjacency matrix of the filtered face graph. m is the number of faces
        that remain after filtering. Entry (i, j) contains the weight between face i and j,
        combining spatial distance and curvature penalty.
    face_coords : numpy.ndarray, shape (m, 3)
        Array of 3D centroids corresponding to the faces in the graph. Row index i of
        `sparse_matrix` maps directly to `face_coords[i]`.
    """

    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    edge_lengths = mesh.edges_unique_length
    adj = mesh.face_adjacency

    avg_edge_length = np.mean(edge_lengths)

    if enhanced_mode:
        # Use enhanced distance metrics for user-seeded segmentation
        distance_metrics = compute_enhanced_distance_metrics(mesh, adj, user_seeds)
        
        spatial_dist = distance_metrics['spatial_dist']
        angle = distance_metrics['normal_angle']
        area_ratio = distance_metrics['area_ratio']
        shape_similarity = distance_metrics['shape_similarity']
        seed_affinity = distance_metrics['seed_affinity']
        
        # Filter edges based on angle threshold
        mask = angle <= max_normal_angle
        
        # Enhanced weight calculation
        curvature_penalty = np.exp(curvature_penalty_strength * angle[mask])
        spatial_penalty = 1 + (spatial_dist[mask] / avg_edge_length) ** 2

        # Texture penalty
        face_lab = extract_face_colors(mesh)
        if face_lab is None:
            texture_penalty = np.ones_like(spatial_penalty)
        else:
            lab1 = face_lab[adj[mask][:, 0]]
            lab2 = face_lab[adj[mask][:, 1]]

            deltaE_norm = np.linalg.norm(lab1 - lab2, axis=1) / 100.0
            texture_penalty = np.exp(texture_strength * deltaE_norm)

        area_penalty = 1 + area_ratio[mask] * 0.5  # Penalize area differences
        shape_penalty = 1 + shape_similarity[mask] * 0.3  # Penalize shape differences
        user_penalty = 2.0 - seed_affinity[mask]  # Lower penalty near user seeds
        
        weights = spatial_penalty * curvature_penalty * area_penalty * shape_penalty * user_penalty * texture_penalty
        
    else:
        # Original calculation
        # Vectorized calculations
        p1 = face_centers[adj[:, 0]]
        p2 = face_centers[adj[:, 1]]
        n1 = face_normals[adj[:, 0]]
        n2 = face_normals[adj[:, 1]]

        spatial_dist = np.linalg.norm(p1 - p2, axis=1)
        angle = np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))

        # Filter edges based on angle threshold
        mask = angle <= max_normal_angle

        curvature_penalty = np.exp(curvature_penalty_strength * angle[mask])
        spatial_penalty = 1 + (spatial_dist[mask] / avg_edge_length) ** 2

        # Texture penalty
        face_lab = extract_face_colors(mesh)
        if face_lab is None:
            texture_penalty = np.ones_like(spatial_penalty)
        else:
            lab1 = face_lab[adj[mask][:, 0]]
            lab2 = face_lab[adj[mask][:, 1]]

            deltaE_norm = np.linalg.norm(lab1 - lab2, axis=1) / 100.0
            texture_penalty = np.exp(texture_strength * deltaE_norm)

        weights = spatial_penalty * curvature_penalty * texture_penalty

    # Filters the edges
    adj_valid = adj[mask]

    active_faces = np.unique(adj_valid.flatten())

    face_coords = face_centers[active_faces]

    row = np.searchsorted(active_faces, adj_valid[:, 0])
    col = np.searchsorted(active_faces, adj_valid[:, 1])
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    all_weights = np.concatenate([weights, weights]).astype(np.float64)

    sparse_matrix = sparse.csr_matrix(
        (all_weights, (all_row, all_col)),
        shape=(len(active_faces),) * 2, dtype=np.float64)

    print("Graph built")  # DEBUG
    return sparse_matrix, face_coords, active_faces

def pick_first_seed(face_coords,  pool_size=64):
    """
    Picks a pool of faces at random, selects the one with the greatest average distance from the pool.

    Args:
        face_coords: numpy.ndarray, shape (m, 3)
        pool_size: int, number of faces in the pool
    """
    rng = np.random.default_rng(42)
    n_faces = face_coords.shape[0]

    pool = rng.choice(n_faces, size=pool_size, replace=False)
    sub = face_coords[pool]
    dist = np.linalg.norm(sub[:, None] - sub[None], axis=2)

    return pool[np.argmax(dist.sum(axis=1))]


def select_seeds(face_coords, n_seeds):
    """
    Select seed faces using farthest-point sampling.
    Args:
        face_coords: (m×3) array of 3D coordinates for active faces
        n_seeds: number of seeds to select

    Returns:
        seed_idx: array of matrix row indices [0, m-1]
    """
    rng = np.random.default_rng(42)
    n_faces = face_coords.shape[0]

    seed_idx = [pick_first_seed(face_coords)]
    d_min = np.linalg.norm(face_coords - face_coords[seed_idx[0]], axis=1)

    for _ in range(1, n_seeds):
        probs = d_min / d_min.sum()
        new_seed = rng.choice(n_faces, p=probs)
        seed_idx.append(new_seed)

        d_new = np.linalg.norm(face_coords - face_coords[new_seed], axis=1)
        d_min = np.minimum(d_min, d_new)

    print("seeds selected")  # DEBUG
    return np.array(seed_idx)


def segment_mesh(sparse_matrix, seed_idx):
    """
    Segment a mesh by multi‑source geodesic propagation on its adjacency matrix.

    Returns:
         face_labels : dict
        Mapping `{face_i: seed_j}` where both keys and values are row indices into
        `sparse_matrix`. Each face_i is assigned the seed_j to which it has the
        shortest geodesic (edge‑weight) distance. Faces unreachable from any seed
        (distance = inf) are omitted.

    Note:
        Ties in distance are broken by adding a small epsilon offset to each seed’s
      distance array, favoring seeds with lower index in `seed_idx`.
    """

    # Multi-source dijkstra on all seed nodes (seed_idx)
    print(f"Running dijkstra from {len(seed_idx)} seeds: {seed_idx}")
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx,
                            directed=False, return_predecessors=False)
    
    # Check for any infinite distances that shouldn't be there
    inf_count = np.sum(~np.isfinite(dist), axis=None)
    total_entries = dist.size
    print(f"Distance matrix: {inf_count}/{total_entries} infinite entries ({100*inf_count/total_entries:.2f}%)")

    # Generate offsets for each seed so that the earliest seed wins on exact distance matches
    eps = np.linspace(0.0, 1e-9, len(seed_idx), endpoint=False)[:, None]
    winner = (dist + eps).argmin(axis=0)

    # Create face labels mapping matrix indices to seed indices
    face_labels = {}
    reachable_faces = 0
    unreachable_faces = 0
    
    for i in range(sparse_matrix.shape[0]):
        if np.isfinite(dist[winner[i], i]):
            face_labels[i] = seed_idx[winner[i]]
            reachable_faces += 1
        else:
            unreachable_faces += 1

    print(f"Segmentation results: {reachable_faces} reachable faces, {unreachable_faces} unreachable faces")
    
    if unreachable_faces > 0:
        print(f"Warning: {unreachable_faces} faces are unreachable from any seed.")
        
        # Attempt to handle unreachable faces by finding connected components
        # Only do this if unreachable faces are significant (> 1% of total)
        if unreachable_faces > sparse_matrix.shape[0] * 0.01:
            print(f"Unreachable faces ({unreachable_faces}) > 1% of total. Computing connected components...")
            n_components, component_labels = csgraph.connected_components(sparse_matrix, directed=False)
            
            if n_components > 1:
                print(f"Mesh has {n_components} disconnected components. Assigning isolated components to nearest seeds...")
                
                # Find which components have seeds
                seeded_components = set()
                for seed in seed_idx:
                    if seed < len(component_labels):
                        seeded_components.add(component_labels[seed])
                
                # For each unseeded component, assign faces to the first available seed
                assigned = 0
                for i in range(sparse_matrix.shape[0]):
                    if i not in face_labels:  # This face was unreachable
                        component = component_labels[i]
                        if component not in seeded_components:
                            if len(seed_idx) > 0:
                                face_labels[i] = seed_idx[0]  # Assign to first seed
                                assigned += 1
                
                reachable_faces += assigned
                unreachable_faces -= assigned
                print(f"Assigned {assigned} faces from isolated components. New stats: {reachable_faces} reachable, {unreachable_faces} unreachable")
        else:
            print(f"Unreachable faces ({unreachable_faces}) < 1% of total. Skipping expensive component analysis.")
    
    return face_labels


def export_segments(mesh, face_labels, seed_idx, active_faces, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    row_to_segment = {row: i for i, row in enumerate(seed_idx)}
    segments = [[] for _ in range(len(seed_idx))]

    for row_idx, seed_row in face_labels.items():
        seg_id = row_to_segment[seed_row]
        face_id = active_faces[row_idx]
        segments[seg_id].append(face_id)

    for i, face_ids in enumerate(segments):
        if face_ids:
            mesh.submesh([face_ids], append=True)\
                .export(os.path.join(output_dir, f"segment_{i}.obj"))


def main():
    parser = argparse.ArgumentParser(description="3D Mesh Segmentation")
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=r"input\run\example.obj",
        help="Path to mesh file (default: input\\run\\example.obj)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to output directory (default: output)",
    )
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of segments")
    parser.add_argument(
        "--curvature_penalty_strength",
        type=float,
        default=100.0,
        help="Angle punishment strength",
    )
    parser.add_argument("--texture_strength", type=float, default=5.0,
                        help="Texture difference penalty strength")
    parser.add_argument(
        "--seed_idx",
        type=int,
        nargs="*",
        help="Manual seed face indices (optional)",
    )

    args = parser.parse_args()

    t0 = time.perf_counter()  # DEBUG

    print("Segmentation Started")

    mesh = load_and_clean_mesh(args.mesh_path)
    print("Faces:", len(mesh.faces))  # DEBUG
    sparse_matrix, face_coords, active_faces = build_adjacency_graph(mesh, args.curvature_penalty_strength)

    if args.seed_idx is None:
        seed_idx = select_seeds(face_coords, args.n_seeds)
    else:
        seed_idx = np.array(args.seed_idx, dtype=int)

    print(f"Using seed face indices: {seed_idx}")
    face_labels = segment_mesh(sparse_matrix, seed_idx)

    export_segments(mesh, face_labels, seed_idx, active_faces, args.output_dir)
    print("Segmentation complete.")

    print("Elapsed:", time.perf_counter() - t0) # DEBUG

if __name__ == "__main__":
    main()
