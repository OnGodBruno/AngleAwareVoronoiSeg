import trimesh
import numpy as np
from scipy.sparse import csgraph
import scipy.sparse as sparse
import os
import argparse

import time # Debug


def load_and_clean_mesh(mesh_path):
    """
    Load and clean a 3D mesh.
    """
    mesh = trimesh.load(mesh_path, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    return mesh


def build_adjacency_graph(mesh, curvature_penalty_strength, user_seeds=None, return_stats=False, mode='segmentation'):
    """
    Builds a face adjacency graph with directional curvature-aware edge weights.

    Args:
         mesh: trimesh object
         curvature_penalty_strength: float, strength of curvature penalty
         user_seeds: list, user-selected seed points or face indices
         return_stats: bool, whether to return curvature statistics
         mode: str, adjacency graph building mode ('segmentation' or 'facility_placement')
    Returns:
         sparse_matrix : scipy.sparse.csr_matrix, shape (N, N)
            Weighted adjacency matrix of the filtered face graph. N is the number of faces.
            Entry (i, j) contains the weight between face i and j,
            combining spatial distance and directional curvature penalty.
        face_centers : numpy.ndarray, shape (N, 3)
            Array of 3D centroids corresponding to the faces in the graph. Row index i of
            `sparse_matrix` maps directly to `face_centers[i]`.
        curvature_stats : dict (optional, if return_stats=True)
            Dictionary containing curvature penalty statistics
    """
    # Validate mode parameter
    if mode not in ['segmentation', 'facility_placement']:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'segmentation' or 'facility_placement'")
    
    print(f"Building adjacency graph using '{mode}' mode...")
    
    # Get basic mesh data
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    edge_lengths = mesh.edges_unique_length
    adj = mesh.face_adjacency

    avg_edge_length = np.mean(edge_lengths)

    # List of neighbors in adjacency
    p1 = face_centers[adj[:, 0]]
    p2 = face_centers[adj[:, 1]]
    n1 = face_normals[adj[:, 0]]
    n2 = face_normals[adj[:, 1]]

    spatial_dist = np.linalg.norm(p2 - p1, axis=1)
    angle = np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))

    # Apply mode-specific weight calculations
    if mode == 'segmentation':
        valley_barrier_alpha: float = 20.0
        # Penalties
        curvature_penalty = np.exp(curvature_penalty_strength * angle)
        spatial_penalty = 1 + (spatial_dist / avg_edge_length )**2
        
        base_weights = spatial_penalty + curvature_penalty

        # Find valleys and propagate penalties to nearby faces
        print("Finding valleys...")
        valley_edge_scores = find_valleys(mesh, angle_threshold_deg=20.0,
                                        p=1.0, normal_smoothing=False)
        
        valley_edges_count = np.sum(valley_edge_scores > 0)
        print(f"Found {valley_edges_count} valley edges out of {len(valley_edge_scores)} total edges")
        
        # Propagate valley penalties to faces within 3 face distances
        print("Propagating valley penalties to nearby faces...")
        face_valley_penalties = propagate_valley_penalties(mesh, valley_edge_scores, max_distance=3)
        
        affected_faces = np.sum(face_valley_penalties > 0)
        print(f"Valley penalties applied to {affected_faces} faces out of {len(mesh.faces)} total faces")
        
        # Apply valley penalties to edges based on the faces they connect
        edge_valley_penalties = np.maximum(face_valley_penalties[adj[:, 0]], 
                                         face_valley_penalties[adj[:, 1]])

        eps = 1e-6
        
        tau = float(valley_barrier_alpha) * float(np.median(base_weights)) if base_weights.size else float(valley_barrier_alpha)
        barrier = tau * (edge_valley_penalties / (1.0 - edge_valley_penalties + eps))

        # Apply strong penalty where valleys are detected (including propagated areas)
        valley_mask = edge_valley_penalties > 0
        weights = base_weights.copy()
        weights[valley_mask] = edge_valley_penalties[valley_mask] * 100000 + 1

        
    elif mode == 'facility_placement':
        # Facility placement mode - same as segmentation for now
        curvature_penalty = curvature_penalty_strength * angle
        spatial_penalty = (spatial_dist / avg_edge_length)
        weights = spatial_penalty + curvature_penalty * 10
    
    # Store statistics for reporting
    curvature_stats = {
        'min': float(curvature_penalty.min()),
        'max': float(curvature_penalty.max()),
        'mean': float(curvature_penalty.mean()),
        'median': float(np.median(curvature_penalty)),
        'strength': curvature_penalty_strength,
        'mode': mode
    }
    
    print(f"{mode.title()} mode - Curvature penalty - Min: {curvature_stats['min']:.4f}, Max: {curvature_stats['max']:.4f}, Mean: {curvature_stats['mean']:.4f}, Median: {curvature_stats['median']:.4f}")

    # Build adjacency matrix
    row = adj[:, 0]
    col = adj[:, 1]
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    all_weights = np.concatenate([weights, weights]).astype(np.float64)

    # Create sparse matrix
    N = face_centers.shape[0]
    sparse_matrix = sparse.csr_matrix((all_weights, (all_row, all_col)),
                                      shape=(N, N), dtype=np.float64)

    if return_stats:
        return sparse_matrix, face_centers, curvature_stats
    else:
        return sparse_matrix, face_centers

def propagate_valley_penalties(mesh, valley_scores, max_distance=3):
    """
    Efficiently propagate valley penalties to faces within max_distance of valley edges using BFS.
    
    Args:
        mesh: trimesh object
        valley_scores: array of valley scores per edge from find_valleys
        max_distance: maximum face distance to propagate valley penalties
    
    Returns:
        face_valley_penalties: array of valley penalties per face
    """
    from collections import deque
    
    # Build face adjacency graph as a dictionary for fast lookup
    adj = mesh.face_adjacency
    n_faces = len(mesh.faces)
    
    if len(adj) == 0 or len(valley_scores) == 0:
        return np.zeros(n_faces)
    
    # Build adjacency list for efficient neighbor lookup
    face_neighbors = [[] for _ in range(n_faces)]
    for i, (face1, face2) in enumerate(adj):
        face_neighbors[face1].append(face2)
        face_neighbors[face2].append(face1)
    
    # Initialize face valley penalties
    face_valley_penalties = np.zeros(n_faces)
    
    # Find all valley edges and their connected faces
    valley_edges = np.where(valley_scores > 0)[0]
    
    if len(valley_edges) == 0:
        return face_valley_penalties
    
    # Collect all valley faces with their maximum valley scores
    valley_face_scores = {}
    for edge_idx in valley_edges:
        valley_score = valley_scores[edge_idx]
        face1, face2 = adj[edge_idx]
        
        # Keep the maximum valley score for each face
        valley_face_scores[face1] = max(valley_face_scores.get(face1, 0), valley_score)
        valley_face_scores[face2] = max(valley_face_scores.get(face2, 0), valley_score)
    
    # Multi-source BFS from all valley faces simultaneously
    queue = deque()
    visited = np.full(n_faces, -1, dtype=int)  # -1 means unvisited, otherwise stores distance
    
    # Initialize queue with all valley faces at distance 0
    for face_idx, valley_score in valley_face_scores.items():
        queue.append((face_idx, 0, valley_score))
        visited[face_idx] = 0
        face_valley_penalties[face_idx] = valley_score
    
    # BFS propagation
    while queue:
        current_face, distance, original_score = queue.popleft()
        
        if distance >= max_distance:
            continue
        
        # Visit all neighbors
        for neighbor_face in face_neighbors[current_face]:
            new_distance = distance + 1
            
            if new_distance <= max_distance:
                # Calculate penalty with distance decay
                distance_factor = 1.0 - (new_distance / max_distance)
                penalty = original_score * distance_factor
                
                # Only update if this is a better (closer) path or unvisited
                if visited[neighbor_face] == -1 or (visited[neighbor_face] > new_distance):
                    visited[neighbor_face] = new_distance
                    face_valley_penalties[neighbor_face] = max(face_valley_penalties[neighbor_face], penalty)
                    queue.append((neighbor_face, new_distance, original_score))
                elif visited[neighbor_face] == new_distance:
                    # Same distance, take maximum penalty
                    face_valley_penalties[neighbor_face] = max(face_valley_penalties[neighbor_face], penalty)
    
    return face_valley_penalties


def find_valleys(mesh, angle_threshold_deg: float = 20.0, p: float = 2.0, normal_smoothing: bool = True) -> np.ndarray:
    """
    Finds concave edges by calculating the hinge vector and comparing it to the cross product of the normals. Calculates a valley score for each edge.
    valley_score = ((-theta - angle_threshold_degree) / (180 - angle_threshold_degree))^p for theta < -angle_threshold_degree else 0.0
    That means the score is between 0.0 and 1.0, where 1.0 is a perfectly concave edge (theta = -180 degree) and 0.0 is a flat edge (theta = 0 degree), convex edge or a concave edge under the angle threshold.
    Args:
        angle_threshold_deg (float, optional): The angle threshold at which an angle is accepted as a concave edge. Defaults to 20.0.
        p (float, optional): p > 1 emphasizes sharp valleys (slow rise near threshold, fast toward 1), .p < 1 makes it "softer". Defaults to 2.0.
        normal_smoothing (bool, optional): Whether to use smoothed normals for angle calculation. Defaults to True.
    Returns:
        numpy.ndarray of float: contains the valley score for each edge in the face adjacency (shape (num_edges, 1))
    """
    adj = mesh.face_adjacency
    n1 = mesh.face_normals[adj[:, 0]]
    n2 = mesh.face_normals[adj[:, 1]]
    edges = mesh.face_adjacency_edges  # (vert_a, vert_b)
    vertices = mesh.vertices
    faces = mesh.faces
    
    if len(adj) == 0:
        return np.zeros((0, 1), dtype=float)
    
    faces_L = faces[adj[:, 0]]
    a = edges[:, 0]                  
    b = edges[:, 1]
    
    # mask for every valid edge
    mask_a = faces_L == a[:, None]
    mask_b = faces_L == b[:, None]
    has_a = mask_a.any(axis=1)
    has_b = mask_b.any(axis=1)
    valid_in_face = has_a & has_b

    # index idx of the vertex a and b in the face
    idx_a = mask_a.argmax(axis=1)
    idx_b = mask_b.argmax(axis=1)

    # index idx of the next vertex in the face
    nxt_a_idx = (idx_a + 1) % 3
    nxt_b_idx = (idx_b + 1) % 3
    rows = np.arange(len(adj))
    
    # value of the next vertex in the face
    nxt_a_val = faces_L[rows, nxt_a_idx]
    nxt_b_val = faces_L[rows, nxt_b_idx]

    # start and end vertex of the hinge (depending on the face orientation)
    cond1 = valid_in_face & (nxt_a_val == b)
    cond2 = valid_in_face & (nxt_b_val == a)
    start = np.where(cond1, a, np.where(cond2, b, a))
    end = np.where(cond1, b, np.where(cond2, a, b))

    hinges = vertices[end] - vertices[start]
    norms = np.linalg.norm(hinges, axis=1)
    mask_ok = valid_in_face & (norms > 0)  # only norm vectors with length > 0
    hinges[mask_ok] = hinges[mask_ok] / norms[mask_ok, None]
    hv  = hinges[mask_ok] # contains all valid hinge vectors
    
    # raw face normals
    n1r = n1[mask_ok]
    n2r = n2[mask_ok]

    if normal_smoothing:
        # smoothed face normals
        N_smooth = smooth_normals(mesh, k=2, sigma_deg=25.0)
        n1s = N_smooth[adj[:, 0]][mask_ok]
        n2s = N_smooth[adj[:, 1]][mask_ok]
    
        # absolute value of the sinus and the cosine are calculated with the smoothed normals, the sign with the raw normals
        sin_mag = np.linalg.norm(np.cross(n1s, n2s), axis=1)
        sign = np.sign(np.einsum('ij,ij->i', np.cross(n1r, n2r), hv))
        
        # tan(angle) = sin(angle) / cos(angle) => angle = atan2(sin(angle), cos(angle))
        sin = sign * sin_mag
        cos = np.einsum('ij,ij->i', n1s, n2s).clip(-1, 1)
    else:
        # tan(angle) = sin(angle) / cos(angle) => angle = atan2(sin(angle), cos(angle))
        cos = np.einsum('ij,ij->i', n1r, n2r).clip(-1, 1)
        sin = np.einsum('ij,ij->i', np.cross(n1r, n2r), hv)
    
    theta_deg = np.degrees(np.arctan2(sin, cos)) # range [-180°, 180°]

    denom = max(1e-9, (180.0 - angle_threshold_deg)) # avoid division by zero
    sharp = np.clip(((-theta_deg) - angle_threshold_deg) / denom, 0.0, 1.0)
    valley_valid = np.where(theta_deg < -angle_threshold_deg, np.power(sharp, p), 0.0)

    valley_scores = np.zeros(len(adj), dtype=float)
    valley_scores[mask_ok] = valley_valid
    
    return valley_scores.reshape(-1, 1).ravel()

def smooth_normals(mesh, k:int=3, sigma_deg: float | None = 25.0):
    """
    Smooth face normals of a triangular mesh using iterative neighbor averaging.

    Each face normal is updated by combining its own direction with those of
    adjacent faces, weighted by similarity and optionally by a bilateral
    (angle-based) Gaussian kernel. Neighbor contributions are sign-corrected
    so that flipped normals do not cancel out.

    Args:
        mesh: A mesh object with attributes `faces`, `face_normals`,
              and `face_adjacency` (e.g., a trimesh.Trimesh).
        k (int): Number of smoothing iterations to apply (default: 3).
        sigma_deg (float | None): Std. dev. σ of bilateral Gaussian in degrees. 
            For angle θ between normals: w = exp(-(θ²)/(2σ²)).
            Small σ → preserves sharp edges (weights ~0 if θ large). 
            Large σ → smooths across edges (weights ~1). 
             Valid range: 0 < σ ≤ 180, or None to disable.
    Returns:
        (ndarray): Array of shape (n_faces, 3) with smoothed, unit-length
        face normals.
    """
    
    N0 = mesh.face_normals
    N = N0.astype(np.float32, copy=True)
    
    adj = mesh.face_adjacency
    if adj.size == 0:
        return N

    i = adj[:, 0].astype(np.int64)
    j = adj[:, 1].astype(np.int64)

    # positive weights
    w_pos = np.ones(len(i), dtype=np.float32)

    if sigma_deg is not None:
        d = np.einsum('ij,ij->i', N[i], N[j]).clip(-1, 1)
        ang = np.degrees(np.arccos(d))
        w_pos *= np.exp(-(ang * ang) / (2.0 * (sigma_deg ** 2))).astype(w_pos.dtype, copy=False)

    # sign-weights s_ij = sign(dot(n_i, n_j))  (+1 oder -1)
    sgn = np.sign(np.einsum('ij,ij->i', N[i], N[j])).astype(w_pos.dtype, copy=False)
    w_signed = w_pos * sgn

    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    dat_signed = np.concatenate([w_signed, w_signed])      # mit Vorzeichen
    dat_pos    = np.concatenate([w_pos,    w_pos])         # immer >= 0

    # (N x N)adjancency matrix W. One just for the sign (+1 or -1), one just with positive weights
    nF = len(mesh.faces)
    W_signed = sparse.csr_matrix((dat_signed, (row, col)), shape=(nF, nF))
    W_pos    = sparse.csr_matrix((dat_pos,    (row, col)), shape=(nF, nF))

    deg = (W_pos.sum(axis=1).A.ravel() + 1.0).astype(W_pos.dtype, copy=False)

    # k iterations: N <- (N + W_signed @ N) / deg, then normalize
    for _ in range(max(0, k)):
        tmp = W_signed @ N
        N = (N + tmp) / deg[:, None]
        # normalize to unit length
        nr = np.linalg.norm(N, axis=1, keepdims=True)
        np.divide(N, np.clip(nr, 1e-12, None), out=N)

    return N

def pick_first_seed(face_centers, pool_size=64):
    """
    Picks a pool of faces at random, selects the one with the greatest average distance from all the faces in the pool.
    Args:
        face_centers: numpy.ndarray, shape (N, 3)
        pool_size: int, number of faces in the pool
    """
    rng = np.random.default_rng(42)  # 42 is for debugging needs to be removed in final version
    n_faces = face_centers.shape[0]

    pool = rng.choice(n_faces, size=pool_size, replace=False)
    sub = face_centers[pool]
    dist = np.linalg.norm(sub[:, None] - sub[None], axis=2)

    max_dist = np.argmax(dist.sum(axis=1))

    return pool[max_dist]


def select_seeds(face_centers, n_seeds):
    """
        Select seed faces using stochastic farthest-point sampling.
        Args:
            face_centers: (N×3) array of 3D coordinates for active faces
            n_seeds: number of seeds to select
        Returns:
            seed_idx: array of matrix row indices [0, m-1], with length equal to n_seeds.
        """
    rng = np.random.default_rng(42)  # 42 is for debugging needs to be removed in final version
    n_faces = face_centers.shape[0]

    seed_idx = [pick_first_seed(face_centers)]
    dist = np.linalg.norm(face_centers - face_centers[seed_idx[0]], axis=1)

    for _ in range(1, n_seeds):
        probs = dist / dist.sum()
        new_seed = rng.choice(n_faces, p=probs)
        seed_idx.append(new_seed)

        new_dist = np.linalg.norm(face_centers - face_centers[new_seed], axis=1)
        dist = np.minimum(dist, new_dist)

    return np.array(seed_idx)


def segment_mesh(sparse_matrix, seed_idx):
    """
    Segment a mesh by multi‑source geodesic propagation on its adjacency matrix.
    Args:
        sparse_matrix : scipy.sparse.csr_matrix, shape (N, N)
            Weighted adjacency matrix of the filtered face graph. N is the number of faces.
            Entry (i, j) contains the weight between face i and j,
            combining spatial distance and curvature penalty.
        seed_idx: array of matrix row indices [0, m-1] containing the indexes of the selected seeds.
    Returns:
         face_labels : dict
        Mapping `{face_i: seed_j}` where both keys and values are row indices into
        `sparse_matrix`. Each face_i is assigned the seed_j to which it has the
        shortest geodesic (edge‑weight) distance.
    """
    # Modify Dijkstra to include additional penalties during propagation
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx, directed=False, return_predecessors=False)

    # Apply additional penalties based on context
    for i, seed in enumerate(seed_idx):
        for j in range(dist.shape[1]):
            if np.isfinite(dist[i, j]):
                # Example: Add a penalty based on distance and curvature
                dist[i, j] += 0.1 * dist[i, j]  # Adjust this formula as needed

    # DEBUG--
    inf_count = np.sum(~np.isfinite(dist), axis=None)
    print(f"Distance matrix after penalties: {inf_count}/{dist.size} infinite entries ({100 * inf_count / dist.size:.2f}%)")
    # --------

    winner = np.argmin(dist, axis=0)

    face_labels = {i: int(seed_idx[winner[i]]) for i in range(sparse_matrix.shape[0])}
    return face_labels


def export_segment(mesh, face_labels, seed_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    seed_to_seg = {int(face): i for i, face in enumerate(seed_idx)}

    segments = [[] for _ in range(len(seed_idx))]
    for face_i, seed_face in face_labels.items():
        seg_id = seed_to_seg[int(seed_face)]
        segments[seg_id].append(int(face_i))

    # Export main segments
    for i, face_ids in enumerate(segments):
        if not face_ids:
            continue
        sub = mesh.submesh([np.asarray(face_ids, dtype=np.int64)], append=True)
        sub.export(os.path.join(output_dir, f"segment_{i}.obj"))

    # Export individual seed faces as separate segments
    for i, seed_face_idx in enumerate(seed_idx):
        seed_sub = mesh.submesh([np.asarray([seed_face_idx], dtype=np.int64)], append=True)
        seed_sub.export(os.path.join(output_dir, f"seed_{i}.obj"))


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

    args = parser.parse_args()

    t0 = time.perf_counter()  # DEBUG

    print("Segmentation Started")

    mesh = load_and_clean_mesh(args.mesh_path)
    print("Faces:", len(mesh.faces))  # DEBUG

    sparse_matrix, face_centers = build_adjacency_graph(mesh, args.curvature_penalty_strength, user_seeds=None)
    print("Graph Built")

    seed_idx = select_seeds(face_centers, args.n_seeds)
    print("Seeds selected")
    print(f"Using seed face indices: {seed_idx}")

    face_labels = segment_mesh(sparse_matrix, seed_idx)
    print("Mesh Segmented")

    export_segment(mesh, face_labels, seed_idx, args.output_dir)
    print("Segmentation Finished")

    print("Elapsed:", time.perf_counter() - t0)  # DEBUG

if __name__ == "__main__":
    main()
