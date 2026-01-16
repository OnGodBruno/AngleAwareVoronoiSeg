import networkx as nx
def export_feature_lines_obj(mesh, feature_chains, output_path):
    """
    Export feature lines as a .obj file (as polylines).
    Args:
        mesh: trimesh.Trimesh
        feature_chains: list of chains (each chain is a list of edge indices)
        output_path: str, path to .obj file
    """
    with open(output_path, 'w') as f:
        # Write vertices
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # Write polylines for each chain
        for chain in feature_chains:
            # Each chain is a list of edge indices (into mesh.face_adjacency_edges)
            edge_verts = []
            for edge_idx in chain:
                e = mesh.face_adjacency_edges[edge_idx]
                edge_verts.extend(e)
            # Remove duplicates, keep order
            seen = set()
            poly = []
            for idx in edge_verts:
                if idx not in seen:
                    poly.append(idx)
                    seen.add(idx)
            # OBJ is 1-based
            if len(poly) > 1:
                f.write("l " + " ".join(str(i+1) for i in poly) + "\n")

def extract_feature_lines(mesh, concave_thresh=-0.1, hyperbolic_thresh=0.15, connect_radius=0.02, min_chain_length=3):
    """
    Extract and connect concave and hyperbolic edges as continuous feature lines.
    Args:
        mesh: trimesh.Trimesh
        concave_thresh: float, dihedral angle cosine threshold for concave
        hyperbolic_thresh: float, dihedral angle cosine threshold for hyperbolic
        connect_radius: float, max geodesic distance to bridge gaps
        min_chain_length: int, minimum length of feature chain to keep
    Returns:
        List of feature lines, each as a list of edge indices
    """
    adj = mesh.face_adjacency
    shared_edges = mesh.face_adjacency_edges
    n1 = mesh.face_normals[adj[:, 0]]
    n2 = mesh.face_normals[adj[:, 1]]
    dot = np.einsum('ij,ij->i', n1, n2).clip(-1, 1)
    angle = np.arccos(dot)
    # Dihedral sign
    v0 = mesh.vertices[shared_edges[:, 0]]
    v1 = mesh.vertices[shared_edges[:, 1]]
    edge_vec = v1 - v0
    cross1 = np.cross(n1, edge_vec)
    sign = np.sign(np.einsum('ij,ij->i', cross1, n2))
    # Feature edge mask
    concave_mask = (sign < 0) & (dot < concave_thresh)
    hyperbolic_mask = (np.abs(dot) < hyperbolic_thresh)
    feature_mask = concave_mask | hyperbolic_mask
    feature_edges = shared_edges[feature_mask]
    feature_edge_indices = np.where(feature_mask)[0]

    # Build edge-to-edge connectivity for feature edges
    from collections import defaultdict, deque
    edge_map = defaultdict(list)
    for idx, (a, b) in zip(feature_edge_indices, feature_edges):
        edge_map[a].append((idx, b))
        edge_map[b].append((idx, a))

    # Find chains by DFS, bridging small gaps
    visited = set()
    chains = []
    for idx, (a, b) in zip(feature_edge_indices, feature_edges):
        if idx in visited:
            continue
        chain = [idx]
        visited.add(idx)
        queue = deque([(a, b)])
        while queue:
            start, current = queue.popleft()
            for nidx, neighbor in edge_map[current]:
                if nidx not in visited:
                    # Check if this edge is close enough to connect
                    dist = np.linalg.norm(mesh.vertices[start] - mesh.vertices[neighbor])
                    if dist < connect_radius * mesh.scale:
                        chain.append(nidx)
                        visited.add(nidx)
                        queue.append((current, neighbor))
        if len(chain) >= min_chain_length:
            chains.append(chain)
    return chains
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


def build_adjacency_graph(mesh, curvature_penalty_strength, user_seeds=None):
    """
    Builds a face adjacency graph with curvature-aware edge weights.

    Args:
         mesh: trimesh object
         curvature_penalty_strength: float, strength of curvature penalty
         user_seeds: list, user-selected seed points or face indices
    Returns:
         sparse_matrix : scipy.sparse.csr_matrix, shape (N, N)
            Weighted adjacency matrix of the filtered face graph. N is the number of faces.
            Entry (i, j) contains the weight between face i and j,
            combining spatial distance and curvature penalty.
        face_centers : numpy.ndarray, shape (N, 3)
            Array of 3D centroids corresponding to the faces in the graph. Row index i of
            `sparse_matrix` maps directly to `face_centers[i]`.
    """
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

    # Penalties
    curvature_penalty = np.exp(curvature_penalty_strength * angle)
    spatial_penalty = 1 + (spatial_dist / avg_edge_length)**2

    weights = spatial_penalty + curvature_penalty

    row = adj[:, 0]
    col = adj[:, 1]
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    all_weights = np.concatenate([weights, weights]).astype(np.float64)

    # Graph building
    N = face_centers.shape[0]
    sparse_matrix = sparse.csr_matrix((all_weights, (all_row, all_col)),
                                      shape=(N, N), dtype=np.float64)

    return sparse_matrix, face_centers


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
    rng = np.random.default_rng(40)  # 42 is for debugging needs to be removed in final version
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
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx, directed=False, return_predecessors=False)

    # DEBUG--
    inf_count = np.sum(~np.isfinite(dist), axis=None)
    print(f"Distance matrix: {inf_count}/{dist.size} infinite entries ({100*inf_count/dist.size:.2f}%)")
    # --------

    winner = np.argmin(dist, axis=0)

    face_labels = {i: int(seed_idx[winner[i]]) for i in range(sparse_matrix.shape[0])}
    return face_labels


def smooth_segment_boundaries(mesh, face_labels, iterations=3):
    """
    Smooth segment boundaries by reassigning boundary faces based on neighbor majority voting.
    This reduces jagged edges by making boundaries follow more natural paths.
    
    Args:
        mesh: trimesh.Trimesh
        face_labels: dict, mapping face_i -> seed_j
        iterations: int, number of smoothing passes
    Returns:
        smoothed_labels: dict, updated face_labels with smoother boundaries
    """
    from collections import Counter
    
    adj = mesh.face_adjacency
    # Build face neighbor lookup
    neighbors = {i: [] for i in range(len(mesh.faces))}
    for f1, f2 in adj:
        neighbors[f1].append(f2)
        neighbors[f2].append(f1)
    
    labels = face_labels.copy()
    
    for _ in range(iterations):
        new_labels = labels.copy()
        for face_i, current_label in labels.items():
            neighbor_labels = [labels[n] for n in neighbors[face_i]]
            if not neighbor_labels:
                continue
            
            # Check if this is a boundary face (has neighbors with different labels)
            if all(nl == current_label for nl in neighbor_labels):
                continue
            
            # Count neighbor labels (include self with lower weight)
            label_counts = Counter(neighbor_labels)
            label_counts[current_label] = label_counts.get(current_label, 0) + 1  # Self-weight
            
            # Assign to majority label
            majority_label = label_counts.most_common(1)[0][0]
            new_labels[face_i] = majority_label
        
        labels = new_labels
    
    return labels


def cleanup_small_regions(mesh, face_labels, min_region_size=5):
    """
    Merge small isolated regions into their largest neighboring segment.
    This fixes small "breaches" where a few faces are incorrectly labeled.
    
    Args:
        mesh: trimesh.Trimesh
        face_labels: dict, mapping face_i -> seed_j
        min_region_size: int, minimum number of faces a connected region must have
    Returns:
        cleaned_labels: dict, updated face_labels with small regions merged
    """
    from collections import deque, Counter
    
    adj = mesh.face_adjacency
    # Build face neighbor lookup
    neighbors = {i: [] for i in range(len(mesh.faces))}
    for f1, f2 in adj:
        neighbors[f1].append(f2)
        neighbors[f2].append(f1)
    
    labels = face_labels.copy()
    
    # Find connected components within each segment
    visited = set()
    all_regions = []
    
    for start_face in range(len(mesh.faces)):
        if start_face in visited:
            continue
        
        # BFS to find connected region with same label
        region = []
        queue = deque([start_face])
        region_label = labels[start_face]
        
        while queue:
            face = queue.popleft()
            if face in visited:
                continue
            if labels[face] != region_label:
                continue
            
            visited.add(face)
            region.append(face)
            
            for neighbor in neighbors[face]:
                if neighbor not in visited and labels[neighbor] == region_label:
                    queue.append(neighbor)
        
        all_regions.append((region, region_label))
    
    # Merge small regions into neighboring segments
    for region, region_label in all_regions:
        if len(region) >= min_region_size:
            continue
        
        # Find all neighboring labels (from other segments)
        neighbor_labels = Counter()
        for face in region:
            for neighbor in neighbors[face]:
                if labels[neighbor] != region_label:
                    neighbor_labels[labels[neighbor]] += 1
        
        if neighbor_labels:
            # Merge into the most common neighboring segment
            new_label = neighbor_labels.most_common(1)[0][0]
            for face in region:
                labels[face] = new_label
    
    return labels


def smooth_boundaries_geodesic(mesh, face_labels, smoothing_iterations=10, reassign_distance=2):
    """
    Smooth segment boundaries by fitting smooth curves along boundary edges
    and re-assigning faces based on their distance to the smoothed boundary.
    
    This produces much smoother boundaries than simple majority voting by
    treating the boundary as a continuous curve rather than individual faces.
    
    Args:
        mesh: trimesh.Trimesh
        face_labels: dict, mapping face_i -> seed_j
        smoothing_iterations: int, number of Laplacian smoothing passes on boundary vertices
        reassign_distance: int, max rings of faces from boundary to consider for reassignment
    Returns:
        smoothed_labels: dict, updated face_labels with smoother boundaries
    """
    from collections import defaultdict, deque
    
    adj = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    
    # Build face neighbor lookup
    neighbors = defaultdict(set)
    for f1, f2 in adj:
        neighbors[f1].add(f2)
        neighbors[f2].add(f1)
    
    labels = face_labels.copy()
    
    # Step 1: Detect boundary edges (edges between faces with different labels)
    boundary_edges = []
    boundary_edge_faces = []  # Store which faces each boundary edge separates
    for i, (f1, f2) in enumerate(adj):
        if labels[f1] != labels[f2]:
            edge = tuple(adj_edges[i])
            boundary_edges.append(edge)
            boundary_edge_faces.append((f1, f2, labels[f1], labels[f2]))
    
    if not boundary_edges:
        return labels
    
    # Step 2: Build boundary vertex graph and chains
    vertex_edges = defaultdict(list)
    for i, (v1, v2) in enumerate(boundary_edges):
        vertex_edges[v1].append((i, v2))
        vertex_edges[v2].append((i, v1))
    
    # Find connected boundary chains
    visited_edges = set()
    boundary_chains = []
    
    for start_idx, (v1, v2) in enumerate(boundary_edges):
        if start_idx in visited_edges:
            continue
        
        # BFS to find connected chain of boundary edges
        chain_vertices = []
        chain_edges = []
        queue = deque([start_idx])
        
        while queue:
            edge_idx = queue.popleft()
            if edge_idx in visited_edges:
                continue
            visited_edges.add(edge_idx)
            chain_edges.append(edge_idx)
            
            ev1, ev2 = boundary_edges[edge_idx]
            if ev1 not in chain_vertices:
                chain_vertices.append(ev1)
            if ev2 not in chain_vertices:
                chain_vertices.append(ev2)
            
            # Add connected boundary edges
            for next_idx, _ in vertex_edges[ev1]:
                if next_idx not in visited_edges:
                    queue.append(next_idx)
            for next_idx, _ in vertex_edges[ev2]:
                if next_idx not in visited_edges:
                    queue.append(next_idx)
        
        if len(chain_vertices) >= 3:
            boundary_chains.append((chain_vertices, chain_edges))
    
    # Step 3: Laplacian smoothing of boundary vertices
    # Create smoothed positions for boundary vertices
    boundary_vertex_set = set()
    for chain_verts, _ in boundary_chains:
        boundary_vertex_set.update(chain_verts)
    
    # Build boundary vertex neighbors (only other boundary vertices)
    boundary_neighbors = defaultdict(set)
    for v1, v2 in boundary_edges:
        if v1 in boundary_vertex_set and v2 in boundary_vertex_set:
            boundary_neighbors[v1].add(v2)
            boundary_neighbors[v2].add(v1)
    
    # Smooth boundary vertex positions (virtual, for computing face reassignment)
    smoothed_positions = {v: mesh.vertices[v].copy() for v in boundary_vertex_set}
    
    for _ in range(smoothing_iterations):
        new_positions = {}
        for v in boundary_vertex_set:
            if boundary_neighbors[v]:
                neighbor_avg = np.mean([smoothed_positions[n] for n in boundary_neighbors[v]], axis=0)
                # Blend: 50% original, 50% neighbor average (controls smoothing strength)
                new_positions[v] = 0.5 * smoothed_positions[v] + 0.5 * neighbor_avg
            else:
                new_positions[v] = smoothed_positions[v]
        smoothed_positions = new_positions
    
    # Step 4: Re-assign boundary faces based on smoothed boundary
    # Find faces within N rings of boundary
    boundary_faces = set()
    for f1, f2, l1, l2 in boundary_edge_faces:
        boundary_faces.add(f1)
        boundary_faces.add(f2)
    
    # Expand to N rings
    faces_to_consider = set(boundary_faces)
    current_ring = set(boundary_faces)
    for _ in range(reassign_distance - 1):
        next_ring = set()
        for f in current_ring:
            for n in neighbors[f]:
                if n not in faces_to_consider:
                    next_ring.add(n)
        faces_to_consider.update(next_ring)
        current_ring = next_ring
    
    # For each face near boundary, compute distance to smoothed boundary curve
    # and assign to the segment whose smoothed boundary is farther (i.e., face is "inside")
    for face_i in faces_to_consider:
        face_center = mesh.triangles_center[face_i]
        current_label = labels[face_i]
        
        # Find nearby boundary edges and their smoothed midpoints
        nearby_boundary_info = []
        face_verts = set(mesh.faces[face_i])
        
        for edge_idx, (v1, v2) in enumerate(boundary_edges):
            if v1 in boundary_vertex_set and v2 in boundary_vertex_set:
                # Check if this edge is near this face
                if v1 in face_verts or v2 in face_verts:
                    # Compute smoothed edge midpoint
                    smoothed_mid = (smoothed_positions[v1] + smoothed_positions[v2]) / 2
                    original_mid = (mesh.vertices[v1] + mesh.vertices[v2]) / 2
                    
                    # Direction of smoothing
                    smooth_dir = smoothed_mid - original_mid
                    
                    # Vector from original edge to face center
                    to_face = face_center - original_mid
                    
                    # If face is in the direction of smoothing, it might need reassignment
                    f1, f2, l1, l2 = boundary_edge_faces[edge_idx]
                    nearby_boundary_info.append((l1, l2, smooth_dir, to_face, f1, f2))
        
        # Use voting based on smoothed boundary positions
        if nearby_boundary_info:
            label_scores = defaultdict(float)
            for l1, l2, smooth_dir, to_face, f1, f2 in nearby_boundary_info:
                # Determine which side of the smoothed boundary this face is on
                dot = np.dot(smooth_dir, to_face)
                if dot > 0:
                    # Face is in direction of smoothing - favor the label that was "pushed" this way
                    label_scores[l2] += 1
                else:
                    label_scores[l1] += 1
            
            if label_scores:
                # Include current label with some weight to avoid excessive changes
                label_scores[current_label] += 0.5
                best_label = max(label_scores, key=label_scores.get)
                labels[face_i] = best_label
    
    return labels




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

    # --- Post-processing: smooth boundaries and cleanup small regions ---
    face_labels = smooth_segment_boundaries(mesh, face_labels, iterations=8)
    print("Boundaries Smoothed (majority voting)")
    
    face_labels = cleanup_small_regions(mesh, face_labels, min_region_size=5)
    print("Small Regions Cleaned")

    # --- Feature line extraction and export ---
    feature_chains = extract_feature_lines(mesh)
    export_feature_lines_obj(mesh, feature_chains, os.path.join(args.output_dir, "feature_lines_experimental.obj"))
    print(f"Exported {len(feature_chains)} feature chains to OBJ.")

    export_segment(mesh, face_labels, seed_idx, args.output_dir)
    print("Segmentation Finished")

    print("Elapsed:", time.perf_counter() - t0)  # DEBUG

if __name__ == "__main__":
    main()