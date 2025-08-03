

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


def build_adjacency_graph(mesh, curvature_penalty_strength, max_normal_angle=np.radians(20)):
    """
    Build a face adjacency graph with curvature-aware edge weights.
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
    weights = spatial_penalty * curvature_penalty

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

def pick_first_seed(face_coords,  pool_size=32):
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
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx,
                            directed=False, return_predecessors=False)

    # Generate offsets for each seed so that the earliest seed wins on exact distance matches
    eps = np.linspace(0.0, 1e-9, len(seed_idx), endpoint=False)[:, None]
    winner = (dist + eps).argmin(axis=0)

    # Create face labels mapping matrix indices to seed indices
    face_labels = {i: seed_idx[winner[i]]
                   for i in range(sparse_matrix.shape[0])
                   if np.isfinite(dist[winner[i], i])}

    print("mesh segmented")  # DEBUG
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
