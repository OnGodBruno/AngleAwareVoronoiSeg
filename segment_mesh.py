
import trimesh
import numpy as np
import os
import argparse
from scipy.sparse import csr_matrix, csgraph
import connect_components as cc

import time # Debug
    

def load_and_clean_mesh(mesh_path):
    """
    Load and clean a 3D mesh.
    """
    mesh = trimesh.load(mesh_path, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fix_normals()
    return mesh


def get_valley_faces(mesh, valley_threshold=0.1, connection_runs=5, curvature_penalty_strength=100.0):
    """Find faces that have at least one valley edge.
    
    Args:
        mesh: The mesh object
        valley_threshold: Minimum valley score to consider an edge as a valley (currently not used, 
                         connect_components uses its own threshold internally)
        connection_runs: Number of iterations for component connection
        curvature_penalty_strength: Penalty strength for the face graph used in connection
    
    Returns:
        valley_face_mask: Boolean mask indicating which faces have valley edges
        face_scores: Maximum valley score for each face
    """
    
    # Connect components - this does EVERYTHING: find_valleys + connection
    print(f"\nConnecting components (runs={connection_runs})...")
    valley_mask_new = cc.connect_components(mesh, runs=connection_runs, curvature_penalty_strength=curvature_penalty_strength, valley_threshold=valley_threshold)
    
    valley_scores, _ = cc.find_valleys(mesh, normal_smoothing=True, valley_threshold=valley_threshold)
    
    print(f"\nFinal valley edges: {valley_mask_new.sum()}")

    # Use the NEW mask from connect_components
    valley_face_pairs = mesh.face_adjacency[valley_mask_new]
    valley_faces = np.unique(valley_face_pairs.reshape(-1))

    valley_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    valley_face_mask[valley_faces] = True

    # Max score per face, but only from KEPT edges
    face_scores = np.zeros(len(mesh.faces), dtype=float)
    for i, (f1, f2) in enumerate(mesh.face_adjacency):
        if not valley_mask_new[i]:
            continue
        s = valley_scores[i]
        if s > face_scores[f1]:
            face_scores[f1] = s
        if s > face_scores[f2]:
            face_scores[f2] = s

    return valley_face_mask, face_scores

  
    
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
    
    # DEBUG VALLEY TESTING--------------------------
    valley_mask = cc.connect_components(mesh, runs=5)
    
    weights = np.where(valley_mask, np.inf, 1.0)
    #-----------------------------------------------

    # Penalties
    #curvature_penalty = np.exp(curvature_penalty_strength * angle)
    #spatial_penalty = 1 + (spatial_dist / avg_edge_length )**2

    #weights = spatial_penalty + curvature_penalty

    row = adj[:, 0]
    col = adj[:, 1]
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    all_weights = np.concatenate([weights, weights]).astype(np.float64)

    # Graph building
    N = face_centers.shape[0]
    sparse_matrix = csr_matrix((all_weights, (all_row, all_col)),
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
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx, directed=False, return_predecessors=False)

    # DEBUG--
    inf_count = np.sum(~np.isfinite(dist), axis=None)
    print(print(f"Distance matrix: {inf_count}/{dist.size} infinite entries ({100 * inf_count /dist.size:.2f}%)"))
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
