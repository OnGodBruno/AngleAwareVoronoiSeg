
import trimesh
import numpy as np
from scipy.sparse import csgraph
import scipy.sparse as sparse
import os
import argparse
import automatic_seed_selection as autoseed

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
    spatial_penalty = 1 + (spatial_dist / avg_edge_length )**2

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



def select_seeds(mesh, adj_csr, n_seeds):
    # Compute boundary likelihood
    be, _ = autoseed.compute_boundary_likelihood(mesh)
    
    # Compute boundary set (this was missing!)
    boundary_edges, boundary_faces = autoseed.compute_boundary_set(be, mesh)
    
    # Now compute distance to boundary with all required arguments
    dtb = autoseed.compute_distance_to_boundary(mesh, be, boundary_faces)
    
    # Compute candidates with NMS
    candidates, r_nms, DM_est, T_min = autoseed.compute_dtb_candidates_nms(mesh, be, dtb)
    
    if candidates is None or len(candidates) == 0:
        candidates = np.arange(mesh.faces.shape[0], dtype=np.int32)
    
    # Compute candidate quality (note the typo fix here)
    q_cand, _ = autoseed.compute_candidate_quality(mesh, dtb, candidates)
    
    # Select seeds with graph diversity
    seed_idx = autoseed.select_seeds_graph_diverse(
        adj_csr=adj_csr,
        candidates=candidates,
        q_cand=q_cand,
        n_seeds=int(n_seeds),
        beta=2.0,
        limit=None,
        initial_seeds=None
    )
    
    return np.asarray(seed_idx, dtype=np.int32)

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

    seed_idx = select_seeds(mesh, sparse_matrix, args.n_seeds)
    print("Seeds selected")
    print(f"Using seed face indices: {seed_idx}")

    face_labels = segment_mesh(sparse_matrix, seed_idx)
    print("Mesh Segmented")

    export_segment(mesh, face_labels, seed_idx, args.output_dir)
    print("Segmentation Finished")

    print("Elapsed:", time.perf_counter() - t0)  # DEBUG

if __name__ == "__main__":
    main()
