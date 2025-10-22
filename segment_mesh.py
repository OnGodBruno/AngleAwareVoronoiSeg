
import trimesh
import numpy as np
from scipy.sparse import csgraph
import scipy.sparse as sparse
import os
import argparse
from collections import deque

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


import numpy as np
from scipy.sparse import csr_matrix, csgraph

import numpy as np
from scipy.sparse import csr_matrix, csgraph

def find_valley_edge_components(mesh, edge_mask):
    """
    Finde Komponenten im Kanten-Subgraph (Nodes = valley edges).
    Rückgabe:
      components: list[np.ndarray]  # Edge-Indizes (im face_adjacency)
      endpoints:  list[np.ndarray]  # Submengen pro Komponente (Grad <= 1)
    """
    FA = mesh.face_adjacency
    sub_idx = np.flatnonzero(edge_mask)
    if sub_idx.size == 0:
        return [], []

    F = FA[sub_idx]
    f0, f1 = F[:, 0], F[:, 1]

    faces = np.concatenate([f0, f1])
    edges = np.concatenate([np.arange(sub_idx.size)] * 2)
    order = np.argsort(faces, kind="mergesort")
    faces_s, edges_s = faces[order], edges[order]

    # group borders
    bounds = np.flatnonzero(np.diff(faces_s)) + 1
    starts = np.concatenate(([0], bounds))
    stops  = np.concatenate((bounds, [faces_s.size]))

    # connect valley edges sharing a face
    rows, cols = [], []
    for a, b in zip(starts, stops):
        grp = edges_s[a:b]
        if grp.size <= 1:
            continue
        ii = np.repeat(grp, grp.size - 1) # to connect every edge with every other edge in the group
        jj = np.concatenate([np.delete(grp, i) for i in range(grp.size)])
        rows.append(ii)
        cols.append(jj)
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.size, dtype=bool)
        A = csr_matrix((data, (rows, cols)), shape=(sub_idx.size, sub_idx.size))
    else:
        A = csr_matrix((sub_idx.size, sub_idx.size), dtype=bool)

    # components
    _, labels = csgraph.connected_components(A, directed=False, return_labels=True)

    components = []
    endpoints = []
    for lab in np.unique(labels):
        comp_edges = sub_idx[labels == lab]
        components.append(comp_edges)

        # deg[i] = number of neighbors of edge i
        deg = np.array(A[labels == lab][:, labels == lab].sum(1)).ravel()
        endpoints.append(comp_edges[deg <= 1])

    return components, endpoints

    

def find_valleys(mesh, normal_smoothing: bool = True) -> np.ndarray:
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
    
    theta_deg = np.degrees(np.arctan2(sin, cos))
    theta_min = 5.0
    cap = 160.0
    neg = np.where(theta_deg < -theta_min, -theta_deg, 0.0)
    valley_valid = np.clip(neg, 0.0, cap) / cap
    valley_scores = np.zeros(len(adj), dtype=float)
    valley_scores[mask_ok] = valley_valid
    
    
    def build_edge_neighbors(mesh):
        adj = mesh.face_adjacency
        E = len(adj)
        ev = mesh.face_adjacency_edges              # (E,2) Vertexpaare
        # Vertex-Nachbarn
        v2e = {}
        for ei,(va,vb) in enumerate(ev):
            v2e.setdefault(va, []).append(ei)
            v2e.setdefault(vb, []).append(ei)
        nbr = [set() for _ in range(E)]
        for lst in v2e.values():
            for i in lst:
                for j in lst:
                    if i!=j: nbr[i].add(j)
        # Face-Nachbarn
        f2e = {}
        for ei,(f0,f1) in enumerate(adj):
            f2e.setdefault(f0, []).append(ei)
            f2e.setdefault(f1, []).append(ei)
        for lst in f2e.values():
            for i in lst:
                for j in lst:
                    if i!=j: nbr[i].add(j)
        return [np.fromiter(s, dtype=int) if s else np.empty(0,dtype=int) for s in nbr]

    def hysteresis(valley_scores, edge_neighbors, T_low, T_high):
        strong = valley_scores >= T_high
        weak = valley_scores >= T_low
        mask = np.zeros_like(valley_scores, dtype=bool)
        dq = deque(np.flatnonzero(strong))
        while dq:
            e = dq.pop()
            if mask[e]: 
                continue
            mask[e] = True
            for nb in edge_neighbors[e]:
                if weak[nb] and not mask[nb]:
                    dq.append(nb)
        return mask
    
    edge_neighbors = build_edge_neighbors(mesh)
    valley_mask = hysteresis(valley_scores, edge_neighbors, T_low=0.12, T_high=0.32)
    return valley_scores, valley_mask

    
   

def get_valley_faces(mesh):
    """Find faces that have at least one valley edge."""
    valley_scores, valley_mask = find_valleys(mesh, normal_smoothing=False)
    print(f"Found {len(valley_scores[valley_scores > 0.0])} valley edges out of {len(valley_scores)} total edges.") # DEBUG
    
    # --- DEBUG: Valley-Score-Statistik ---
    print(f"ValleyScores: min={valley_scores.min():.6f}, max={valley_scores.max():.6f}, mean={valley_scores.mean():.6f}, >0 count={valley_mask.sum()}/{valley_scores.size}")
    if valley_mask.any():
        print(f"ValleyScores (>0): min={valley_scores[valley_mask].min():.6f}, max={valley_scores[valley_mask].max():.6f}, mean={valley_scores[valley_mask].mean():.6f}")
    else:
        print("ValleyScores (>0): keine positiven Werte")
    # --------------------------------------
    
    components, endpoints = find_valley_edge_components(mesh, valley_mask)
    print(f"Found {len(components)} connected components of valley edges.") # DEBUG
    
    #-------------DEBUG-----------------
    endpoints = find_valley_edge_components(mesh, valley_mask)[1]

    ep_counts   = np.array([len(ep) for ep in endpoints], dtype=int)
    comp_counts = np.array([len(c)  for c  in components], dtype=int)

    if ep_counts.size:
        print(f"Endpoints per group: mean={ep_counts.mean():.2f}, min={ep_counts.min()}, max={ep_counts.max()}")
    else:
        print("No endpoints found.")

    if comp_counts.size:
        print(f"Edges per group:    mean={comp_counts.mean():.2f}, min={comp_counts.min()}, max={comp_counts.max()}")
    else:
        print("No edge components found.")
    #-----------------------------------
    
    
    valley_face_pairs = mesh.face_adjacency[valley_mask]
    
    valley_faces = np.unique(valley_face_pairs.reshape(-1))
    
    # Create a mask for all faces
    valley_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    valley_face_mask[valley_faces] = True
    
    # Get scores for visualization (max valley score for each face)
    face_scores = np.zeros(len(mesh.faces), dtype=float)
    for i, (f1, f2) in enumerate(mesh.face_adjacency):
        s = valley_scores[i]
        if s > 0.0:
            if s > face_scores[f1]:
                face_scores[f1] = s
            if s > face_scores[f2]:
                face_scores[f2] = s
    
    return valley_face_mask, face_scores
  
    
def build_adjacency_graph(mesh, curvature_penalty_strength, user_seeds=None, valley_barrier_alpha: float = 20.0):
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
    
    base_weights = spatial_penalty + curvature_penalty

    
    valley, _ = find_valleys(mesh, normal_smoothing=True)

    eps = 1e-6
    
    tau = float(valley_barrier_alpha) * float(np.median(base_weights)) if base_weights.size else float(valley_barrier_alpha)
    barrier = tau * (valley / (1.0 - valley + eps))*1000


    # weights = base_weights + barrier
    weights = 1 + barrier
    

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
