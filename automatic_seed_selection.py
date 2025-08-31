import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph

def compute_boundary_likelihood(
    mesh,
    *,
    theta0_rad: float = np.deg2rad(35.0),    # angle threshold in degrees for "sharp"
    gamma: float = 8.0,          # sigmoid steepness
    delta: float = 2.0,          # bias for concave edges
    tau_b: float = 0.6,          # threshold for marking an edge as boundary
    boundary_value: float = 0.95,# likelihood for open boundary edges
    nonmanifold_value: float = 1.0 # likelihood for non-manifold edges
):
    """
    Compute boundary likelihood b_e ∈ (0,1) for each unique edge in a trimesh mesh.

    Uses:
      - mesh.edges_unique: (E,2) array of unique undirected edges (sorted vertex IDs)
      - mesh.edges_unique_inverse: (3*F,) map from each face-edge to a unique edge index
      - mesh.face_adjacency: (M,2) pairs of adjacent face indices
      - mesh.face_adjacency_edges: (M,2) vertex pairs of the shared edge for each adjacency
      - mesh.face_normals: (F,3) normals of faces
      - mesh.vertices: (V,3) vertex positions

    Returns:
      - b_e: (E,) array of float32, boundary likelihood per unique edge
      - boundary_edges: indices of edges where b_e >= tau_b
    """

    V = np.asarray(mesh.vertices, dtype=np.float32)
    fn = np.asarray(mesh.face_normals, dtype=np.float32)
    edges_u = np.asarray(mesh.edges_unique, dtype=np.int32)           # (E,2)
    inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int32)       # (3*F,)

    E = edges_u.shape[0]

    # Compute how many times each unique edge is referenced (1=boundary, 2=internal, >2=nonmanifold)
    counts = np.bincount(inv, minlength=E)

    b_e = np.zeros(E, dtype=np.float32)

    # Case A: boundary edges (count==1)
    b_e[counts == 1] = boundary_value

    # Case B: non-manifold edges (count>2)
    b_e[counts > 2] = nonmanifold_value

    # Case C: internal edges (count==2)
    if np.any(counts == 2):
        fa = np.asarray(mesh.face_adjacency, dtype=np.int64)           # (M,2)
        fa_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int32) # (M,2)

        if fa.shape[0] > 0:
            # sort vertex IDs so they match edges_unique convention
            fae_sorted = np.sort(fa_edges, axis=1)

            # Build mapping (v_min,v_max) -> unique edge index
            # Do this by structured array trick
            def to_struct(arr):
                return arr.view([('v0', np.int32), ('v1', np.int32)]).reshape(-1)

            eu_struct = to_struct(edges_u)
            fa_struct = to_struct(fae_sorted)

            order = np.argsort(eu_struct, kind='mergesort')
            eu_sorted = eu_struct[order]
            pos = np.searchsorted(eu_sorted, fa_struct)
            valid = (pos >= 0) & (pos < eu_sorted.shape[0])
            match = np.zeros_like(pos, dtype=bool)
            match[valid] = (eu_sorted[pos[valid]] == fa_struct[valid])
            idx_fa = np.where(match)[0]
            eu_idx = order[pos[idx_fa]]  # indices in edges_unique
            faces_pair = fa[idx_fa]      # (K,2)

            # Edge directions (from smaller to larger vertex index)
            vA = edges_u[eu_idx, 0]
            vB = edges_u[eu_idx, 1]
            e_vec = V[vB] - V[vA]
            e_len = np.linalg.norm(e_vec, axis=1)
            eps = 1e-12
            e_hat = e_vec / np.maximum(e_len[:, None], eps) # edge vector

            # Normals of the two adjacent faces
            n1 = fn[faces_pair[:, 0]]
            n2 = fn[faces_pair[:, 1]]

            # Dihedral angle magnitude
            angle = np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))

            # Concave vs convex test:
            # (n1 x n2) · e_hat > 0 -> convex
            # (n1 x n2) · e_hat < 0 -> concave
            cross = np.cross(n1, n2)
            sign_term = np.sum(cross * e_hat, axis=1)
            is_concave = (sign_term < 0.0).astype(np.float32)

            # Hinge with konkav-Bonus
            x = gamma * (angle - theta0_rad) + delta * is_concave 

            # Threshold T: gamma * 20° + delta
            T = gamma * (np.deg2rad(20.0)) + delta  
            
            p = 2.0 

            # Exponent p = 2
            be_internal = np.where(x <= 0.0, 0.0, np.where(x >= T, 1.0, (x / T) ** p)).astype(np.float32)

            # Assign values back to b_e for these internal edges
            b_e[eu_idx] = be_internal

    # Indices of edges considered as boundary (above threshold)
    boundary_edges = np.where(b_e >= tau_b)[0].astype(np.int64)

    return b_e, boundary_edges


def compute_boundary_set(b_e: np.ndarray,
                         mesh,
                         tau_b: float = 0.6):
    """
    Step 2: secure boundary set B and associated boundary faces Q.

    Inputs:
      - b_e: (E,) boundary likelihood per unique primal edge
      - mesh: trimesh-like with .edges_unique_inverse (3*F,)

    Returns:
      - boundary_edges: (|B|,) int64 indices in edges_unique, where b_e >= tau_b
      - boundary_faces: (|Q|,) int64 face indices that contain at least one boundary edge
    """
    # 2.2: B = { e : b_e >= tau_b }
    boundary_edges = np.where(b_e >= tau_b)[0].astype(np.int64)

    # 2.3: Q = { f ∈ F : ∃ e ⊂ f with e ∈ B }
    inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)  # (3*F,)
    faces = np.asarray(mesh.faces, dtype=np.int64)   # (F, 3) bei Dreiecksmesh
    F = faces.shape[0]

    inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64).ravel()

    if inv.size != 3 * F:
        raise ValueError(
            f"edges_unique_inverse has {inv.size} Entries, expected 3*F = {3*F}. "
        )

    face_edge_uidx = inv.reshape(F, 3)

    edge_flags = np.zeros(b_e.shape[0], dtype=bool)
    edge_flags[boundary_edges] = True

    has_boundary = np.any(edge_flags[face_edge_uidx], axis=1)    # (F,)
    boundary_faces = np.where(has_boundary)[0].astype(np.int64)

    return boundary_edges, boundary_faces


def compute_distance_to_boundary(
    mesh,
    b_e: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    eps: float = 1e-3,
    dtype_float=np.float32,
    dtype_index=np.int32,
) -> np.ndarray:
    """
    Step 3: Distance-to-Boundary (DTB) per Face.

    For each face f, DTB(f) = shortest path distance (in the face dual graph)
    to any boundary face in boundary_faces is computed.
    Dual edge costs w = eps + (1 - b_e[shared_edge_uidx]).

    Uses scipy.sparse.csgraph.dijkstra (one run from a supersource).

    Args:
        mesh: trimesh-like object with attributes:
              - faces (F,3)
              - face_adjacency (M,2)
              - face_adjacency_edges (M,2)  (vertex IDs of the shared edge respectively)
              - edges_unique (E,2)          (sorted vertex IDs per edge)
              - edges_unique_inverse (3*F,) Mapping face-edges -> unique edge index
        b_e: (E,) float array, boundary likelihood per unique edge (from Step 1).
        boundary_faces: (|Q|,) int array, face indices that carry at least one boundary edge (from Step 2).
        eps: Small offset so that no 0-edges arise.
        dtype_float: Float data type for weights (default float32).
        dtype_index: Integer data type for indices (default int32).

    Returns:
        dtb: (F,) dtype_float, Distance-to-Boundary per face (>=0).
    """
    # --- Inputs from the mesh ---
    faces = np.asarray(mesh.faces, dtype=dtype_index)
    F = faces.shape[0]

    fa = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=dtype_index)), dtype=dtype_index)
    fa_edges = np.asarray(
        getattr(mesh, "face_adjacency_edges", np.empty((0, 2), dtype=dtype_index)),
        dtype=dtype_index,
    )

    edges_u = np.asarray(mesh.edges_unique, dtype=dtype_index)  # (E,2)
    E = edges_u.shape[0]

    # --- Mapping: face_adjacency_edges -> edges_unique indices ---
    # We need for each adjacency (f0,f1) the unique-edge-index of the shared primal edge.
    def to_struct(arr: np.ndarray) -> np.ndarray:
        return arr.view([('v0', dtype_index), ('v1', dtype_index)]).reshape(-1)

    if fa.shape[0] > 0:
        fae_sorted = np.sort(fa_edges, axis=1)
        eu_struct = to_struct(edges_u)
        fa_struct = to_struct(fae_sorted)

        order = np.argsort(eu_struct, kind="mergesort")
        eu_sorted = eu_struct[order]

        pos = np.searchsorted(eu_sorted, fa_struct)
        valid = (pos >= 0) & (pos < eu_sorted.shape[0])
        match = np.zeros_like(pos, dtype=bool)
        match[valid] = (eu_sorted[pos[valid]] == fa_struct[valid])

        if not np.all(match):
            # Filter to the valid adjacencies (should rarely be necessary)
            keep = np.where(match)[0]
            fa = fa[keep]
            fae_sorted = fae_sorted[keep]
            pos = pos[keep]

        eu_idx_for_fa = order[pos]  # (M_valid,)
        M = fa.shape[0]
    else:
        eu_idx_for_fa = np.empty((0,), dtype=dtype_index)
        M = 0

    # --- Edge costs for dual graph: w = eps + (1 - b_e) ---
    if M > 0:
        be_vals = np.asarray(b_e, dtype=dtype_float)[eu_idx_for_fa]
        w = (eps + (1.0 - be_vals)).astype(dtype_float, copy=False)  # (M,)
    else:
        w = np.empty((0,), dtype=dtype_float)

    ss = np.array([F], dtype=dtype_index)[0]

    Q = np.asarray(boundary_faces, dtype=dtype_index).ravel()
    Q = Q[(Q >= 0) & (Q < F)]
    
    # Fallback: if Q is empty, select faces with the highest max(b_e) over their three edges (top 1% or at least 1)
    if Q.size == 0:
        inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64).reshape(F, 3)
        face_max_be = np.max(np.asarray(b_e, dtype=dtype_float)[inv], axis=1)
        thresh = np.quantile(face_max_be, 0.99) if F > 100 else np.max(face_max_be)
        Q = np.where(face_max_be >= thresh)[0].astype(dtype_index)
        if Q.size == 0:
            Q = np.array([int(np.argmax(face_max_be))], dtype=dtype_index)

    # Number of non-zero-entries: 2*M (bidirectional face-face-edges) + 2*|Q| (supersource-edges)
    nnz = 2 * M + 2 * Q.size
    rows = np.empty(nnz, dtype=dtype_index)
    cols = np.empty(nnz, dtype=dtype_index)
    data = np.empty(nnz, dtype=dtype_float)

    if M > 0:
        rows[0 : 2 * M : 2] = fa[:, 0]
        cols[0 : 2 * M : 2] = fa[:, 1]
        data[0 : 2 * M : 2] = w

        rows[1 : 2 * M : 2] = fa[:, 1]
        cols[1 : 2 * M : 2] = fa[:, 0]
        data[1 : 2 * M : 2] = w

    # Supersource edges (ss <-> Q) with cost 0
    start = 2 * M
    end = start + Q.size
    rows[start:end] = ss
    cols[start:end] = Q
    data[start:end] = 0.0

    rows[end : end + Q.size] = Q
    cols[end : end + Q.size] = ss
    data[end : end + Q.size] = 0.0

    G = csr_matrix((data, (rows, cols)), shape=(F + 1, F + 1))

    dist = csgraph.dijkstra(G, directed=False, indices=ss, unweighted=False, return_predecessors=False)
    dtb = np.asarray(dist[:-1], dtype=dtype_float)  # drop supersource

    # Fallback for isolated components without Q (dist == inf)
    if not np.all(np.isfinite(dtb)):
        # Find components in the face dual graph (without supersource)
        A = G[:-1, :-1]
        n_comp, comp_labels = csgraph.connected_components(A, directed=False)
        inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64).reshape(F, 3)
        face_max_be = np.max(np.asarray(b_e, dtype=dtype_float)[inv], axis=1)

        # For each component without seeds: choose 1 face with maximum face_max_be as additional Q
        Q_extra = []
        for c in range(n_comp):
            comp_faces = np.where(comp_labels == c)[0]
            if comp_faces.size == 0:
                continue
            if np.isfinite(dtb[comp_faces]).any():
                continue  # reachable
            f_star = comp_faces[np.argmax(face_max_be[comp_faces])]
            Q_extra.append(f_star)

        if Q_extra:
            Q_extra = np.asarray(Q_extra, dtype=dtype_index)
            rows2 = np.concatenate([np.full(Q_extra.size, ss, dtype=dtype_index), Q_extra])
            cols2 = np.concatenate([Q_extra, np.full(Q_extra.size, ss, dtype=dtype_index)])
            data2 = np.zeros(rows2.size, dtype=dtype_float)
            G = G + csr_matrix((data2, (rows2, cols2)), shape=G.shape)
            dist = csgraph.dijkstra(G, directed=False, indices=ss, unweighted=False, return_predecessors=False)
            dtb = np.asarray(dist[:-1], dtype=dtype_float)
            # If infinities still remain (pathological cases), set to 0
            dtb[~np.isfinite(dtb)] = dtype_float(0.0)

    return dtb


def compute_dtb_candidates_nms(
    mesh,
    b_e: np.ndarray,
    dtb: np.ndarray,
    *,
    rho: float = 0.05,           # r_NMS = rho * DM
    T_mode: str = "median",      # "median" | "quantile" | "value"
    T_quantile: float = 0.50,    # if T_mode=="quantile"
    T_value: float = None,       # if T_mode=="value"
    eps: float = 1e-3,           # as in Step 3
    dtype_float=np.float32,
    dtype_index=np.int32,
    max_candidates: int = None,  # optional: hard limit for number of accepted candidates
):
    """
    Step 4: Candidates (local DTB maxima) with NMS in the graph.

    - Coarse filter: keep faces with DTB >= T_min (default: median).
    - Estimate geodesic diameter DM in face dual graph (weights w = eps + (1 - b_e)).
    - NMS radius r_NMS = rho * DM.
    - Sort candidates descending by DTB; greedily select and suppress all
      candidates within graph distance <= r_NMS (Dijkstra with limit).

    Args:
        mesh: trimesh-like with:
              faces, face_adjacency, face_adjacency_edges, edges_unique, edges_unique_inverse
        b_e: (E,) Boundary likelihoods (from Step 1)
        dtb: (F,) Distance-to-Boundary per face (from Step 3)
        rho: Radius factor for NMS relative to DM
        T_mode: Threshold mode for coarse filter ("median" | "quantile" | "value")
        T_quantile: Quantile, e.g. 0.5 (median), 0.7 (70%-quantile), ...
        T_value: fixed threshold value if T_mode == "value"
        eps: small offset in dual edge costs
        dtype_float, dtype_index: data types
        max_candidates: if set, accept at most this many candidates

    Returns:
        candidates: (C,) int32 face indices of selected local maxima (descending by DTB)
        r_nms: used NMS radius (float)
        DM_est: estimated geodesic diameter in dual graph (float)
        T_min: used DTB threshold (float)
    """
    # ---------- Helper function: Face dual graph CSR (weights w = eps + (1 - b_e)) ----------
    def _build_face_dual_csr(mesh, b_e, eps, dtype_float, dtype_index):
        fa = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=dtype_index)), dtype=dtype_index)
        fa_edges = np.asarray(
            getattr(mesh, "face_adjacency_edges", np.empty((0, 2), dtype=dtype_index)),
            dtype=dtype_index,
        )
        faces = np.asarray(mesh.faces, dtype=dtype_index)
        F = faces.shape[0]
        edges_u = np.asarray(mesh.edges_unique, dtype=dtype_index)

        # Map face_adjacency_edges -> edges_unique index
        def to_struct(arr: np.ndarray) -> np.ndarray:
            return arr.view([('v0', dtype_index), ('v1', dtype_index)]).reshape(-1)

        if fa.shape[0] == 0:
            return csr_matrix((F, F), dtype=dtype_float)

        fae_sorted = np.sort(fa_edges, axis=1)
        eu_struct = to_struct(edges_u)
        fa_struct = to_struct(fae_sorted)

        order = np.argsort(eu_struct, kind="mergesort")
        eu_sorted = eu_struct[order]
        pos = np.searchsorted(eu_sorted, fa_struct)

        valid = (pos >= 0) & (pos < eu_sorted.shape[0])
        match = np.zeros_like(pos, dtype=bool)
        match[valid] = (eu_sorted[pos[valid]] == fa_struct[valid])

        if not np.all(match):
            keep = np.where(match)[0]
            fa = fa[keep]
            pos = pos[keep]

        eu_idx_for_fa = order[pos]  # (M_valid,)
        M = fa.shape[0]

        # Edge costs
        w = (eps + (1.0 - np.asarray(b_e, dtype=dtype_float)[eu_idx_for_fa])).astype(dtype_float, copy=False)

        # CSR (bidirectional)
        rows = np.empty(2 * M, dtype=dtype_index)
        cols = np.empty(2 * M, dtype=dtype_index)
        data = np.empty(2 * M, dtype=dtype_float)

        rows[0::2] = fa[:, 0]
        cols[0::2] = fa[:, 1]
        data[0::2] = w

        rows[1::2] = fa[:, 1]
        cols[1::2] = fa[:, 0]
        data[1::2] = w

        A = csr_matrix((data, (rows, cols)), shape=(F, F))
        return A

    # ---------- 1) Build CSR ----------
    A = _build_face_dual_csr(mesh, b_e, eps, dtype_float, dtype_index)
    F = A.shape[0]

    # ---------- 2) Determine T_min ----------
    dtb = np.asarray(dtb, dtype=dtype_float).copy()
    finite = np.isfinite(dtb)
    if not finite.any():
        # Pathological case: nothing reachable -> everything zero
        dtb = np.zeros_like(dtb)
        finite[:] = True

    if T_mode == "value" and T_value is not None:
        T_min = dtype_float(T_value)
    elif T_mode == "quantile":
        q = np.clip(float(T_quantile), 0.0, 1.0)
        T_min = dtype_float(np.quantile(dtb[finite], q))
    else:  # "median" default
        T_min = dtype_float(np.median(dtb[finite]))

    cand_mask = (dtb >= T_min) & finite
    cand_idx = np.where(cand_mask)[0].astype(dtype_index)

    if cand_idx.size == 0:
        # no candidates above threshold -> take global maximum
        f_star = int(np.argmax(dtb))
        return np.array([f_star], dtype=dtype_index), dtype_float(0.0), dtype_float(0.0), float(dtb[f_star])

    # ---------- 3) Diameter estimation DM (2x farthest-point heuristic) ----------
    # Start node: highest DTB (robust)
    s0 = int(cand_idx[np.argmax(dtb[cand_idx])])
    d0 = csgraph.dijkstra(A, directed=False, indices=s0, unweighted=False)
    # If graph is disconnected, take farthest FINITE target
    if np.isfinite(d0).any():
        a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    else:
        a = s0
    d1 = csgraph.dijkstra(A, directed=False, indices=a, unweighted=False)
    DM_est = dtype_float(np.max(np.where(np.isfinite(d1), d1, 0.0)))

    # Robustness fallback
    if not np.isfinite(DM_est) or DM_est <= 0:
        DM_est = dtype_float(np.max(dtb[finite]) - np.min(dtb[finite]) + eps)

    r_nms = dtype_float(max(eps * 5.0, rho * float(DM_est)))

    # ---------- 4) NMS: sort candidates, greedily select ----------
    order = np.argsort(-dtb[cand_idx], kind="mergesort")
    cand_idx_sorted = cand_idx[order]

    accepted = []
    suppressed = np.zeros(cand_idx_sorted.size, dtype=bool)

    # Fast lookup: Face-ID -> position in sorted candidate array
    pos_of_face = -np.ones(F, dtype=dtype_index)
    pos_of_face[cand_idx_sorted] = np.arange(cand_idx_sorted.size, dtype=dtype_index)

    for k, f in enumerate(cand_idx_sorted):
        if suppressed[k]:
            continue
        accepted.append(int(f))

        if max_candidates is not None and len(accepted) >= int(max_candidates):
            break

        # Truncated Dijkstra from f, only up to r_nms
        dist = csgraph.dijkstra(A, directed=False, indices=int(f), unweighted=False, limit=float(r_nms))
        # Suppress all candidates within radius
        hit_mask = np.isfinite(dist)
        # Candidates that were reached and lie within r_nms:
        reached_pos = pos_of_face[np.where(hit_mask & (dist <= r_nms))[0]]
        reached_pos = reached_pos[reached_pos >= 0]
        suppressed[reached_pos] = True
        # release the just accepted one again (it should naturally remain)
        suppressed[k] = False

    candidates = np.array(accepted, dtype=dtype_index)

    # Safety: return sorted by DTB
    if candidates.size > 1:
        candidates = candidates[np.argsort(-dtb[candidates], kind="mergesort")]

    return candidates, float(r_nms), float(DM_est), float(T_min)
  

def compute_candidate_quality(
    mesh,
    dtb: np.ndarray,
    candidates: np.ndarray,
    *,
    # Normalization of DTB to [0,1]
    norm_mode: str = "quantile",   # "minmax" | "quantile"
    q_low: float = 0.05,           # lower quantile for robust scaling
    q_high: float = 0.95,          # upper quantile for robust scaling
    # Optional area prior (light additive mix)
    area_mix: float = 0.0,         # 0.0 disables area; e.g. 0.1 adds a small area prior
    area_power: float = 0.5,       # compress area dynamic range with a power (<1.0)
    area_norm_mode: str = "quantile",
    area_q_low: float = 0.05,
    area_q_high: float = 0.95,
    # Dtypes
    dtype_float=np.float32,
    dtype_index=np.int32,
):
    """
    Step 5: Compute a quality score q_i ∈ [0,1] per candidate face.

    Baseline quality is a normalized DTB value. Optionally, a small face-area prior
    can be mixed in additively to bias seeds slightly toward larger (or smaller)
    regions depending on your downstream needs.

    Args:
        mesh: trimesh-like object; only needed if area_mix > 0 to get per-face areas.
        dtb: (F,) Distance-to-Boundary per face (from Step 3).
        candidates: (C,) face indices (from Step 4 NMS).
        norm_mode: "minmax" uses global min/max; "quantile" uses robust [q_low, q_high].
        q_low, q_high: quantiles for robust DTB scaling to [0,1].
        area_mix: 0..1 weight to blend a normalized area prior; 0 disables area prior.
        area_power: gamma-like compression for area prior (<1 flattens extremes).
        area_norm_mode, area_q_low, area_q_high: normalization settings for area prior.
        dtype_float, dtype_index: output and index dtypes.

    Returns:
        q_cand: (C,) float32 scores in [0,1] matching the order of `candidates`.
        meta: dict with scaling metadata (useful for debugging/repro):
              {
                "dtb_min": ..., "dtb_max": ...,
                "dtb_q_low": ..., "dtb_q_high": ...,
                "area_used": bool,
                "area_min": ..., "area_max": ...,
                "area_q_low": ..., "area_q_high": ...
              }
    """
    # ---- Validate and prepare inputs ----
    dtb = np.asarray(dtb, dtype=dtype_float)
    C = np.asarray(candidates, dtype=dtype_index).ravel()
    F = dtb.shape[0]
    if C.size == 0:
        return np.zeros((0,), dtype=dtype_float), {
            "dtb_min": None, "dtb_max": None, "dtb_q_low": None, "dtb_q_high": None,
            "area_used": False, "area_min": None, "area_max": None,
            "area_q_low": None, "area_q_high": None,
        }

    # Clamp candidates to valid range
    C = C[(C >= 0) & (C < F)]
    if C.size == 0:
        return np.zeros((0,), dtype=dtype_float), {
            "dtb_min": None, "dtb_max": None, "dtb_q_low": None, "dtb_q_high": None,
            "area_used": False, "area_min": None, "area_max": None,
            "area_q_low": None, "area_q_high": None,
        }

    # ---- Normalize DTB to [0,1] (global, not only over candidates) ----
    finite = np.isfinite(dtb)
    if not finite.any():
        # Degenerate case: fall back to zeros
        dtb_norm = np.zeros_like(dtb, dtype=dtype_float)
        dtb_stats = dict(dtb_min=None, dtb_max=None, dtb_q_low=None, dtb_q_high=None)
    else:
        x = dtb[finite]
        if norm_mode == "minmax":
            lo = float(np.min(x))
            hi = float(np.max(x))
        else:  # "quantile" (robust)
            ql = float(np.clip(q_low, 0.0, 1.0))
            qh = float(np.clip(max(q_high, q_low + 1e-6), 0.0, 1.0))
            lo = float(np.quantile(x, ql))
            hi = float(np.quantile(x, qh))
        # Avoid division by zero
        if hi <= lo:
            hi = lo + 1e-8

        dtb_norm = np.zeros_like(dtb, dtype=dtype_float)
        dtb_norm[finite] = np.clip((dtb[finite] - lo) / (hi - lo), 0.0, 1.0).astype(dtype=dtype_float)
        dtb_stats = dict(dtb_min=lo, dtb_max=hi, dtb_q_low=q_low if norm_mode == "quantile" else None,
                         dtb_q_high=q_high if norm_mode == "quantile" else None)

    q = dtb_norm[C].astype(dtype=dtype_float, copy=False)

    # ---- Optional: add a light face-area prior ----
    meta_area = {
        "area_used": False,
        "area_min": None, "area_max": None,
        "area_q_low": None, "area_q_high": None
    }
    if area_mix > 0.0:
        # Get per-face areas; try common attributes first
        if hasattr(mesh, "area_faces"):
            areas = np.asarray(mesh.area_faces, dtype=dtype_float)
        elif hasattr(mesh, "face_areas"):
            areas = np.asarray(mesh.face_areas, dtype=dtype_float)
        else:
            # Fallback: compute from vertices and faces
            V = np.asarray(mesh.vertices, dtype=dtype_float)
            Fidx = np.asarray(mesh.faces, dtype=dtype_index)
            v0 = V[Fidx[:, 1]] - V[Fidx[:, 0]]
            v1 = V[Fidx[:, 2]] - V[Fidx[:, 0]]
            areas = 0.5 * np.linalg.norm(np.cross(v0, v1), axis=1).astype(dtype=dtype_float)

        # Normalize areas to [0,1] (robust or plain minmax)
        finite_a = np.isfinite(areas)
        ax = areas[finite_a]
        if ax.size == 0:
            a_norm = np.zeros_like(areas, dtype=dtype_float)
            a_lo = a_hi = 0.0
        else:
            if area_norm_mode == "minmax":
                a_lo = float(np.min(ax))
                a_hi = float(np.max(ax))
            else:
                aql = float(np.clip(area_q_low, 0.0, 1.0))
                aqh = float(np.clip(max(area_q_high, area_q_low + 1e-6), 0.0, 1.0))
                a_lo = float(np.quantile(ax, aql))
                a_hi = float(np.quantile(ax, aqh))
            if a_hi <= a_lo:
                a_hi = a_lo + 1e-12
            a_norm = np.zeros_like(areas, dtype=dtype_float)
            a_norm[finite_a] = np.clip((areas[finite_a] - a_lo) / (a_hi - a_lo), 0.0, 1.0).astype(dtype=dtype_float)

        # Apply power to compress/extand dynamic range
        if area_power != 1.0:
            a_norm = np.power(a_norm, float(area_power), dtype=dtype_float)

        # Blend into q (convex combination keeps q in [0,1])
        q = (1.0 - float(area_mix)) * q + float(area_mix) * a_norm[C].astype(dtype=dtype_float)

        meta_area.update({
            "area_used": True,
            "area_min": a_lo, "area_max": a_hi,
            "area_q_low": area_q_low if area_norm_mode == "quantile" else None,
            "area_q_high": area_q_high if area_norm_mode == "quantile" else None
        })

    meta = {**dtb_stats, **meta_area}
    return q.astype(dtype=dtype_float, copy=False), meta
  
  
import numpy as np
from scipy.sparse import csgraph


def select_seeds_graph_diverse(
    adj_csr,
    candidates: np.ndarray,
    q_cand: np.ndarray,
    n_seeds: int,
    *,
    beta: float = 2.0,
    limit: float | None = None,
    initial_seeds: np.ndarray | None = None,
) -> np.ndarray:
    """
    Step 7: Deterministic, k-means++-style seed selection on the face-dual graph.

    Picks exactly `n_seeds` face indices that are both high-quality (q) and
    far apart under your segmentation cost metric (graph shortest-path distances).

    Core idea:
      - Maintain D_all[v] = current min graph distance from v to any selected seed.
      - At each iteration select i = argmax_{i in C \ S} q[i] * (D_all[i])**beta.
      - After selecting a new seed s_t, run one Dijkstra from s_t and update D_all = min(D_all, dist_s_t).
      - Repeat until K seeds selected.

    Args:
        adj_csr: (N,N) CSR weighted adjacency (your segmentation costs c_ij as edge weights).
        candidates: (C,) int array of candidate face indices (from Step 4 NMS).
        q_cand: (C,) float array of quality scores in [0,1] aligned to `candidates` (from Step 5).
        n_seeds: desired number of seeds K.
        beta: exponent on D to trade off quality vs. diversity (beta=2 per spec).
        limit: optional Dijkstra cut-off for speed (np.inf if None).
        initial_seeds: optional list/array of preselected seeds (face indices). They are inserted first.

    Returns:
        seed_idx: (K,) int array of selected face indices (matrix row indices), same return
                  convention as your previous `select_seeds(...)`.
    """
    # ---------- Basic validation ----------
    N = adj_csr.shape[0]
    if n_seeds <= 0:
        return np.empty((0,), dtype=np.int64)

    # Normalize and sanitize inputs
    C = np.asarray(candidates, dtype=np.int64).ravel()
    C = C[(C >= 0) & (C < N)]
    if C.size == 0:
        # No candidates -> fall back to farthest-point sampling on the whole graph
        # using pure diversity (q=1). This mirrors FPS but in graph space.
        return _fallback_graph_fps(adj_csr, n_seeds, limit)

    q_cand = np.asarray(q_cand, dtype=np.float64).ravel()
    if q_cand.shape[0] != C.shape[0]:
        raise ValueError("q_cand must have the same length as candidates.")

    K = int(min(n_seeds, N))  # cannot pick more than N unique faces
    lim = np.inf if limit is None else float(limit)

    # Map quality to the full node set for quick lookup; only defined on candidates
    q_full = np.zeros(N, dtype=np.float64)
    q_full[C] = q_cand

    # Candidate mask to restrict argmax domain
    cand_mask = np.zeros(N, dtype=bool)
    cand_mask[C] = True

    # Selected seeds
    seeds = []

    # ---------- Initialize D_all with initial_seeds (if any) ----------
    D_all = np.full(N, np.inf, dtype=np.float64)
    if initial_seeds is not None and len(initial_seeds) > 0:
        init = np.asarray(initial_seeds, dtype=np.int64).ravel()
        init = init[(init >= 0) & (init < N)]
        if init.size > 0:
            seeds.extend([int(s) for s in init])
            # Multi-source: run Dijkstra once with multiple indices and take the row-wise minimum
            dist_init = csgraph.dijkstra(adj_csr, directed=False, indices=init, unweighted=False, limit=lim)
            # If only one source, shape is (N,), ensure 2D for min
            if dist_init.ndim == 1:
                dist_init = dist_init[None, :]
            D_all = np.min(dist_init, axis=0, where=np.isfinite(dist_init), initial=D_all)

    # ---------- If no initial seed, pick the best-quality candidate ----------
    if len(seeds) == 0:
        # Choose s1 = argmax q_i over candidates (deterministic)
        s1 = int(C[np.argmax(q_cand)])
        seeds.append(s1)
        # Initialize D_all with distances from s1
        dist = csgraph.dijkstra(adj_csr, directed=False, indices=s1, unweighted=False, limit=lim)
        D_all = np.minimum(D_all, dist)

    # Ensure uniqueness and respect candidate domain for scoring
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[np.asarray(seeds, dtype=np.int64)] = True

    # ---------- Iteratively select remaining seeds ----------
    while len(seeds) < K:
        # Score only on *remaining candidates*
        # We ignore already selected ones and those outside candidate set.
        viable = cand_mask & (~selected_mask)

        if viable.any():
            # Compute q[i] * D[i]**beta; if D[i] is inf, the product is inf (good: cover new components)
            # To avoid warnings on inf**beta, handle with np.where
            Dpow = np.where(np.isfinite(D_all), np.power(D_all, beta), np.inf)
            score = np.zeros(N, dtype=np.float64)
            score[viable] = q_full[viable] * Dpow[viable]

            s_t = int(np.argmax(score))
            # If all scores are zero (e.g., q=0 or D=0), argmax returns 0; guard by picking best q
            if (score[s_t] == 0.0) and (not np.isinf(D_all[viable]).any()):
                # fallback: pick highest q among viable
                idx = np.argmax(q_full[viable])
                s_t = int(np.where(viable)[0][idx])
        else:
            # Exhausted candidate set: fill remaining seeds by pure diversity (max D_all) on all nodes
            remaining = ~selected_mask
            if not remaining.any():
                break
            s_t = int(np.argmax(np.where(remaining, D_all, -np.inf)))

        # Add the new seed
        if selected_mask[s_t]:
            # Extremely rare in case of ties/fallbacks; break to avoid infinite loop
            break
        seeds.append(s_t)
        selected_mask[s_t] = True

        # Update D_all with one Dijkstra from the newly added seed
        dist = csgraph.dijkstra(adj_csr, directed=False, indices=s_t, unweighted=False, limit=lim)
        D_all = np.minimum(D_all, dist)

    # If for some reason we still have fewer than K (e.g., disconnected singletons without edges),
    # pad deterministically with the smallest unselected indices.
    if len(seeds) < K:
        remaining = np.where(~selected_mask)[0]
        fill = remaining[: (K - len(seeds))]
        seeds.extend([int(x) for x in fill])

    return np.asarray(seeds[:K], dtype=np.int64)


# ---------- Helper: graph farthest-point sampling (deterministic) ----------
def _fallback_graph_fps(adj_csr, n_seeds: int, limit: float | None) -> np.ndarray:
    """
    Deterministic farthest-point sampling on a weighted graph (no randomness),
    used only when no candidate set is provided.
    """
    N = adj_csr.shape[0]
    K = int(min(n_seeds, N))
    if K <= 0:
        return np.empty((0,), dtype=np.int64)

    lim = np.inf if limit is None else float(limit)

    seeds = [0]  # deterministic start
    D_all = csgraph.dijkstra(adj_csr, directed=False, indices=0, unweighted=False, limit=lim)

    for _ in range(1, K):
        # Pick the node farthest from current seeds
        s = int(np.argmax(np.where(np.isfinite(D_all), D_all, -np.inf)))
        if not np.isfinite(D_all[s]):
            # If unreachable, choose the smallest-index node not yet selected
            rem = np.setdiff1d(np.arange(N, dtype=np.int64), np.asarray(seeds, dtype=np.int64), assume_unique=False)
            if rem.size == 0:
                break
            s = int(rem[0])
        seeds.append(s)
        # Update distances
        dist = csgraph.dijkstra(adj_csr, directed=False, indices=s, unweighted=False, limit=lim)
        D_all = np.minimum(D_all, dist)

    return np.asarray(seeds[:K], dtype=np.int64)

