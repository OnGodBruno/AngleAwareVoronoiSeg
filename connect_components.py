
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import segment_mesh as sm
from collections import deque


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
    sparse_matrix = csr_matrix((all_weights, (all_row, all_col)),
                                      shape=(N, N), dtype=np.float64)

    return sparse_matrix, face_centers


def find_valleys(mesh, normal_smoothing: bool = True, valley_threshold: float = 0.0) -> np.ndarray:
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
    
    valley_mask = valley_scores > valley_threshold
    
    return valley_scores, valley_mask


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

    sgn = np.sign(np.einsum('ij,ij->i', N[i], N[j])).astype(w_pos.dtype, copy=False)
    w_signed = w_pos * sgn

    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    dat_signed = np.concatenate([w_signed, w_signed])      # mit Vorzeichen
    dat_pos    = np.concatenate([w_pos,    w_pos])         # immer >= 0

    # (N x N) adjancency matrix W. One just for the sign, one just with positive weights
    nF = len(mesh.faces)
    W_signed = csr_matrix((dat_signed, (row, col)), shape=(nF, nF))
    W_pos    = csr_matrix((dat_pos,    (row, col)), shape=(nF, nF))

    deg = (W_pos.sum(axis=1).A.ravel() + 1.0).astype(W_pos.dtype, copy=False)

    for _ in range(max(0, k)):
        tmp = W_signed @ N
        N = (N + tmp) / deg[:, None]
        # normalize to unit length
        nr = np.linalg.norm(N, axis=1, keepdims=True)
        np.divide(N, np.clip(nr, 1e-12, None), out=N)

    return N



def find_connected_components(mesh, edge_mask):
    adj = mesh.face_adjacency    
    edge_idx = np.flatnonzero(edge_mask)
    if edge_idx.size == 0:
        return [], []

    sub = adj[edge_idx]
    u = sub[:, 0].astype(np.int64, copy=False)
    v = sub[:, 1].astype(np.int64, copy=False)

    # active faces and remapping
    active_faces = np.unique(np.concatenate([u, v]))
    face_size = active_faces.size
    map_idx = np.full(mesh.faces.shape[0], -1, dtype=np.int64)
    map_idx[active_faces] = np.arange(face_size, dtype=np.int64)

    ui = map_idx[u]
    vi = map_idx[v]

    # valley sub graph
    rows = np.concatenate([ui, vi])
    cols = np.concatenate([vi, ui])
    data = np.ones(rows.shape[0], dtype=bool)
    A = csr_matrix((data, (rows, cols)), shape=(face_size, face_size), dtype=bool)


    # Gather components
    _, face_labels = csgraph.connected_components(A, directed=False, return_labels=True)

    edge_comp = face_labels[ui]  
    
    order = np.argsort(edge_comp, kind="mergesort")
    edge_comp_sorted = edge_comp[order]
    edges_sorted = edge_idx[order]
    boundaries = np.flatnonzero(np.diff(edge_comp_sorted)) + 1
    components = [p for p in np.split(edges_sorted, boundaries) if p.size > 0]

    deg_face = np.zeros(mesh.faces.shape[0], dtype=np.int32)
    np.add.at(deg_face, u, 1)
    np.add.at(deg_face, v, 1)

    non_valley_mask = ~edge_mask

    if non_valley_mask.shape[0] != mesh.face_adjacency.shape[0]:
        raise ValueError("edge_mask has not the length of mesh.face_adjacency")


    face_has_nonvalley = np.zeros(mesh.faces.shape[0], dtype=bool)
    if np.any(non_valley_mask):
        adj_nv = adj[non_valley_mask]
        u_nv = adj_nv[:, 0]
        v_nv = adj_nv[:, 1]
        face_has_nonvalley[u_nv] = True
        face_has_nonvalley[v_nv] = True


    endpoints = []
    comp_ids_sorted = np.unique(edge_comp_sorted)
    for comp_id in comp_ids_sorted:
        comp_faces = active_faces[face_labels == comp_id]
        end_faces = comp_faces[face_has_nonvalley[comp_faces]]
        endpoints.append(end_faces.astype(np.int64, copy=False))

    return components, endpoints, deg_face


def get_radius_subgraph(graph, face_centers, center_point, euclidean_radius):
    dists = np.linalg.norm(face_centers - center_point, axis=1)
    nearby = np.flatnonzero(dists <= euclidean_radius)
    if nearby.size == 0:
        return None, None
    subgraph = graph[nearby][:, nearby]
    if subgraph.nnz == 0:
        return None, None
    return subgraph, nearby


def dijkstra_bridge_edges(mesh, graph, face_centers, endpoints_per_comp, face_to_comp, radius, search_radius_factor=2.0):
    adj = mesh.face_adjacency
    f0 = np.minimum(adj[:, 0], adj[:, 1])
    f1 = np.maximum(adj[:, 0], adj[:, 1])
    pair_to_idx = {(int(a), int(b)): int(i) for i, (a, b) in enumerate(zip(f0, f1))}

    ep_faces_list, ep_comp_ids_list = [], []
    for comp_id, ep_faces in enumerate(endpoints_per_comp):
        if ep_faces.size:
            ep_faces_list.append(ep_faces.astype(np.int64, copy=False))
            ep_comp_ids_list.append(np.full(ep_faces.size, comp_id, dtype=np.int64))
    if not ep_faces_list:
        return np.zeros(adj.shape[0], dtype=bool)

    endpoint_faces = np.concatenate(ep_faces_list)
    endpoint_comp_ids = np.concatenate(ep_comp_ids_list)

    ep_points = face_centers[endpoint_faces]
    tree = cKDTree(ep_points)

    new_edges_mask = np.zeros(adj.shape[0], dtype=bool)
    bridges_found = 0
    subgraph_radius = radius * search_radius_factor
    processed = np.zeros(endpoint_faces.size, dtype=bool)

    for i in range(endpoint_faces.size):
        if processed[i]:
            continue

        start_face = int(endpoint_faces[i])
        subgraph, sub_to_orig = get_radius_subgraph(graph, face_centers, face_centers[start_face], subgraph_radius)
        if subgraph is None:
            processed[i] = True
            continue

        orig_to_sub = np.full(mesh.faces.shape[0], -1, dtype=np.int32)
        orig_to_sub[sub_to_orig] = np.arange(sub_to_orig.size, dtype=np.int32)

        idxs = tree.query_ball_point(ep_points[i], r=radius)
        if not idxs:
            processed[i] = True
            continue

        src_idxs = []
        src_lookup = {}
        for j in idxs:
            if processed[j]:
                continue
            sf = int(endpoint_faces[j])
            sj = orig_to_sub[sf]
            if sj == -1:
                continue
            src_lookup[len(src_idxs)] = j
            src_idxs.append(sj)

        if not src_idxs:
            processed[i] = True
            continue

        dist, pred = csgraph.dijkstra(subgraph, directed=False, indices=np.array(src_idxs, dtype=int), return_predecessors=True)

        for row, j in src_lookup.items():
            s_comp = int(endpoint_comp_ids[j])

            cand_idx = tree.query_ball_point(ep_points[j], r=radius)
            if not cand_idx:
                processed[j] = True
                continue
            best_face = None
            best_dist = float('inf')
            for k in cand_idx:
                if k == j:
                    continue
                if int(endpoint_comp_ids[k]) == s_comp:
                    continue
                cf = int(endpoint_faces[k])
                csub = orig_to_sub[cf]
                if csub == -1:
                    continue
                d = dist[row, csub]
                if np.isfinite(d) and d < best_dist:
                    best_dist = d
                    best_face = csub

            if best_face is None:
                processed[j] = True
                continue

            path_sub = []
            cur = int(best_face)
            src_sub = int(src_idxs[row])
            while cur != -9999 and cur != src_sub:
                path_sub.append(cur)
                cur = int(pred[row, cur])
            if cur != src_sub:
                processed[j] = True
                continue
            path_sub.append(src_sub)
            path_sub.reverse()

            path = [int(sub_to_orig[idx]) for idx in path_sub]
            for a, b in zip(path[:-1], path[1:]):
                key = (a, b) if a < b else (b, a)
                k = pair_to_idx.get(key, None)
                if k is not None:
                    new_edges_mask[k] = True
            bridges_found += 1
            processed[j] = True

    print(f"  Built {bridges_found} bridges from {len(endpoint_faces)} endpoints (subgraph radius: {subgraph_radius:.4f})")
    return new_edges_mask



def connect_components(mesh, runs=5, curvature_penalty_strength=100.0, search_radius_factor=2.0, valley_threshold: float = 0.1):
    """
    Iteratively stitches broken valley lines by connecting cross-component endpoint faces
    within an Euclidean radius. Promotes the face-paths to edge indices and unions
    them into the edge-level valley_mask. Returns the final edge-level valley_mask.
    
    Args:
        search_radius_factor: Multiplier for graph distance search (default: 2.0)
                             Increase for longer bridges, decrease for speed
    """
    # Initial valley edges
    valley_scores, valley_mask = find_valleys(mesh, normal_smoothing=True, valley_threshold=valley_threshold)

    # Build face graph once
    graph, face_centers = build_adjacency_graph(
        mesh, curvature_penalty_strength=curvature_penalty_strength
    )

    # Radius in world units
    edge_lengths = getattr(mesh, "edges_unique_length", None)
    if edge_lengths is None or edge_lengths.size == 0:
        tri = mesh.triangles
        a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
        b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
        c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
        edge_lengths = np.concatenate([a, b, c])
    
    median_edge = np.median(edge_lengths)
    radius = 3.0 * median_edge
    
    # Print initial component count
    components, endpoints_per_comp, face_to_comp = find_connected_components(mesh, valley_mask)
    total_endpoints = sum(len(ep) for ep in endpoints_per_comp)
    print(f"Iteration 0 (initial): {len(components)} components, {total_endpoints} endpoint faces")
    print(f"Median edge length: {median_edge:.6f}, Euclidean radius: {radius:.6f}, Graph search radius: {radius * search_radius_factor:.6f}")

    for iteration in range(int(runs)):
        # Recompute components/endpoints
        components, endpoints_per_comp, face_to_comp = find_connected_components(mesh, valley_mask)
        
        # Stitch with distance-limited Dijkstra
        new_edges_mask = dijkstra_bridge_edges(
            mesh=mesh,
            graph=graph,
            face_centers=face_centers,
            endpoints_per_comp=endpoints_per_comp,
            face_to_comp=face_to_comp,
            radius=radius,
            search_radius_factor=search_radius_factor,
        )
        
        num_new_edges = np.sum(new_edges_mask)
        
        # Early exit if nothing new was added
        if not np.any(new_edges_mask):
            print(f"Iteration {iteration + 1}: {len(components)} components, 0 new edges - stopping early")
            break

        # Union into valley mask
        valley_mask = np.logical_or(valley_mask, new_edges_mask)
        print(f"Iteration {iteration + 1}: added {num_new_edges} bridge edges, total valley edges: {np.sum(valley_mask)}")

    # Print final component count
    components, endpoints_per_comp, face_to_comp = find_connected_components(mesh, valley_mask)
    print(f"Final: {len(components)} components")

    return valley_mask
           
    


