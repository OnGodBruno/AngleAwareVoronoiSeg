
import trimesh
import numpy as np
import os
from scipy.sparse import csr_matrix, csgraph
import connect_components as cc
from collections import deque

# Valley extension configuration
MAX_VALLEY_EXTENSION_DISTANCE = 100  # Maximum number of edges to extend from valley endpoints


def detect_valley_lines(mesh, valley_mask, min_line_length=3):
    """Detect connected sequences of valley edges (valley lines).
    
    Args:
        mesh: trimesh object
        valley_mask: Boolean array marking which edges are valleys
        min_line_length: Minimum number of edges to consider as a valley line
        
    Returns:
        valley_lines: List of lists, where each inner list contains edge indices forming a connected line
        line_endpoints: List of tuples (vertex_idx, line_idx) marking the endpoints of each valley line
    """
    if not valley_mask.any():
        print("No valleys detected")
        return [], []
    
    edges = mesh.face_adjacency_edges  # (num_edges, 2) - vertex pairs for each edge
    valley_edge_indices = np.where(valley_mask)[0]
    
    print(f"\n=== VALLEY LINE DETECTION ===")
    print(f"Total valley edges: {len(valley_edge_indices)}")
    
    # Build vertex-to-valley-edge mapping
    vertex_to_valley_edges = {}
    for edge_idx in valley_edge_indices:
        v1, v2 = edges[edge_idx]
        if v1 not in vertex_to_valley_edges:
            vertex_to_valley_edges[v1] = []
        if v2 not in vertex_to_valley_edges:
            vertex_to_valley_edges[v2] = []
        vertex_to_valley_edges[v1].append(edge_idx)
        vertex_to_valley_edges[v2].append(edge_idx)
    
    # Group connected valley edges into lines using DFS
    visited_edges = set()
    valley_lines = []
    
    for start_edge_idx in valley_edge_indices:
        if start_edge_idx in visited_edges:
            continue
        
        # Grow a line from this edge using DFS
        current_line = []
        stack = [start_edge_idx]
        
        while stack:
            edge_idx = stack.pop()
            if edge_idx in visited_edges:
                continue
            
            visited_edges.add(edge_idx)
            current_line.append(edge_idx)
            
            # Find neighboring valley edges
            v1, v2 = edges[edge_idx]
            for vertex in [v1, v2]:
                for neighbor_edge_idx in vertex_to_valley_edges.get(vertex, []):
                    if neighbor_edge_idx not in visited_edges:
                        stack.append(neighbor_edge_idx)
        
        # Only keep lines with minimum length
        if len(current_line) >= min_line_length:
            valley_lines.append(current_line)
    
    print(f"Found {len(valley_lines)} valley lines with {min_line_length}+ edges")
    
    # Find endpoints for each valley line
    # An endpoint is a vertex that belongs to only 1 valley edge in the line
    line_endpoints = []
    
    for line_idx, line_edges in enumerate(valley_lines):
        # Count how many edges in this line each vertex belongs to
        vertex_count = {}
        for edge_idx in line_edges:
            v1, v2 = edges[edge_idx]
            vertex_count[v1] = vertex_count.get(v1, 0) + 1
            vertex_count[v2] = vertex_count.get(v2, 0) + 1
        
        # Endpoints have count == 1 (they're at the end of the line)
        for vertex, count in vertex_count.items():
            if count == 1:
                line_endpoints.append((vertex, line_idx))
    
    print(f"Found {len(line_endpoints)} endpoints across all valley lines")
    for line_idx, line_edges in enumerate(valley_lines):
        endpoints_in_line = [ep for ep in line_endpoints if ep[1] == line_idx]
        print(f"  Line {line_idx}: {len(line_edges)} edges, {len(endpoints_in_line)} endpoints")
    
    print(f"=== END VALLEY LINE DETECTION ===\n")
    
    return valley_lines, line_endpoints


def extend_valley_lines_from_endpoints(mesh, valley_mask, valley_lines, line_endpoints, max_extension_distance=None):
    """Extend valley lines from their endpoints until hitting another valley edge.
    
    This follows the straightest available path from each endpoint, extending the valley line
    until it connects to an existing valley edge.
    
    Args:
        mesh: trimesh object
        valley_mask: Boolean array marking which edges are valleys
        valley_lines: List of valley line edge indices
        line_endpoints: List of (vertex_idx, line_idx) tuples
        max_extension_distance: Maximum number of edges to extend (uses global constant if None)
        
    Returns:
        valley_mask_extended: Updated valley mask with extensions
        attempted_extensions: List of edge indices that were attempted but didn't connect
    """
    if max_extension_distance is None:
        max_extension_distance = MAX_VALLEY_EXTENSION_DISTANCE
        
    print(f"\n!!! EXTEND FUNCTION CALLED !!!")
    print(f"\n=== EXTENDING VALLEY LINES FROM ENDPOINTS ===")
    print(f"Processing {len(line_endpoints)} endpoints from {len(valley_lines)} valley lines")
    
    if len(line_endpoints) == 0:
        print("No endpoints to extend!")
        return valley_mask.copy(), []
    
    valley_mask_extended = valley_mask.copy()
    attempted_extensions = []  # Store all attempted extension paths
    num_failed_paths = 0
    max_failed_paths = 200  # Show attempts for most endpoints
    edges = mesh.face_adjacency_edges
    vertices = mesh.vertices
    
    # Track successful extensions separately, apply them all at the end
    extensions_to_add = []
    
    # Build vertex-to-edge mapping for all edges (not just valleys)
    vertex_to_all_edges = {}
    for edge_idx, (v1, v2) in enumerate(edges):
        if v1 not in vertex_to_all_edges:
            vertex_to_all_edges[v1] = []
        if v2 not in vertex_to_all_edges:
            vertex_to_all_edges[v2] = []
        vertex_to_all_edges[v1].append(edge_idx)
        vertex_to_all_edges[v2].append(edge_idx)
    
    extensions_made = 0
    total_edges_added = 0
    endpoints_already_connected = 0
    
    # Create a set of all endpoint vertices for fast lookup
    endpoint_vertices = {vertex for vertex, _ in line_endpoints}
    
    print(f"Starting extension process...")
    
    for idx, (endpoint_vertex, line_idx) in enumerate(line_endpoints):
        # Find the direction of the valley line at this endpoint
        line_edge_indices = valley_lines[line_idx]
        line_edges_set = set(line_edge_indices)  # For quick lookup
        
        # Find which edge in the line connects to this endpoint
        endpoint_edge_idx = None
        for edge_idx in line_edge_indices:
            v1, v2 = edges[edge_idx]
            if v1 == endpoint_vertex or v2 == endpoint_vertex:
                endpoint_edge_idx = edge_idx
                break
        
        if endpoint_edge_idx is None:
            continue
        
        # Start extending from the endpoint
        # First, we need to move to the NEXT vertex (not the endpoint itself)
        # Find a non-valley edge from the endpoint to start the extension
        current_vertex = None
        initial_direction = None
        
        # Get the direction from the endpoint edge
        v1, v2 = edges[endpoint_edge_idx]
        if v1 == endpoint_vertex:
            other_vertex_in_line = v2
        else:
            other_vertex_in_line = v1
        
        # Direction vector from other vertex TO endpoint
        edge_vec = vertices[endpoint_vertex] - vertices[other_vertex_in_line]
        edge_vec_norm = np.linalg.norm(edge_vec)
        if edge_vec_norm > 1e-10:
            edge_vec = edge_vec / edge_vec_norm
        else:
            continue
        
        # Find the best non-valley edge from endpoint to start extension
        best_start_edge = None
        best_alignment = -2
        
        for edge_idx in vertex_to_all_edges.get(endpoint_vertex, []):
            # Skip edges from our own valley line
            if edge_idx in line_edges_set:
                continue
            
            # Skip valley edges (we want to extend through non-valley space)
            if valley_mask[edge_idx]:
                continue
            
            ev1, ev2 = edges[edge_idx]
            next_vertex = ev2 if ev1 == endpoint_vertex else ev1
            
            # Calculate direction
            candidate_vec = vertices[next_vertex] - vertices[endpoint_vertex]
            candidate_norm = np.linalg.norm(candidate_vec)
            if candidate_norm > 1e-10:
                candidate_vec = candidate_vec / candidate_norm
            else:
                continue
            
            # Measure alignment with valley line direction
            alignment = np.dot(edge_vec, candidate_vec)
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_start_edge = edge_idx
                current_vertex = next_vertex
                initial_direction = candidate_vec
        
        if current_vertex is None:
            # No non-valley edge to start from - endpoint is surrounded by valleys
            endpoints_already_connected += 1
            continue
        
        # Now start the extension from the first vertex
        visited_vertices = {endpoint_vertex, current_vertex}
        path_edges = [best_start_edge]
        hit_valley = False
        edge_vec = initial_direction
        
        for step in range(max_extension_distance):
            # Find candidate edges from current vertex (non-valley, unvisited)
            candidate_edges = []
            
            edges_at_vertex = vertex_to_all_edges.get(current_vertex, [])
            
            for edge_idx in edges_at_vertex:
                # IMPORTANT: Skip edges that are part of OUR OWN valley line
                if edge_idx in line_edges_set:
                    continue
                
                ev1, ev2 = edges[edge_idx]
                next_vertex = ev2 if ev1 == current_vertex else ev1
                
                # Check if we reached another valley ENDPOINT (not just any valley edge)
                if next_vertex in endpoint_vertices and next_vertex != endpoint_vertex:
                    # We hit a different endpoint! Success!
                    hit_valley = True
                    path_edges.append(edge_idx)  # Include the final connecting edge
                    break
                
                # Skip valley edges (we only want to connect endpoint-to-endpoint, not to any valley)
                if valley_mask[edge_idx]:
                    continue
                
                # Skip if already in our ORIGINAL valleys (not the extended ones yet)
                # We don't check valley_mask_extended here to avoid interference between extensions
                
                if next_vertex in visited_vertices:
                    continue
                
                candidate_edges.append((edge_idx, next_vertex))
            
            # If we hit an endpoint, save the path for later
            if hit_valley:
                if len(path_edges) > 0:
                    extensions_to_add.extend(path_edges)
                    extensions_made += 1
                    total_edges_added += len(path_edges)
                    if extensions_made <= 10:  # Print first 10 for clarity
                        print(f"  ✓ Extended line {line_idx}: connected to endpoint with {len(path_edges)} edges")
                else:
                    endpoints_already_connected += 1
                break
            
            if not candidate_edges:
                # Dead end - no valley found, but store the attempted path
                if len(path_edges) > 0 and num_failed_paths < max_failed_paths:
                    attempted_extensions.extend(path_edges)
                    num_failed_paths += 1
                    print(f"  ✗ Failed to connect line {line_idx}: stored {len(path_edges)} edges as attempted extension")
                break
            
            # Choose the straightest edge (most aligned with current direction)
            best_edge_idx = None
            best_alignment = -2
            best_next_vertex = None
            
            for edge_idx, next_vertex in candidate_edges:
                ev1, ev2 = edges[edge_idx]
                candidate_vec = vertices[ev2] - vertices[ev1]
                candidate_norm = np.linalg.norm(candidate_vec)
                if candidate_norm > 1e-10:
                    candidate_vec = candidate_vec / candidate_norm
                else:
                    continue
                
                # Measure alignment with previous direction
                alignment = np.abs(np.dot(edge_vec, candidate_vec))
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_edge_idx = edge_idx
                    best_next_vertex = next_vertex
            
            if best_edge_idx is None:
                break
            
            # Add this edge to the path
            path_edges.append(best_edge_idx)
            visited_vertices.add(best_next_vertex)
            current_vertex = best_next_vertex
            
            # Update direction for next iteration
            ev1, ev2 = edges[best_edge_idx]
            edge_vec = vertices[ev2] - vertices[ev1]
            edge_vec_norm = np.linalg.norm(edge_vec)
            if edge_vec_norm > 1e-10:
                edge_vec = edge_vec / edge_vec_norm
        
        # If we didn't hit a valley (loop ended without break or hit max distance), store as failed attempt
        if not hit_valley and len(path_edges) > 0 and num_failed_paths < max_failed_paths:
            # Calculate total path length for debugging
            total_path_length = 0
            for edge_idx in path_edges:
                v1, v2 = edges[edge_idx]
                edge_length = np.linalg.norm(vertices[v2] - vertices[v1])
                total_path_length += edge_length
            
            attempted_extensions.extend(path_edges)
            num_failed_paths += 1
            print(f"  ✗ Failed to connect line {line_idx}: reached max distance, stored {len(path_edges)} edges (total length: {total_path_length:.4f})")
    
    # Now apply all successful extensions to valley_mask_extended
    for edge_idx in extensions_to_add:
        valley_mask_extended[edge_idx] = True
    
    print(f"Extensions: {extensions_made} successful ({total_edges_added} edges added)")
    print(f"Already connected: {endpoints_already_connected} endpoints were already next to valleys")
    print(f"Failed attempts: {num_failed_paths} paths couldn't connect")
    print(f"Attempted extensions stored: {len(attempted_extensions)} edges")
    print(f"=== END VALLEY LINE EXTENSION ===\n")
    
    return valley_mask_extended, attempted_extensions


def connect_valley_lines(mesh, valley_mask, max_gap_distance=3):
    """Connect valley lines using two strategies:
    
    Strategy 1 - Fix Offsets: Connect endpoints of valley lines that are close to each other.
    When a valley line shifts one edge row, this creates a gap. Connect endpoints within 3 edges.
    
    Strategy 2 - Extend Lines: From each line endpoint, extend by following the straightest path
    and connect if we get within 3 edges of another endpoint.
    
    Args:
        mesh: The mesh object
        valley_mask: Boolean array indicating which face adjacency edges are valleys
        max_gap_distance: Maximum edge distance to bridge (default: 3)
    
    Returns:
        valley_mask_connected: Updated valley mask with connections added
    """
    print(f"\n=== Connecting valley lines (max gap: {max_gap_distance} edges) ===")
    
    valley_mask_connected = valley_mask.copy()
    edges = mesh.face_adjacency_edges
    
    # Build edge-to-edge adjacency (edges that share a vertex)
    vertex_to_edge_indices = {}
    for edge_idx, (v1, v2) in enumerate(edges):
        if v1 not in vertex_to_edge_indices:
            vertex_to_edge_indices[v1] = []
        if v2 not in vertex_to_edge_indices:
            vertex_to_edge_indices[v2] = []
        vertex_to_edge_indices[v1].append(edge_idx)
        vertex_to_edge_indices[v2].append(edge_idx)
    
    # Find valley edges and build valley edge connectivity
    valley_edge_indices = np.where(valley_mask)[0]
    valley_edges_set = set(valley_edge_indices)
    
    # For each vertex, count how many valley edges connect to it
    vertex_valley_count = {}
    for edge_idx in valley_edge_indices:
        v1, v2 = edges[edge_idx]
        vertex_valley_count[v1] = vertex_valley_count.get(v1, 0) + 1
        vertex_valley_count[v2] = vertex_valley_count.get(v2, 0) + 1
    
    # Find endpoints: vertices with exactly 1 valley edge
    endpoints = [v for v, count in vertex_valley_count.items() if count == 1]
    
    print(f"Found {len(valley_edge_indices)} valley edges")
    print(f"Found {len(endpoints)} endpoints")
    
    if len(endpoints) == 0:
        print("No endpoints to connect")
        return valley_mask_connected
    
    # --- Strategy 1: Fix Offsets ---
    # Connect endpoints that are close to each other (within max_gap_distance edges)
    print("\nStrategy 1: Fixing offsets...")
    connected_endpoints = set()
    connections_made_s1 = 0
    
    for start_vertex in endpoints:
        if start_vertex in connected_endpoints:
            continue
        
        # BFS to find nearby endpoints
        queue = deque([(start_vertex, [start_vertex], 0)])
        visited = {start_vertex}
        
        while queue:
            current_vertex, path, depth = queue.popleft()
            
            if depth >= max_gap_distance:
                continue
            
            # Get all edges connected to this vertex
            for edge_idx in vertex_to_edge_indices.get(current_vertex, []):
                # Skip if already a valley edge
                if edge_idx in valley_edges_set:
                    continue
                
                # Get the other vertex
                v1, v2 = edges[edge_idx]
                next_vertex = v2 if v1 == current_vertex else v1
                
                if next_vertex in visited:
                    continue
                
                visited.add(next_vertex)
                new_path = path + [next_vertex]
                
                # Check if we reached another endpoint
                if next_vertex in endpoints and next_vertex != start_vertex and next_vertex not in connected_endpoints:
                    # Connect by marking all edges in path as valleys
                    for i in range(len(new_path) - 1):
                        # Find edge between new_path[i] and new_path[i+1]
                        for e_idx in vertex_to_edge_indices.get(new_path[i], []):
                            ev1, ev2 = edges[e_idx]
                            if (ev1 == new_path[i] and ev2 == new_path[i+1]) or \
                               (ev2 == new_path[i] and ev1 == new_path[i+1]):
                                valley_mask_connected[e_idx] = True
                                valley_edges_set.add(e_idx)
                                break
                    
                    connected_endpoints.add(start_vertex)
                    connected_endpoints.add(next_vertex)
                    connections_made_s1 += 1
                    print(f"  Connected endpoints {start_vertex} <-> {next_vertex} (distance: {len(new_path)-1})")
                    break
                
                queue.append((next_vertex, new_path, depth + 1))
    
    print(f"Strategy 1 complete: Made {connections_made_s1} connections")
    
    # --- Strategy 2: Extend Lines ---
    # From each remaining endpoint, extend along the straightest path
    print("\nStrategy 2: Extending lines...")
    
    # Recompute endpoints after Strategy 1
    vertex_valley_count = {}
    valley_edge_indices = np.where(valley_mask_connected)[0]
    valley_edges_set = set(valley_edge_indices)
    for edge_idx in valley_edge_indices:
        v1, v2 = edges[edge_idx]
        vertex_valley_count[v1] = vertex_valley_count.get(v1, 0) + 1
        vertex_valley_count[v2] = vertex_valley_count.get(v2, 0) + 1
    
    endpoints = [v for v, count in vertex_valley_count.items() if count == 1]
    connections_made_s2 = 0
    
    for start_vertex in endpoints:
        # Find the direction of the existing valley line at this endpoint
        start_valley_edges = [e for e in vertex_to_edge_indices.get(start_vertex, []) if e in valley_edges_set]
        if len(start_valley_edges) != 1:
            continue
        
        start_edge_idx = start_valley_edges[0]
        v1, v2 = edges[start_edge_idx]
        prev_vertex = v2 if v1 == start_vertex else v1
        
        # Calculate direction vector of the valley line
        prev_dir = mesh.vertices[start_vertex] - mesh.vertices[prev_vertex]
        prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-10)
        
        # Extend by following the straightest path
        current_vertex = start_vertex
        current_dir = prev_dir
        path = [start_vertex]
        visited = {start_vertex, prev_vertex}
        
        for step in range(max_gap_distance * 3):  # Allow longer extension
            # Find all candidate next edges
            best_edge = None
            best_next_vertex = None
            best_alignment = -2  # Start with worst possible
            
            for edge_idx in vertex_to_edge_indices.get(current_vertex, []):
                if edge_idx in valley_edges_set:
                    continue
                
                ev1, ev2 = edges[edge_idx]
                next_vertex = ev2 if ev1 == current_vertex else ev1
                
                if next_vertex in visited:
                    continue
                
                # Calculate direction to next vertex
                next_dir = mesh.vertices[next_vertex] - mesh.vertices[current_vertex]
                next_dir = next_dir / (np.linalg.norm(next_dir) + 1e-10)
                
                # How aligned is this with our current direction?
                alignment = np.dot(current_dir, next_dir)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_edge = edge_idx
                    best_next_vertex = next_vertex
            
            if best_next_vertex is None:
                break  # No more candidate edges
            
            visited.add(best_next_vertex)
            path.append(best_next_vertex)
            
            # Check if we're within max_gap_distance of any endpoint
            if best_next_vertex in endpoints and best_next_vertex != start_vertex:
                # Found an endpoint! Connect the path
                for i in range(len(path) - 1):
                    for e_idx in vertex_to_edge_indices.get(path[i], []):
                        ev1, ev2 = edges[e_idx]
                        if (ev1 == path[i] and ev2 == path[i+1]) or \
                           (ev2 == path[i] and ev1 == path[i+1]):
                            valley_mask_connected[e_idx] = True
                            valley_edges_set.add(e_idx)
                            break
                
                connections_made_s2 += 1
                print(f"  Extended from {start_vertex} to endpoint {best_next_vertex} (distance: {len(path)-1})")
                break
            
            # Check if within max_gap_distance edges of any endpoint via BFS
            if len(path) % 2 == 0:  # Check periodically to avoid slowdown
                for endpoint in endpoints:
                    if endpoint == start_vertex or endpoint in visited:
                        continue
                    
                    # Quick BFS to see if endpoint is close
                    bfs_queue = deque([(best_next_vertex, 0)])
                    bfs_visited = {best_next_vertex}
                    found_close = False
                    
                    while bfs_queue and not found_close:
                        check_v, dist = bfs_queue.popleft()
                        if dist >= max_gap_distance:
                            continue
                        
                        if check_v == endpoint:
                            found_close = True
                            break
                        
                        for check_e in vertex_to_edge_indices.get(check_v, []):
                            if check_e in valley_edges_set:
                                continue
                            cev1, cev2 = edges[check_e]
                            check_next = cev2 if cev1 == check_v else cev1
                            if check_next not in bfs_visited:
                                bfs_visited.add(check_next)
                                bfs_queue.append((check_next, dist + 1))
                    
                    if found_close:
                        # Found a close endpoint, but we'll let Strategy 1 handle it in next iteration
                        pass
            
            # Update current position and direction
            current_vertex = best_next_vertex
            current_dir = mesh.vertices[current_vertex] - mesh.vertices[path[-2]]
            current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-10)
    
    print(f"Strategy 2 complete: Made {connections_made_s2} connections")
    print(f"\nTotal: Added {valley_mask_connected.sum() - valley_mask.sum()} bridge edges")
    
    return valley_mask_connected


def load_and_clean_mesh(mesh_path):
    """Load and clean a 3D mesh."""
    mesh = trimesh.load(mesh_path, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fix_normals()
    return mesh


def get_valley_faces(mesh, valley_threshold=0.1, connection_runs=5, curvature_penalty_strength=100.0):
    """Find valley edges and return their vertex positions for visualization.
    
    Args:
        mesh: The mesh object
        valley_threshold: Minimum valley score to consider an edge as a valley
        connection_runs: Not used anymore
        curvature_penalty_strength: Not used, kept for compatibility
    
    Returns:
        valley_edges: Array of edge vertex positions, shape (N, 2, 3) where N is number of valley edges
        valley_scores_edges: Valley score for each edge
    """
    
    # Find initial valley edges
    print(f"\nFinding valley edges (threshold={valley_threshold})...")
    valley_scores, valley_mask = cc.find_valleys(mesh, normal_smoothing=True, valley_threshold=valley_threshold)
    
    print(f"Initial valley edges: {valley_mask.sum()}")
    
    # Detect valley lines and extend them
    valley_lines, line_endpoints = detect_valley_lines(mesh, valley_mask, min_line_length=3)
    valley_mask_extended, attempted_extensions = extend_valley_lines_from_endpoints(
        mesh, valley_mask, valley_lines, line_endpoints
    )
    
    print(f"Final valley edges (after extension): {valley_mask_extended.sum()}")

    # Get the actual edges (vertex pairs) for valley edges
    valley_edge_indices = mesh.face_adjacency_edges[valley_mask_extended]
    
    # Get the 3D positions of the edge vertices
    valley_edges = mesh.vertices[valley_edge_indices]  # Shape: (num_valley_edges, 2, 3)
    
    # Get scores for valley edges (extended edges get score of 0.5)
    valley_scores_edges = np.where(valley_mask, valley_scores, 0.5)[valley_mask_extended]
    
    # Get attempted extension edges for visualization
    attempted_edge_positions = []
    if len(attempted_extensions) > 0:
        attempted_edge_indices = mesh.face_adjacency_edges[attempted_extensions]
        attempted_edge_positions = mesh.vertices[attempted_edge_indices]
    
    print(f"Returning {len(valley_edges)} valley edges and {len(attempted_edge_positions)} attempted extensions for visualization")

    return valley_edges, valley_scores_edges, attempted_edge_positions


def connect_nearby_valleys(mesh, valley_mask, max_iterations=5, max_gap_distance=3):
    """Connect valley edges that are separated by small gaps.
    
    Strategy:
    1. Build valley "lines" by following connected valley edges
    2. Find endpoints of these lines (vertices with only 1 valley edge)
    3. Connect nearby endpoints by finding shortest paths
    
    Args:
        mesh: The mesh object
        valley_mask: Boolean array indicating which edges are valleys
        max_iterations: Maximum number of connection iterations
        max_gap_distance: Maximum edge distance to bridge between endpoints
    
    Returns:
        valley_mask_connected: Updated valley mask with connections added
    """
    valley_mask_connected = valley_mask.copy()
    
    # Build vertex-to-edges mapping
    vertex_to_edges = {}
    for edge_idx, (v1, v2) in enumerate(mesh.face_adjacency_edges):
        if v1 not in vertex_to_edges:
            vertex_to_edges[v1] = []
        if v2 not in vertex_to_edges:
            vertex_to_edges[v2] = []
        vertex_to_edges[v1].append(edge_idx)
        vertex_to_edges[v2].append(edge_idx)
    
    for iteration in range(max_iterations):
        added_this_iteration = 0
        
        # Step 1: Find all line endpoints
        # An endpoint is a vertex that has exactly 1 valley edge attached
        endpoints = []
        for vertex_idx in vertex_to_edges.keys():
            connected_edges = vertex_to_edges[vertex_idx]
            valley_count = sum(1 for e in connected_edges if e < len(valley_mask_connected) and valley_mask_connected[e])
            
            if valley_count == 1:
                endpoints.append(vertex_idx)
        
        print(f"  Iteration {iteration + 1}: Found {len(endpoints)} valley line endpoints")
        
        if len(endpoints) == 0:
            print(f"  No endpoints found, stopping")
            break
        
        # Step 2: For each endpoint, find the closest other endpoint within max_gap_distance
        connections_to_make = []
        
        for start_endpoint in endpoints:
            # Use Dijkstra-like BFS to find distance to all other vertices
            distances = {start_endpoint: 0}
            parent_edges = {start_endpoint: None}
            queue = [start_endpoint]
            visited = set([start_endpoint])
            
            while queue:
                current_vertex = queue.pop(0)
                current_dist = distances[current_vertex]
                
                # Don't go beyond max distance
                if current_dist >= max_gap_distance:
                    continue
                
                # Explore neighboring vertices via non-valley edges
                for edge_idx in vertex_to_edges[current_vertex]:
                    # Skip valley edges (we only cross non-valley edges)
                    if valley_mask_connected[edge_idx]:
                        continue
                    
                    # Get the other vertex of this edge
                    edge_vertices = mesh.face_adjacency_edges[edge_idx]
                    next_vertex = edge_vertices[1] if edge_vertices[0] == current_vertex else edge_vertices[0]
                    
                    if next_vertex in visited:
                        continue
                    
                    visited.add(next_vertex)
                    distances[next_vertex] = current_dist + 1
                    parent_edges[next_vertex] = edge_idx
                    queue.append(next_vertex)
            
            # Find the closest endpoint (excluding self)
            closest_endpoint = None
            closest_distance = float('inf')
            
            for endpoint in endpoints:
                if endpoint == start_endpoint:
                    continue
                if endpoint in distances and distances[endpoint] < closest_distance:
                    closest_distance = distances[endpoint]
                    closest_endpoint = endpoint
            
            # If we found a close endpoint, record the connection
            if closest_endpoint is not None and closest_distance <= max_gap_distance:
                # Reconstruct the path
                path_edges = []
                current = closest_endpoint
                while parent_edges.get(current) is not None:
                    path_edges.append(parent_edges[current])
                    # Get the parent vertex
                    edge_vertices = mesh.face_adjacency_edges[parent_edges[current]]
                    prev_vertex = edge_vertices[0] if edge_vertices[1] == current else edge_vertices[1]
                    current = prev_vertex
                
                if len(path_edges) > 0:
                    # Store as tuple (sorted to avoid duplicates)
                    connection_key = tuple(sorted([start_endpoint, closest_endpoint]))
                    connections_to_make.append((connection_key, path_edges))
        
        # Step 3: Apply connections (remove duplicates)
        unique_connections = {}
        for connection_key, path_edges in connections_to_make:
            if connection_key not in unique_connections:
                unique_connections[connection_key] = path_edges
        
        # Add all edges in the paths
        for path_edges in unique_connections.values():
            for edge_idx in path_edges:
                if not valley_mask_connected[edge_idx]:
                    valley_mask_connected[edge_idx] = True
                    added_this_iteration += 1
        
        print(f"  Iteration {iteration + 1}: Made {len(unique_connections)} connections, added {added_this_iteration} edges")
        
        if added_this_iteration == 0:
            print(f"  No new connections made, stopping")
            break
    
    return valley_mask_connected

  
    
def build_adjacency_graph(mesh, curvature_penalty_strength=0.1, user_seeds=None):
    """Build a face adjacency graph with valley-aware edge weights.
    
    Args:
        mesh: trimesh object
        curvature_penalty_strength: Valley detection threshold (lower = fewer valleys)
        user_seeds: Not used, kept for compatibility
        
    Returns:
        sparse_matrix: Weighted adjacency matrix (valleys are REMOVED, not infinite)
        face_centers: Array of face centroids
    """
    face_centers = mesh.triangles_center
    adj = mesh.face_adjacency

    # Find valley edges - they will be REMOVED from the graph
    valley_scores, valley_mask = cc.find_valleys(mesh, normal_smoothing=True, valley_threshold=curvature_penalty_strength)
    
    print(f"\n=== VALLEY DETECTION DEBUG ===")
    print(f"Valley scores: min={valley_scores.min():.4f}, max={valley_scores.max():.4f}, mean={valley_scores.mean():.4f}")
    print(f"Threshold: {curvature_penalty_strength}")
    print(f"Initial valleys detected: {valley_mask.sum()} out of {len(valley_mask)} edges")
    
    if valley_mask.sum() > 0:
        print(f"Valley score range for detected valleys: {valley_scores[valley_mask].min():.4f} to {valley_scores[valley_mask].max():.4f}")
    
    # Detect valley lines and their endpoints
    valley_lines, line_endpoints = detect_valley_lines(mesh, valley_mask, min_line_length=3)
    
    # Extend valley lines from endpoints
    valley_mask_extended, _ = extend_valley_lines_from_endpoints(
        mesh, valley_mask, valley_lines, line_endpoints
    )
    
    print(f"After extension: {valley_mask_extended.sum()} valley edges (was {valley_mask.sum()})")
    
    # CRITICAL FIX: Instead of setting valleys to inf, we EXCLUDE them from the graph
    # Only keep non-valley edges
    non_valley_mask = ~valley_mask_extended
    
    print(f"Building graph with {non_valley_mask.sum()} edges (removed {valley_mask_extended.sum()} valley edges)")
    print(f"=== END DEBUG ===\n")

    # Build symmetric sparse matrix WITHOUT valley edges
    row = adj[non_valley_mask, 0]
    col = adj[non_valley_mask, 1]
    weights = np.ones(len(row), dtype=np.float64)
    
    all_row = np.concatenate([row, col])
    all_col = np.concatenate([col, row])
    all_weights = np.concatenate([weights, weights])

    N = face_centers.shape[0]
    sparse_matrix = csr_matrix((all_weights, (all_row, all_col)),
                              shape=(N, N), dtype=np.float64)

    return sparse_matrix, face_centers


def pick_first_seed(face_centers, pool_size=64):
    """Pick initial seed using random pool selection.
    
    Args:
        face_centers: Array of face centroids, shape (N, 3)
        pool_size: Number of random candidates to consider
        
    Returns:
        Index of the selected seed face
    """
    rng = np.random.default_rng(42)
    n_faces = face_centers.shape[0]

    pool = rng.choice(n_faces, size=pool_size, replace=False)
    sub = face_centers[pool]
    dist = np.linalg.norm(sub[:, None] - sub[None], axis=2)

    max_dist = np.argmax(dist.sum(axis=1))
    return pool[max_dist]


def select_seeds(face_centers, n_seeds):
    """Select seed faces using stochastic farthest-point sampling.
    
    Args:
        face_centers: Array of face centroids, shape (N, 3)
        n_seeds: Number of seeds to select
        
    Returns:
        Array of seed face indices
    """
    rng = np.random.default_rng(42)
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
    """Segment a mesh by multi-source geodesic propagation.
    
    Args:
        sparse_matrix: Weighted adjacency matrix
        seed_idx: Array of seed face indices
        
    Returns:
        face_labels: Dict mapping face_i -> seed_j (closest seed)
        face_distances: Dict mapping face_i -> distance from its seed
    """
    dist = csgraph.dijkstra(sparse_matrix, indices=seed_idx, directed=False, return_predecessors=False)

    # Count unreachable faces
    inf_count = np.sum(~np.isfinite(dist), axis=None)
    if inf_count > 0:
        print(f"Warning: {inf_count}/{dist.size} faces unreachable ({100 * inf_count / dist.size:.2f}%)")

    winner = np.argmin(dist, axis=0)
    face_labels = {i: int(seed_idx[winner[i]]) for i in range(sparse_matrix.shape[0])}
    
    # Extract the distance for each face to its assigned seed
    face_distances = {i: float(dist[winner[i], i]) for i in range(sparse_matrix.shape[0])}
    
    return face_labels, face_distances


def find_dijkstra_path(sparse_matrix, start_face_idx, end_face_idx):
    """Find the shortest path between two faces using Dijkstra's algorithm.
    
    Args:
        sparse_matrix: Weighted adjacency matrix
        start_face_idx: Starting face index
        end_face_idx: Ending face index
        
    Returns:
        path: List of face indices from start to end (empty if no path exists)
        distance: Total distance of the path
    """
    print(f"\n=== PATH FINDING DEBUG ===")
    print(f"Finding path from face {start_face_idx} to face {end_face_idx}")
    print(f"Graph: {sparse_matrix.shape[0]} faces, {sparse_matrix.nnz} total edges")
    
    # Run Dijkstra from the start face, getting both distances and predecessors
    distances, predecessors = csgraph.dijkstra(
        sparse_matrix, 
        indices=[start_face_idx], 
        directed=False, 
        return_predecessors=True
    )
    
    # Extract for our single source
    dist_array = distances[0]
    pred_array = predecessors[0]
    
    # Check if end is reachable
    if not np.isfinite(dist_array[end_face_idx]):
        print(f"✓ CORRECT: No path exists (separated by valleys)")
        print(f"=== END PATH DEBUG ===\n")
        return [], float('inf')
    
    print(f"✗ WARNING: Path found with distance {dist_array[end_face_idx]}")
    
    # Reconstruct path from predecessors
    path = []
    current = end_face_idx
    
    while current != start_face_idx:
        path.append(int(current))
        current = pred_array[current]
        
        if current == -9999:  # No predecessor (shouldn't happen if reachable)
            print(f"Path reconstruction failed")
            print(f"=== END PATH DEBUG ===\n")
            return [], float('inf')
        
        if len(path) > sparse_matrix.shape[0]:  # Safety check for cycles
            print(f"Cycle detected in path reconstruction")
            print(f"=== END PATH DEBUG ===\n")
            return [], float('inf')
    
    path.append(int(start_face_idx))
    path.reverse()  # Reverse to get start -> end order
    
    total_distance = float(dist_array[end_face_idx])
    print(f"Path has {len(path)} faces, total distance: {total_distance:.2f}")
    print(f"=== END PATH DEBUG ===\n")
    
    return path, total_distance


def export_segment(mesh, face_labels, seed_idx, output_dir):
    """Export segmented mesh parts to OBJ files."""
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
