"""
Integration module for automatic seed placement.

This module provides a drop-in replacement for the existing auto_place_optimal_seeds
function using the new facility placement algorithms.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from services.facility_placement import automatic_seed_placement
    FACILITY_PLACEMENT_AVAILABLE = True
except ImportError:
    FACILITY_PLACEMENT_AVAILABLE = False
    print("Warning: Facility placement module not available, using fallback")


def improved_auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds):
    """
    Improved version of auto_place_optimal_seeds using facility placement algorithms.
    
    This is a drop-in replacement for the original function that uses the new
    efficient facility placement algorithms to optimize seed positions.
    
    Args:
        mesh: trimesh object
        curvature_penalty_strength: float, strength of curvature penalty  
        num_seeds: int, number of seeds to place
        
    Returns:
        List of seed positions as [x, y, z] coordinates
    """
    
    if not FACILITY_PLACEMENT_AVAILABLE:
        # Fallback to original implementation if new module not available
        return _fallback_auto_place_seeds(mesh, curvature_penalty_strength, num_seeds)
    
    print(f"Running improved automatic seed placement for {num_seeds} seeds...")
    start_time = time.time()
    
    try:
        # Build adjacency graph using existing function
        from segment_mesh import build_adjacency_graph
        
        sparse_matrix, face_centers = build_adjacency_graph(
            mesh, curvature_penalty_strength, user_seeds=None
        )
        
        print(f"Built adjacency graph: {sparse_matrix.shape[0]} faces, {sparse_matrix.nnz} edges")
        
        # Use the new automatic seed placement algorithm
        seed_indices = automatic_seed_placement(
            adjacency_graph=sparse_matrix,
            num_seeds=num_seeds,
            face_centers=face_centers,
            strategy="adaptive_hybrid",  # Let the algorithm choose the best method
            max_computation_time=30.0,   # Reasonable time limit for web interface
            verbose=True
        )
        
        # Convert indices to face center coordinates (for web interface compatibility)
        seed_positions = []
        for idx in seed_indices:
            if idx < len(face_centers):
                seed_positions.append(face_centers[idx].tolist())
        
        computation_time = time.time() - start_time
        print(f"Improved placement completed in {computation_time:.2f}s")
        print(f"Successfully placed {len(seed_positions)} optimal seeds")
        
        # Verify final distances between seeds for debugging
        if len(seed_positions) > 1:
            min_distance = float('inf')
            max_distance = 0
            for i in range(len(seed_positions)):
                for j in range(i + 1, len(seed_positions)):
                    dist = np.linalg.norm(np.array(seed_positions[i]) - np.array(seed_positions[j]))
                    min_distance = min(min_distance, dist)
                    max_distance = max(max_distance, dist)
            print(f"Seed separation - Min: {min_distance:.3f}, Max: {max_distance:.3f}")
        
        return seed_positions
        
    except Exception as e:
        print(f"Error in improved seed placement: {e}")
        print("Falling back to original implementation")
        return _fallback_auto_place_seeds(mesh, curvature_penalty_strength, num_seeds)


def _fallback_auto_place_seeds(mesh, curvature_penalty_strength, num_seeds):
    """
    Fallback implementation when facility placement module is not available.
    
    This implements a simplified version of the original algorithm.
    """
    print(f"Using fallback seed placement for {num_seeds} seeds...")
    
    face_centers = mesh.triangles_center
    n_faces = len(face_centers)
    
    if num_seeds >= n_faces:
        # Return all face centers if we need more seeds than faces
        return [center.tolist() for center in face_centers]
    
    # Simple farthest-first placement using Euclidean distances
    np.random.seed(42)  # For reproducible results
    
    # Start with a random seed near the center
    mesh_center = np.mean(face_centers, axis=0)
    distances_to_center = np.linalg.norm(face_centers - mesh_center, axis=1)
    first_seed_idx = np.argmin(distances_to_center)
    
    selected_seeds = [first_seed_idx]
    seed_positions = [face_centers[first_seed_idx].tolist()]
    
    # Place remaining seeds using farthest-first heuristic
    for _ in range(1, num_seeds):
        # Find distances to all current seeds
        min_distances = np.full(n_faces, np.inf)
        for seed_idx in selected_seeds:
            distances = np.linalg.norm(face_centers - face_centers[seed_idx], axis=1)
            min_distances = np.minimum(min_distances, distances)
        
        # Place next seed at the farthest point
        next_seed_idx = np.argmax(min_distances)
        selected_seeds.append(next_seed_idx)
        seed_positions.append(face_centers[next_seed_idx].tolist())
    
    print(f"Fallback placement completed: {len(seed_positions)} seeds")
    return seed_positions


# Integration example for web_interface.py
def integrate_with_web_interface():
    """
    Example showing how to integrate the improved seed placement with web_interface.py.
    
    To use this in your web_interface.py:
    
    1. Add this import at the top of web_interface.py:
       from facility_placement_integration import improved_auto_place_optimal_seeds
    
    2. Replace the call to auto_place_optimal_seeds with:
       seed_positions = improved_auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds)
    
    That's it! The rest of your code will work unchanged.
    """
    
    # Example of how the integration would look in web_interface.py
    example_integration_code = '''
    # In web_interface.py, replace this:
    # seed_positions = auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds)
    
    # With this:
    from facility_placement_integration import improved_auto_place_optimal_seeds
    seed_positions = improved_auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds)
    
    # Everything else remains the same!
    '''
    
    print("Integration Example:")
    print(example_integration_code)
    
    return example_integration_code


if __name__ == "__main__":
    print("=== Facility Placement Integration Module ===")
    print()
    
    # Show integration example
    integrate_with_web_interface()
    
    # Test with synthetic data if possible
    if FACILITY_PLACEMENT_AVAILABLE:
        print("\n=== Testing with Synthetic Mesh ===")
        
        # Create a simple synthetic mesh-like object for testing
        class MockMesh:
            def __init__(self, n_faces=50):
                np.random.seed(42)
                self.triangles_center = np.random.uniform(-5, 5, (n_faces, 3))
                self.face_normals = np.random.uniform(-1, 1, (n_faces, 3))
                # Normalize normals
                self.face_normals = self.face_normals / np.linalg.norm(self.face_normals, axis=1, keepdims=True)
                
                # Create some face adjacency (simplified)
                n_adj = min(n_faces * 2, 100)  
                adj_faces = np.random.randint(0, n_faces, (n_adj, 2))
                # Remove self-adjacencies
                mask = adj_faces[:, 0] != adj_faces[:, 1]
                self.face_adjacency = adj_faces[mask]
                
                # Mock edges for compatibility
                self.edges_unique_length = np.random.uniform(0.5, 2.0, n_adj)
        
        try:
            # Test the improved function
            mock_mesh = MockMesh(50)
            result = improved_auto_place_optimal_seeds(mock_mesh, 100.0, 5)
            print(f"Test successful! Generated {len(result)} seed positions")
            print(f"Sample seed: {result[0] if result else 'None'}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            print("This is expected if segment_mesh.py is not available")
    
    else:
        print("\nFacility placement module not available.")
        print("Please ensure src/services/facility_placement.py is accessible.")
    
    print("\n=== Integration Complete ===")
    print("\nTo use this module:")
    print("1. Copy this file to your project root")
    print("2. Import: from facility_placement_integration import improved_auto_place_optimal_seeds") 
    print("3. Replace calls to auto_place_optimal_seeds with improved_auto_place_optimal_seeds")
    print("4. Enjoy better seed placement with automatic algorithm selection!")