"""
Example usage of the automatic seed placement module.

This script demonstrates how to integrate the facility placement algorithm
with the existing mesh segmentation pipeline.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.facility_placement import automatic_seed_placement, PlacementStrategy
from segment_mesh import load_and_clean_mesh, build_adjacency_graph


def example_automatic_placement():
    """Example of how to use the automatic seed placement function."""
    
    # Load a mesh (you can change this path)
    mesh_path = "uploads/100.obj"  # Example mesh file
    
    try:
        # Load and process mesh
        print(f"Loading mesh from: {mesh_path}")
        mesh = load_and_clean_mesh(mesh_path)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Build adjacency graph with penalties
        curvature_penalty_strength = 100.0
        print(f"Building adjacency graph with curvature penalty: {curvature_penalty_strength}")
        sparse_matrix, face_centers = build_adjacency_graph(
            mesh, curvature_penalty_strength, user_seeds=None
        )
        print(f"Built graph: {sparse_matrix.shape[0]} faces, {sparse_matrix.nnz} edges")
        
        # Test different placement strategies
        num_seeds = 8
        strategies = [
            ("adaptive_hybrid", "Adaptive strategy selection"),
            ("gonzalez_approximation", "Gonzalez 2-approximation"),
            ("farthest_first", "Farthest-first heuristic"),
            ("kmeans_plus_plus", "K-means++ style placement")
        ]
        
        results = {}
        
        for strategy_name, description in strategies:
            print(f"\n--- Testing {description} ---")
            
            try:
                seed_indices = automatic_seed_placement(
                    adjacency_graph=sparse_matrix,
                    num_seeds=num_seeds,
                    face_centers=face_centers,
                    strategy=strategy_name,
                    max_computation_time=30.0,
                    verbose=True
                )
                
                print(f"Selected seeds: {seed_indices}")
                results[strategy_name] = seed_indices
                
            except Exception as e:
                print(f"Error with strategy {strategy_name}: {e}")
        
        # Compare results
        print(f"\n--- Comparison of Results ---")
        for strategy_name, seed_indices in results.items():
            print(f"{strategy_name}: {len(seed_indices)} seeds - {seed_indices}")
        
        return results
        
    except FileNotFoundError:
        print(f"Mesh file not found: {mesh_path}")
        print("Please ensure you have a mesh file available or change the path.")
        return None
    except Exception as e:
        print(f"Error in example: {e}")
        return None


def integration_example():
    """
    Example showing how to integrate with the existing web interface.
    
    This shows how you would modify the existing auto_place_optimal_seeds
    function to use the new facility placement algorithm.
    """
    
    def improved_auto_place_seeds(mesh, curvature_penalty_strength, num_seeds):
        """
        Improved version of auto_place_optimal_seeds using facility placement.
        
        This can replace the existing function in web_interface.py
        """
        
        # Build adjacency graph (using existing function)
        from segment_mesh import build_adjacency_graph
        
        print(f"Building adjacency graph for {num_seeds} seeds...")
        sparse_matrix, face_centers = build_adjacency_graph(
            mesh, curvature_penalty_strength, user_seeds=None
        )
        
        # Use the new automatic seed placement
        seed_indices = automatic_seed_placement(
            adjacency_graph=sparse_matrix,
            num_seeds=num_seeds,
            face_centers=face_centers,
            strategy="adaptive_hybrid",  # Automatically choose best strategy
            verbose=True
        )
        
        # Convert indices to face center coordinates (for compatibility)
        seed_positions = []
        for idx in seed_indices:
            if idx < len(face_centers):
                seed_positions.append(face_centers[idx].tolist())
        
        print(f"Automatically placed {len(seed_positions)} optimal seeds")
        return seed_positions
    
    print("Integration example: improved_auto_place_seeds function created")
    print("This function can replace auto_place_optimal_seeds in web_interface.py")


if __name__ == "__main__":
    print("=== Automatic Seed Placement Example ===\n")
    
    # Run the main example
    results = example_automatic_placement()
    
    if results:
        print("\n=== Integration Example ===\n") 
        integration_example()
        
        print("\n=== Success! ===")
        print("The automatic seed placement module is working correctly.")
        print("\nTo integrate with your existing code:")
        print("1. Import: from src.services.facility_placement import automatic_seed_placement")
        print("2. Replace your existing seed selection with:")
        print("   seed_indices = automatic_seed_placement(sparse_matrix, num_seeds, face_centers)")
        print("3. The function will automatically choose the best algorithm based on your mesh size")
    else:
        print("\nExample failed - please check the mesh file path and try again.")