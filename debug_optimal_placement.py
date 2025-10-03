#!/usr/bin/env python3
"""
Simple test script to debug the optimal placement issue
"""

import sys
import traceback

def test_simple_placement():
    """Test if the issue is in the web interface itself"""
    
    try:
        # Test if we can import the web interface modules
        print("Testing imports...")
        from web_interface import auto_place_optimal_seeds, current_mesh, current_curvature_penalty
        from segment_mesh import load_and_clean_mesh, build_adjacency_graph
        print("✅ Imports successful")
        
        # Test loading a mesh
        print("Testing mesh loading...")
        mesh_path = "input/run/example.obj"
        mesh = load_and_clean_mesh(mesh_path)
        print(f"✅ Mesh loaded: {len(mesh.faces)} faces")
        
        # Test adjacency graph building
        print("Testing adjacency graph...")
        sparse_matrix, face_centers = build_adjacency_graph(mesh, 100.0, user_seeds=None)
        print(f"✅ Graph built: {sparse_matrix.shape}")
        
        # Test optimal seed placement directly
        print("Testing optimal seed placement function...")
        seed_positions = auto_place_optimal_seeds(mesh, 100.0, 3)
        print(f"✅ Optimal placement successful: {len(seed_positions)} seeds placed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_placement()
    sys.exit(0 if success else 1)