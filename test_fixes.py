#!/usr/bin/env python3
"""
Test script to validate the mesh segmentation fixes.
"""

import numpy as np
import trimesh
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh

def test_segmentation_coverage():
    """Test that segmentation properly handles all faces."""
    
    # Load a test mesh
    mesh_path = "input/run/example.obj"
    try:
        mesh = load_and_clean_mesh(mesh_path)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Build adjacency graph
        sparse_matrix, face_coords, active_faces = build_adjacency_graph(mesh, curvature_penalty_strength=100.0)
        print(f"Built graph: {sparse_matrix.shape[0]} active faces out of {len(mesh.faces)} total faces")
        
        # Simulate manual seed selection (pick some faces)
        if len(face_coords) >= 3:
            seed_idx = np.array([0, len(face_coords)//3, 2*len(face_coords)//3])
        else:
            seed_idx = np.array([0])
        
        print(f"Using {len(seed_idx)} seeds: {seed_idx}")
        
        # Run segmentation
        face_labels = segment_mesh(sparse_matrix, seed_idx)
        
        # Analyze results
        total_active_faces = sparse_matrix.shape[0]
        labeled_faces = len(face_labels)
        unlabeled_faces = total_active_faces - labeled_faces
        inactive_faces = len(mesh.faces) - len(active_faces)
        
        print(f"\nResults:")
        print(f"  Total mesh faces: {len(mesh.faces)}")
        print(f"  Active faces (in graph): {len(active_faces)}")
        print(f"  Inactive faces (filtered out): {inactive_faces}")
        print(f"  Labeled faces (reachable): {labeled_faces}")
        print(f"  Unlabeled faces (unreachable): {unlabeled_faces}")
        
        # Calculate coverage percentages
        active_coverage = (len(active_faces) / len(mesh.faces)) * 100
        label_coverage = (labeled_faces / len(active_faces)) * 100 if len(active_faces) > 0 else 0
        total_coverage = (labeled_faces / len(mesh.faces)) * 100
        
        print(f"\nCoverage:")
        print(f"  Active faces: {active_coverage:.1f}% of total mesh")
        print(f"  Labeled faces: {label_coverage:.1f}% of active faces")
        print(f"  Total labeled: {total_coverage:.1f}% of total mesh")
        
        # Simulate the color assignment logic from web_interface.py
        face_colors = [[0.2, 0.2, 0.3] for _ in range(len(mesh.faces))]  # Default dark blue-gray
        
        # Color segments
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        row_to_segment = {row: i for i, row in enumerate(seed_idx)}
        colored_faces = 0
        segmented_faces = set()
        
        for row_idx, seed_row in face_labels.items():
            seg_id = row_to_segment[seed_row]
            face_id = active_faces[row_idx]
            color_idx = seg_id % len(colors)
            face_colors[face_id] = colors[color_idx]
            colored_faces += 1
            segmented_faces.add(face_id)
        
        # Handle inactive faces (same logic as web_interface.py)
        all_face_centers = mesh.triangles_center
        inactive_faces = set(range(len(mesh.faces))) - set(active_faces)
        
        if inactive_faces:
            print(f"Handling {len(inactive_faces)} inactive faces...")
            active_centers = all_face_centers[active_faces]
            
            for inactive_face_id in inactive_faces:
                inactive_center = all_face_centers[inactive_face_id]
                # Find closest active face
                distances = np.linalg.norm(active_centers - inactive_center, axis=1)
                closest_active_idx = np.argmin(distances)
                closest_active_face_id = active_faces[closest_active_idx]
                
                # Copy the color from the closest active face
                if closest_active_face_id in segmented_faces:
                    face_colors[inactive_face_id] = face_colors[closest_active_face_id]
                    colored_faces += 1
        
        uncolored_faces = len(mesh.faces) - colored_faces
        print(f"\nColor assignment:")
        print(f"  Colored faces: {colored_faces}")
        print(f"  Uncolored faces: {uncolored_faces}")
        print(f"  Uncolored percentage: {(uncolored_faces/len(mesh.faces))*100:.1f}%")
        
        if uncolored_faces > 0:
            print(f"\nWARNING: {uncolored_faces} faces will appear in default color!")
            print("This is the root cause of the 'faces not colored' issue.")
            
            # Analyze why faces are uncolored
            print("\nReasons for uncolored faces:")
            print(f"  1. Filtered out by angle threshold: {inactive_faces} faces")
            print(f"  2. Unreachable from seeds: {unlabeled_faces} faces")
        
        return colored_faces == len(mesh.faces)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing mesh segmentation coverage...")
    success = test_segmentation_coverage()
    
    if success:
        print("\n✅ All faces are properly colored!")
    else:
        print("\n❌ Some faces are not colored - this is the bug we're fixing.")
    
    print("\nFixes implemented:")
    print("1. Backend: Better default color for uncolored faces (darker gray)")
    print("2. Backend: Added debugging stats and warnings")
    print("3. Frontend: Fixed mesh coloring to preserve original geometry")
    print("4. Frontend: Better error handling and debugging")
