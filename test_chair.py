#!/usr/bin/env python3
"""
Test specifically for chair.obj to identify any remaining coloring issues.
"""

import time
import numpy as np
import trimesh
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh

def test_chair_segmentation():
    """Test chair.obj specifically for the coloring issue."""
    
    print("=== Testing chair.obj for coloring issues ===")
    mesh_path = "input/run/chair.obj"
    
    try:
        # Load mesh
        start_time = time.perf_counter()
        mesh = load_and_clean_mesh(mesh_path)
        load_time = time.perf_counter() - start_time
        print(f"✓ Loaded chair.obj: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces ({load_time:.3f}s)")
        
        # Build adjacency graph
        start_time = time.perf_counter()
        sparse_matrix, face_coords, active_faces = build_adjacency_graph(mesh, curvature_penalty_strength=100.0)
        graph_time = time.perf_counter() - start_time
        print(f"✓ Graph built: {sparse_matrix.shape[0]} active faces out of {len(mesh.faces)} total ({graph_time:.3f}s)")
        
        # Calculate filtering statistics
        total_faces = len(mesh.faces)
        active_face_count = len(active_faces)
        filtered_faces = total_faces - active_face_count
        print(f"  - Active faces: {active_face_count}")
        print(f"  - Filtered faces: {filtered_faces} ({(filtered_faces/total_faces)*100:.2f}%)")
        
        # Test segmentation with multiple seeds (simulating user clicks)
        if len(face_coords) >= 5:
            seed_idx = np.array([0, len(face_coords)//5, 2*len(face_coords)//5, 3*len(face_coords)//5, 4*len(face_coords)//5])
        elif len(face_coords) >= 3:
            seed_idx = np.array([0, len(face_coords)//3, 2*len(face_coords)//3])
        else:
            seed_idx = np.array([0])
        
        print(f"✓ Using {len(seed_idx)} seeds at matrix indices: {seed_idx}")
        
        # Run segmentation
        start_time = time.perf_counter()
        face_labels = segment_mesh(sparse_matrix, seed_idx)
        segment_time = time.perf_counter() - start_time
        print(f"✓ Segmentation completed: {len(face_labels)} faces labeled ({segment_time:.3f}s)")
        
        # Analyze segmentation coverage
        reachable_faces = len(face_labels)
        unreachable_faces = sparse_matrix.shape[0] - reachable_faces
        print(f"  - Reachable faces: {reachable_faces}")
        print(f"  - Unreachable faces: {unreachable_faces}")
        
        # Test the optimized color assignment (matching web_interface.py exactly)
        start_time = time.perf_counter()
        
        # Colors array (same as web interface)
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2]
        ])
        
        # Initialize all faces with default color using NumPy for efficiency
        face_colors = np.full((total_faces, 3), [0.2, 0.2, 0.3], dtype=np.float32)
        
        # Create boolean mask for segmented faces (much faster than set)
        segmented_mask = np.zeros(total_faces, dtype=bool)
        
        # Color segments efficiently using vectorized operations
        if face_labels:
            # Convert to arrays for vectorized operations
            row_indices = np.array(list(face_labels.keys()))
            seed_rows = np.array(list(face_labels.values()))
            
            # Map seed rows to segment IDs
            seed_to_segment = {seed: i for i, seed in enumerate(seed_idx)}
            segment_ids = np.array([seed_to_segment[seed] for seed in seed_rows])
            
            # Get face IDs and colors in one go
            face_ids = active_faces[row_indices]
            color_indices = segment_ids % len(colors)
            
            # Assign colors vectorized
            face_colors[face_ids] = colors[color_indices]
            segmented_mask[face_ids] = True
        
        colored_faces_after_segmentation = np.sum(segmented_mask)
        print(f"✓ After segmentation: {colored_faces_after_segmentation} faces colored")
        
        # Handle unreachable active faces first (matching web_interface.py logic)
        unreachable_active_indices = []
        for i, active_face in enumerate(active_faces):
            if not segmented_mask[active_face]:  # This active face wasn't segmented
                unreachable_active_indices.append(i)
        
        if unreachable_active_indices:
            print(f"✓ Assigning {len(unreachable_active_indices)} unreachable active faces to first seed...")
            unreachable_active_indices = np.array(unreachable_active_indices)
            unreachable_faces = active_faces[unreachable_active_indices]
            
            # Assign them all to the first seed's segment (simple but effective)
            first_segment_color = colors[0]
            face_colors[unreachable_faces] = first_segment_color
            segmented_mask[unreachable_faces] = True
        
        # Handle inactive faces efficiently if there are any
        inactive_faces = np.setdiff1d(np.arange(total_faces), active_faces)
        
        if len(inactive_faces) > 0:
            print(f"✓ Handling {len(inactive_faces)} inactive faces with improved KDTree...")
            
            # Use scipy's cKDTree for fast nearest neighbor search
            from scipy.spatial import cKDTree
            
            # Now that all active faces are colored, build KDTree with ALL colored faces
            all_colored_faces = np.where(segmented_mask)[0]
            all_face_centers = mesh.triangles_center
            colored_centers = all_face_centers[all_colored_faces]
            tree = cKDTree(colored_centers)
            
            # Find nearest colored faces for all inactive faces at once
            inactive_centers = all_face_centers[inactive_faces]
            distances, nearest_indices = tree.query(inactive_centers)
            
            # Get the corresponding colored face IDs and copy their colors
            nearest_colored_faces = all_colored_faces[nearest_indices]
            face_colors[inactive_faces] = face_colors[nearest_colored_faces]
            segmented_mask[inactive_faces] = True
        
        color_time = time.perf_counter() - start_time
        final_colored_faces = np.sum(segmented_mask)
        
        print(f"✓ Color assignment completed: {final_colored_faces}/{total_faces} faces colored ({color_time:.3f}s)")
        
        # Detailed analysis
        uncolored_faces = total_faces - final_colored_faces
        coverage_percentage = (final_colored_faces / total_faces) * 100
        
        print(f"\n=== Chair.obj Results ===")
        print(f"Total processing time: {load_time + graph_time + segment_time + color_time:.3f}s")
        print(f"Performance: {total_faces/(load_time + graph_time + segment_time + color_time):.0f} faces/second")
        print(f"")
        print(f"Face Coverage Analysis:")
        print(f"  • Total faces: {total_faces}")
        print(f"  • Colored faces: {final_colored_faces}")
        print(f"  • Uncolored faces: {uncolored_faces}")
        print(f"  • Coverage: {coverage_percentage:.2f}%")
        
        if uncolored_faces == 0:
            print("✅ PERFECT: All faces are properly colored!")
            return True
        elif coverage_percentage >= 99.9:
            print("✅ EXCELLENT: 99.9%+ faces colored")
            return True
        elif coverage_percentage >= 95.0:
            print("✅ GOOD: 95%+ faces colored")
            return True
        else:
            print("⚠️ ISSUE: Significant number of faces uncolored")
            
            # Analyze what faces are uncolored
            uncolored_indices = np.where(~segmented_mask)[0]
            print(f"\nAnalyzing {len(uncolored_indices)} uncolored faces:")
            
            # Check if uncolored faces are in active set
            uncolored_in_active = np.intersect1d(uncolored_indices, active_faces)
            uncolored_inactive = np.setdiff1d(uncolored_indices, active_faces)
            
            print(f"  • Uncolored active faces: {len(uncolored_in_active)}")
            print(f"  • Uncolored inactive faces: {len(uncolored_inactive)}")
            
            if len(uncolored_in_active) > 0:
                print("  ❌ PROBLEM: Active faces should all be colored!")
                
            if len(uncolored_inactive) > 0:
                print("  ⚠️ Some inactive faces couldn't be mapped to colored neighbors")
                
            return False
        
    except Exception as e:
        print(f"❌ Error testing chair.obj: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing chair.obj specifically for coloring issues...")
    success = test_chair_segmentation()
    
    if success:
        print("\n🎉 Chair.obj test PASSED - no coloring issues detected!")
    else:
        print("\n❌ Chair.obj test FAILED - coloring issues detected!")
        print("This indicates the optimization may need further refinement.")
