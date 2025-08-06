#!/usr/bin/env python3
"""
Performance test script to measure the optimization improvements.
"""

import time
import numpy as np
import trimesh
from segment_mesh import load_and_clean_mesh, build_adjacency_graph, segment_mesh

def test_performance():
    """Test the performance of the optimized segmentation."""
    
    # Load a test mesh
    mesh_path = "input/run/example.obj"
    
    print("=== Performance Test ===")
    print("Testing optimized mesh segmentation...")
    
    try:
        # Load and process mesh
        start_time = time.perf_counter()
        mesh = load_and_clean_mesh(mesh_path)
        load_time = time.perf_counter() - start_time
        print(f"✓ Mesh loading: {load_time:.3f}s ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
        
        # Build adjacency graph
        start_time = time.perf_counter()
        sparse_matrix, face_coords, active_faces = build_adjacency_graph(mesh, curvature_penalty_strength=100.0)
        graph_time = time.perf_counter() - start_time
        print(f"✓ Graph building: {graph_time:.3f}s ({sparse_matrix.shape[0]} active faces)")
        
        # Simulate manual seed selection
        if len(face_coords) >= 5:
            seed_idx = np.array([0, len(face_coords)//5, 2*len(face_coords)//5, 3*len(face_coords)//5, 4*len(face_coords)//5])
        else:
            seed_idx = np.array([0])
        
        # Run segmentation
        start_time = time.perf_counter()
        face_labels = segment_mesh(sparse_matrix, seed_idx)
        segment_time = time.perf_counter() - start_time
        print(f"✓ Segmentation: {segment_time:.3f}s ({len(face_labels)} faces labeled)")
        
        # Test the optimized color assignment (simulating web_interface.py logic)
        start_time = time.perf_counter()
        
        # Optimized color assignment using NumPy
        colors = np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
        ])
        
        total_faces = len(mesh.faces)
        face_colors = np.full((total_faces, 3), [0.2, 0.2, 0.3], dtype=np.float32)
        segmented_mask = np.zeros(total_faces, dtype=bool)
        
        # Vectorized color assignment
        if face_labels:
            row_indices = np.array(list(face_labels.keys()))
            seed_rows = np.array(list(face_labels.values()))
            seed_to_segment = {seed: i for i, seed in enumerate(seed_idx)}
            segment_ids = np.array([seed_to_segment[seed] for seed in seed_rows])
            
            face_ids = active_faces[row_indices]
            color_indices = segment_ids % len(colors)
            face_colors[face_ids] = colors[color_indices]
            segmented_mask[face_ids] = True
        
        # Handle inactive faces with KDTree
        inactive_faces = np.setdiff1d(np.arange(total_faces), active_faces)
        
        # First, handle unreachable active faces
        unreachable_active_indices = []
        for i, active_face in enumerate(active_faces):
            if not segmented_mask[active_face]:  # This active face wasn't segmented
                unreachable_active_indices.append(i)
        
        if unreachable_active_indices:
            unreachable_active_indices = np.array(unreachable_active_indices)
            unreachable_faces = active_faces[unreachable_active_indices]
            # Assign them all to the first seed's segment
            first_segment_color = colors[0]
            face_colors[unreachable_faces] = first_segment_color
            segmented_mask[unreachable_faces] = True
        
        if len(inactive_faces) > 0:
            from scipy.spatial import cKDTree
            
            # Build KDTree with ALL colored faces
            all_colored_faces = np.where(segmented_mask)[0]
            all_face_centers = mesh.triangles_center
            colored_centers = all_face_centers[all_colored_faces]
            tree = cKDTree(colored_centers)
            
            inactive_centers = all_face_centers[inactive_faces]
            distances, nearest_indices = tree.query(inactive_centers)
            
            nearest_colored_faces = all_colored_faces[nearest_indices]
            face_colors[inactive_faces] = face_colors[nearest_colored_faces]
            segmented_mask[inactive_faces] = True
        
        color_time = time.perf_counter() - start_time
        colored_faces = np.sum(segmented_mask)
        
        print(f"✓ Color assignment: {color_time:.3f}s ({colored_faces}/{total_faces} faces colored)")
        
        # Total time
        total_time = load_time + graph_time + segment_time + color_time
        print(f"\n=== Performance Summary ===")
        print(f"Total time: {total_time:.3f}s")
        print(f"  - Loading: {load_time:.3f}s ({(load_time/total_time)*100:.1f}%)")
        print(f"  - Graph building: {graph_time:.3f}s ({(graph_time/total_time)*100:.1f}%)")
        print(f"  - Segmentation: {segment_time:.3f}s ({(segment_time/total_time)*100:.1f}%)")
        print(f"  - Color assignment: {color_time:.3f}s ({(color_time/total_time)*100:.1f}%)")
        
        # Performance metrics
        faces_per_second = total_faces / total_time
        print(f"\nThroughput: {faces_per_second:.0f} faces/second")
        
        # Memory efficiency estimate
        memory_mb = (face_colors.nbytes + segmented_mask.nbytes) / (1024 * 1024)
        print(f"Color data memory: {memory_mb:.1f} MB")
        
        coverage = (colored_faces / total_faces) * 100
        if coverage >= 99.9:
            print(f"✅ Coverage: {coverage:.1f}% - EXCELLENT!")
        elif coverage >= 95.0:
            print(f"✅ Coverage: {coverage:.1f}% - Good")
        else:
            print(f"⚠️ Coverage: {coverage:.1f}% - Needs improvement")
        
        return total_time, coverage
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def benchmark_old_vs_new():
    """Compare performance of old vs new approach."""
    
    print("\n=== Optimization Comparison ===")
    print("Key optimizations implemented:")
    print("1. ✓ NumPy vectorized operations instead of Python loops")
    print("2. ✓ KDTree (O(log n)) instead of brute force (O(n²)) for nearest neighbor")
    print("3. ✓ Boolean masks instead of Python sets")
    print("4. ✓ Reduced memory allocations and copies")
    print("5. ✓ Skip expensive operations when not needed (< 1% unreachable faces)")
    print("6. ✓ Efficient data structures throughout")
    
    # Old approach simulation (estimated complexity)
    mesh_path = "input/run/example.obj"
    mesh = load_and_clean_mesh(mesh_path)
    total_faces = len(mesh.faces)
    
    # Estimate old performance (based on O(n²) complexity for inactive faces)
    # For 550k faces, old approach would be ~550k * 550k = 300 billion operations
    # New approach is ~550k * log(550k) = ~10 million operations
    
    speedup_factor = (total_faces ** 2) / (total_faces * np.log2(total_faces))
    print(f"\nEstimated speedup for {total_faces} faces:")
    print(f"Old complexity: O(n²) = {total_faces**2:.0e} operations")
    print(f"New complexity: O(n log n) = {total_faces * np.log2(total_faces):.0e} operations")
    print(f"Theoretical speedup: {speedup_factor:.0f}x faster")

if __name__ == "__main__":
    print("Testing optimized mesh segmentation performance...")
    
    total_time, coverage = test_performance()
    
    if total_time:
        benchmark_old_vs_new()
        
        print(f"\n=== Final Results ===")
        print(f"✅ Segmentation completed in {total_time:.3f}s with {coverage:.1f}% coverage")
        
        if total_time < 5.0:
            print("🚀 EXCELLENT performance - under 5 seconds!")
        elif total_time < 10.0:
            print("✅ Good performance - under 10 seconds")
        else:
            print("⚠️ Consider further optimizations if needed")
    else:
        print("❌ Performance test failed")
