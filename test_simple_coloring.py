#!/usr/bin/env python3
"""
Test script to verify the simple nearest-neighbor coloring approach
ensures consistent coloring across segments.
"""

import numpy as np
from scipy.spatial import cKDTree
import trimesh

def test_nearest_neighbor_coloring():
    """Test that nearest neighbor coloring produces consistent results."""
    
    # Load a test mesh using trimesh (which is already in requirements)
    mesh_file = "input/run/chair.obj"
    try:
        mesh = trimesh.load(mesh_file)
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            print(f"❌ Could not load mesh from {mesh_file}")
            return
        
        print(f"✅ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Simulate a segmentation result where only some faces are colored
        total_faces = len(mesh.faces)
        face_colors = np.full((total_faces, 3), [0.2, 0.2, 0.3], dtype=np.float32)  # Default gray
        
        # Simulate that only 30% of faces got colored by the segmentation algorithm
        segmented_count = total_faces // 3
        segmented_faces = np.random.choice(total_faces, segmented_count, replace=False)
        
        # Color these faces with distinct colors (simulating segments)
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green  
            [0.0, 0.0, 1.0],  # Blue
        ])
        
        # Assign random segment colors to segmented faces
        segment_assignments = np.random.randint(0, len(colors), segmented_count)
        face_colors[segmented_faces] = colors[segment_assignments]
        
        # Create mask for which faces are colored
        segmented_mask = np.zeros(total_faces, dtype=bool)
        segmented_mask[segmented_faces] = True
        
        print(f"📊 Initial state: {segmented_count}/{total_faces} faces colored ({(segmented_count/total_faces)*100:.1f}%)")
        
        # Apply the simple nearest neighbor coloring
        uncolored_faces = np.where(~segmented_mask)[0]
        
        if len(uncolored_faces) > 0:
            print(f"🎨 Applying nearest neighbor coloring to {len(uncolored_faces)} uncolored faces...")
            
            # Get face centers
            face_centers = mesh.triangles_center
            
            # Get colored face indices
            colored_faces_indices = np.where(segmented_mask)[0]
            
            # Build KDTree with colored face centers
            colored_centers = face_centers[colored_faces_indices]
            tree = cKDTree(colored_centers)
            
            # Find nearest colored face for each uncolored face
            uncolored_centers = face_centers[uncolored_faces]
            distances, nearest_indices = tree.query(uncolored_centers)
            
            # Assign the same color as the nearest colored face
            nearest_colored_faces = colored_faces_indices[nearest_indices]
            face_colors[uncolored_faces] = face_colors[nearest_colored_faces]
            segmented_mask[uncolored_faces] = True
        
        final_colored = np.sum(segmented_mask)
        print(f"✅ Final result: {final_colored}/{total_faces} faces colored ({(final_colored/total_faces)*100:.1f}%)")
        
        # Verify color consistency
        unique_colors = np.unique(face_colors.reshape(-1, face_colors.shape[-1]), axis=0)
        print(f"🌈 Unique colors in result: {len(unique_colors)}")
        print(f"📈 Expected colors: {len(colors) + 1} (segments + default)")  # +1 for default gray if any
        
        if len(unique_colors) <= len(colors) + 1:
            print("✅ Color consistency check PASSED - no unexpected colors introduced")
        else:
            print("❌ Color consistency check FAILED - unexpected colors found")
            print("Unique colors found:")
            for i, color in enumerate(unique_colors):
                print(f"  {i}: {color}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Simple Nearest-Neighbor Coloring Approach")
    print("=" * 50)
    test_nearest_neighbor_coloring()
