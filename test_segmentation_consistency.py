#!/usr/bin/env python3
"""
Test script to validate the simple nearest-neighbor fix is working correctly.
This replicates the web interface behavior to ensure consistent coloring.
"""

import requests
import json
import numpy as np

def test_segmentation_consistency():
    """Test the actual web interface to ensure consistent segmentation coloring."""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Segmentation Color Consistency")
    print("=" * 50)
    
    # Test with chair.obj which was having issues
    mesh_file = "input/run/chair.obj"
    
    try:
        # Upload mesh
        print(f"📤 Uploading mesh: {mesh_file}")
        with open(mesh_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/upload_mesh", files=files)
        
        if response.status_code != 200:
            print(f"❌ Failed to upload mesh: {response.status_code}")
            return False
        
        upload_result = response.json()
        if not upload_result.get('success'):
            print(f"❌ Upload failed: {upload_result.get('error')}")
            return False
        
        print(f"✅ Mesh uploaded successfully")
        
        # Perform segmentation with some test seed points
        seed_points = [
            [0.1, 0.1, 0.1],    # Point 1
            [-0.1, 0.1, 0.1],   # Point 2  
            [0.1, -0.1, 0.1],   # Point 3
            [0.0, 0.0, 0.5],    # Point 4
        ]
        
        print(f"🎯 Running segmentation with {len(seed_points)} seed points...")
        
        segmentation_data = {
            'clicked_points': seed_points,
            'output_dir': 'output'
        }
        
        response = requests.post(
            f"{base_url}/segment_with_seeds",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(segmentation_data)
        )
        
        if response.status_code != 200:
            print(f"❌ Segmentation request failed: {response.status_code}")
            return False
        
        result = response.json()
        if not result.get('success'):
            print(f"❌ Segmentation failed: {result.get('error')}")
            return False
        
        # Analyze the face colors
        face_colors = np.array(result['face_colors'])
        total_faces = len(face_colors)
        
        print(f"📊 Segmentation completed: {total_faces} faces processed")
        
        # Check for color consistency
        unique_colors = np.unique(face_colors.reshape(-1, face_colors.shape[-1]), axis=0)
        print(f"🌈 Unique colors found: {len(unique_colors)}")
        
        # Count faces by color
        color_counts = {}
        for color in unique_colors:
            # Find faces with this exact color
            mask = np.all(face_colors == color, axis=1)
            count = np.sum(mask)
            color_key = tuple(color.round(3))  # Round for display
            color_counts[color_key] = count
        
        print(f"📈 Color distribution:")
        for color, count in color_counts.items():
            percentage = (count / total_faces) * 100
            print(f"  Color {color}: {count} faces ({percentage:.1f}%)")
        
        # Check if we have reasonable segmentation (not all one color)
        max_color_percentage = max(color_counts.values()) / total_faces * 100
        
        if max_color_percentage > 95:
            print(f"⚠️  Warning: {max_color_percentage:.1f}% of faces have the same color - segmentation may not be working properly")
        else:
            print(f"✅ Good color distribution - largest segment is {max_color_percentage:.1f}% of faces")
        
        # Check for default gray color (indicating uncolored faces)
        default_gray = (0.2, 0.2, 0.3)
        gray_faces = color_counts.get(default_gray, 0)
        
        if gray_faces > 0:
            print(f"❌ Found {gray_faces} faces with default gray color - not all faces were colored!")
            return False
        else:
            print(f"✅ No uncolored faces found - 100% coverage achieved!")
        
        print(f"✅ Segmentation test PASSED - consistent coloring with {len(unique_colors)} distinct segments")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_segmentation_consistency()
    if success:
        print("\n🎉 All tests passed! The simple nearest-neighbor fix is working correctly.")
    else:
        print("\n💥 Tests failed. There may still be issues with the segmentation coloring.")
