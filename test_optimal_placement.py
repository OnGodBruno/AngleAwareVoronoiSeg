import requests
import json

def test_optimal_placement():
    """Test the optimal seed placement functionality"""
    
    # First load a mesh
    print("Loading mesh...")
    load_response = requests.post('http://127.0.0.1:5000/load_mesh', 
                                 json={'mesh_path': 'input/run/example.obj', 'curvature_penalty_strength': 100.0})
    
    if load_response.status_code != 200:
        print(f"Error loading mesh: {load_response.status_code}")
        return
    
    load_result = load_response.json()
    if not load_result.get('success', False):
        print(f"Mesh loading failed: {load_result.get('error', 'Unknown error')}")
        return
        
    print(f"Mesh loaded successfully: {load_result.get('total_faces', 0)} faces")
    
    # Now test optimal seed placement
    print("Testing optimal seed placement...")
    placement_response = requests.post('http://127.0.0.1:5000/auto_place_seeds',
                                      json={'num_seeds': 3})
    
    if placement_response.status_code != 200:
        print(f"Error in optimal placement request: {placement_response.status_code}")
        return
    
    placement_result = placement_response.json()
    if placement_result.get('success', False):
        seed_positions = placement_result.get('seed_positions', [])
        print(f"✅ Optimal placement successful! Placed {len(seed_positions)} seeds")
        print(f"Algorithm info: {placement_result.get('algorithm_info', 'N/A')}")
        for i, pos in enumerate(seed_positions):
            print(f"  Seed {i+1}: {pos}")
    else:
        error = placement_result.get('error', 'Unknown error')
        print(f"❌ Optimal placement failed: {error}")

if __name__ == "__main__":
    test_optimal_placement()