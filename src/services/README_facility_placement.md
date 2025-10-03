# Automatic Seed Placement Module

This module provides efficient facility placement algorithms for optimal seed selection in mesh segmentation. It implements several algorithms to solve the **p-center problem**: placing k facilities (seeds) to minimize the maximum distance from any point to its nearest facility.

## Features

- **Multiple Algorithms**: Implements various placement strategies from simple heuristics to approximation algorithms
- **Adaptive Strategy Selection**: Automatically chooses the best algorithm based on mesh size and computational constraints  
- **Efficient Implementation**: Uses optimized sparse matrix operations and Dijkstra's algorithm
- **Modular Design**: Clean separation from existing codebase, easy to integrate
- **Time Bounded**: Configurable maximum computation time to prevent hanging on large meshes

## Available Algorithms

### 1. Adaptive Hybrid (Default)
Automatically selects the best strategy based on:
- Mesh size (number of faces)
- Number of seeds requested
- Computational complexity estimates

**Recommendation**: Use this for most cases.

### 2. Greedy Minimax
Exact greedy algorithm that provides optimal results for the p-center problem.
- **Complexity**: O(k × n²) where k = seeds, n = faces
- **Quality**: Best possible (optimal for small meshes)
- **Speed**: Slow for large meshes
- **Use when**: Mesh < 1000 faces and you need the best quality

### 3. Gonzalez Approximation  
Fast 2-approximation algorithm with theoretical guarantees.
- **Complexity**: O(k × n)
- **Quality**: At most 2x optimal distance
- **Speed**: Fast
- **Use when**: Medium meshes (1K-10K faces) with quality requirements

### 4. Farthest First
Simple heuristic that places seeds as far as possible from existing ones.
- **Complexity**: O(k × n) 
- **Quality**: Good in practice, no theoretical guarantee
- **Speed**: Very fast
- **Use when**: Large meshes (>10K faces) or real-time applications

### 5. K-means++ Style
Probabilistic placement inspired by k-means++ initialization.
- **Complexity**: O(k × n)
- **Quality**: Good average case, helps avoid local optima
- **Speed**: Fast
- **Use when**: You want randomization to avoid worst-case scenarios

## Quick Start

### Basic Usage

```python
from src.services.facility_placement import automatic_seed_placement

# Assuming you have an adjacency graph with penalties applied
seed_indices = automatic_seed_placement(
    adjacency_graph=sparse_matrix,  # Your penalty-weighted adjacency matrix
    num_seeds=10,                   # Number of seeds to place
    face_centers=face_centers,      # Optional: face center coordinates
    strategy="adaptive_hybrid"      # Let the algorithm choose the best method
)
```

### Integration with Existing Code

Replace your existing seed selection in `web_interface.py`:

```python
# OLD CODE:
# seed_positions = auto_place_optimal_seeds(mesh, curvature_penalty_strength, num_seeds)

# NEW CODE:  
from src.services.facility_placement import automatic_seed_placement

# Build adjacency graph (existing function)
sparse_matrix, face_centers = build_adjacency_graph(
    mesh, curvature_penalty_strength, user_seeds=None
)

# Use optimal seed placement
seed_indices = automatic_seed_placement(
    adjacency_graph=sparse_matrix,
    num_seeds=num_seeds, 
    face_centers=face_centers
)

# Convert to positions if needed
seed_positions = [face_centers[idx].tolist() for idx in seed_indices]
```

## Algorithm Selection Guide

| Mesh Size | Recommended Strategy | Reasoning |
|-----------|---------------------|-----------|
| < 1,000 faces | `greedy_minimax` | Small enough for optimal algorithm |
| 1K - 10K faces | `gonzalez_approximation` | Good balance of speed and quality |
| > 10K faces | `farthest_first` | Fast enough for interactive use |
| Any size | `adaptive_hybrid` | Let the system choose automatically |

## Configuration Options

```python
from src.services.facility_placement import FacilityPlacer, PlacementConfig, PlacementStrategy

config = PlacementConfig(
    strategy=PlacementStrategy.ADAPTIVE_HYBRID,
    max_computation_time=30.0,      # Maximum seconds to compute
    distance_threshold=1e6,         # Consider points beyond this unreachable
    convergence_tolerance=1e-6,     # Convergence criterion
    random_seed=42,                 # For reproducible results
    verbose=True                    # Print progress information
)

placer = FacilityPlacer(config)
seeds = placer.automatic_seed_placement(sparse_matrix, num_seeds, face_centers)
```

## Performance Characteristics

The algorithms are designed to handle different mesh sizes efficiently:

- **Small meshes** (< 1K faces): All algorithms run quickly, use optimal method
- **Medium meshes** (1K-10K faces): Approximation algorithms provide good results in reasonable time
- **Large meshes** (> 10K faces): Fast heuristics ensure interactive performance
- **Huge meshes** (> 100K faces): Time limits and sampling ensure the system remains responsive

## Example Usage

See `example_facility_placement.py` for a complete working example:

```bash
python example_facility_placement.py
```

This will:
1. Load a test mesh
2. Build the adjacency graph with penalties
3. Test all placement strategies
4. Compare results and performance

## Mathematical Background

The module solves the **p-center problem**:

```
minimize max_{i} min_{j} d(i, S_j)
```

Where:
- `i` ranges over all mesh faces
- `j` ranges over placed seeds  
- `d(i, S_j)` is the geodesic distance with penalties from face i to seed j
- We want to place k seeds to minimize the maximum distance from any face to its nearest seed

This is equivalent to the **facility location problem** in operations research and has applications in:
- Emergency service placement (hospitals, fire stations)
- Supply chain optimization (warehouses, distribution centers)  
- Telecommunications (cell towers, routers)
- Mesh processing (segmentation seeds, sampling points)

## Integration Points

The module integrates with your existing code at these points:

1. **Input**: Takes the same `sparse_matrix` from `build_adjacency_graph()`
2. **Output**: Returns seed indices compatible with `segment_mesh()`  
3. **Configuration**: Uses same parameters (curvature penalties, number of seeds)
4. **Performance**: Respects time limits for interactive use

## Future Enhancements

Potential improvements for future versions:

- **Parallel Processing**: Multi-threaded distance computations for huge meshes
- **Incremental Updates**: Efficiently update seed placement when penalties change
- **Quality Metrics**: Additional measures beyond maximum distance (e.g., variance)
- **Constrained Placement**: Respect user-specified regions or exclusion zones
- **Multi-objective**: Balance multiple criteria (distance, coverage, boundary alignment)

## Troubleshooting

### Common Issues

**1. "Computation time limit exceeded"**
- Increase `max_computation_time` parameter
- Use a faster strategy like `farthest_first`
- The algorithm will return partial results

**2. "Dijkstra computation failed"**  
- Check that your adjacency matrix is valid (symmetric, non-negative weights)
- Ensure the graph is connected or has large connected components
- The algorithm will use fallback methods

**3. Poor seed placement quality**
- Try `greedy_minimax` for small meshes
- Ensure curvature penalties are properly calibrated
- Check that `face_centers` are provided for spatial algorithms

**4. Seeds placed in same location**
- This can happen if the mesh has disconnected components
- Check mesh connectivity with `mesh.is_watertight` and `mesh.body_count`
- Consider preprocessing to handle disconnected regions

### Performance Tips

- **Precompute distances**: If running multiple times with same mesh, cache distance computations
- **Reduce mesh complexity**: Simplify mesh for seed placement, then map back to original
- **Spatial acceleration**: For very large meshes, use spatial data structures for initial filtering
- **Progressive refinement**: Start with few seeds using fast method, then add more with precise method