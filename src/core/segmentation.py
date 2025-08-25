"""
Clean core segmentation algorithms.
"""
import numpy as np
import sys
from scipy.sparse import csgraph
import scipy.sparse as sparse
from typing import Tuple, Optional, List
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent  
sys.path.insert(0, str(src_path))

from models.mesh import MeshData, SegmentationConfig, SegmentationResult


class DistanceCalculator:
    """Handles different distance metric calculations."""
    
    @staticmethod
    def spatial_distance(face_centers: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Calculate spatial distance between adjacent faces."""
        p1 = face_centers[adjacency[:, 0]]
        p2 = face_centers[adjacency[:, 1]]
        return np.linalg.norm(p1 - p2, axis=1)
    
    @staticmethod
    def normal_angle(face_normals: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Calculate angle between face normals."""
        n1 = face_normals[adjacency[:, 0]]
        n2 = face_normals[adjacency[:, 1]]
        return np.arccos(np.einsum('ij,ij->i', n1, n2).clip(-1, 1))
    
    @staticmethod
    def area_ratio(face_areas: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Calculate area ratio difference between faces."""
        a1 = face_areas[adjacency[:, 0]]
        a2 = face_areas[adjacency[:, 1]]
        return np.abs(a1 - a2) / (a1 + a2 + 1e-8)


class GraphBuilder:
    """Builds adjacency graphs for mesh segmentation."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.distance_calc = DistanceCalculator()
    
    def build_adjacency_graph(self, mesh_data: MeshData) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        """
        Build face adjacency graph with weighted edges.
        
        Returns:
            - Weighted sparse adjacency matrix
            - Face coordinates of active faces  
            - Array of active face indices
        """
        try:
            # Get face adjacency from trimesh
            import trimesh
            temp_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces)
            adjacency = temp_mesh.face_adjacency
            
            # Calculate distances
            spatial_dist = self.distance_calc.spatial_distance(mesh_data.face_centers, adjacency)
            normal_angles = self.distance_calc.normal_angle(mesh_data.face_normals, adjacency)
            
            # Filter by angle threshold
            angle_mask = normal_angles <= self.config.max_normal_angle
            valid_adjacency = adjacency[angle_mask]
            
            # Calculate edge weights
            weights = self._calculate_weights(
                spatial_dist[angle_mask], 
                normal_angles[angle_mask],
                mesh_data,
                valid_adjacency
            )
            
            # Get active faces (faces that have valid adjacencies)
            active_faces = np.unique(valid_adjacency.flatten())
            face_coords = mesh_data.face_centers[active_faces]
            
            # Build sparse matrix
            sparse_matrix = self._build_sparse_matrix(valid_adjacency, weights, active_faces)
            
            return sparse_matrix, face_coords, active_faces
            
        except Exception as e:
            raise RuntimeError(f"Failed to build adjacency graph: {e}")
    
    def _calculate_weights(self, spatial_dist: np.ndarray, angles: np.ndarray, 
                          mesh_data: MeshData, adjacency: np.ndarray) -> np.ndarray:
        """Calculate edge weights based on configuration."""
        # Basic weight calculation
        avg_edge_length = np.mean(spatial_dist) if len(spatial_dist) > 0 else 1.0
        
        curvature_penalty = np.exp(self.config.curvature_penalty * angles)
        spatial_penalty = 1 + (spatial_dist / avg_edge_length) ** 2
        
        weights = spatial_penalty * curvature_penalty
        
        # Add enhanced features if enabled
        if self.config.enhanced_mode:
            area_ratios = self.distance_calc.area_ratio(mesh_data.face_areas, adjacency)
            area_penalty = 1 + area_ratios * 0.5
            weights *= area_penalty
        
        return weights
    
    def _build_sparse_matrix(self, adjacency: np.ndarray, weights: np.ndarray, 
                           active_faces: np.ndarray) -> sparse.csr_matrix:
        """Build symmetric sparse adjacency matrix."""
        # Map face indices to matrix indices
        row = np.searchsorted(active_faces, adjacency[:, 0])
        col = np.searchsorted(active_faces, adjacency[:, 1])
        
        # Make symmetric
        all_row = np.concatenate([row, col])
        all_col = np.concatenate([col, row])
        all_weights = np.concatenate([weights, weights]).astype(np.float64)
        
        n_faces = len(active_faces)
        return sparse.csr_matrix(
            (all_weights, (all_row, all_col)),
            shape=(n_faces, n_faces),
            dtype=np.float64
        )


class SeedSelector:
    """Handles seed selection for segmentation."""
    
    @staticmethod
    def select_farthest_point_seeds(face_coords: np.ndarray, n_seeds: int, 
                                   random_seed: int = 42) -> np.ndarray:
        """Select seeds using farthest-point sampling algorithm."""
        if len(face_coords) == 0:
            return np.array([], dtype=int)
        
        if n_seeds >= len(face_coords):
            return np.arange(len(face_coords))
        
        rng = np.random.default_rng(random_seed)
        
        # Pick first seed randomly from a pool
        pool_size = min(64, len(face_coords))
        pool = rng.choice(len(face_coords), size=pool_size, replace=False)
        sub_coords = face_coords[pool]
        distances = np.linalg.norm(sub_coords[:, None] - sub_coords[None], axis=2)
        first_seed = pool[np.argmax(distances.sum(axis=1))]
        
        seed_indices = [first_seed]
        min_distances = np.linalg.norm(face_coords - face_coords[first_seed], axis=1)
        
        # Iteratively select farthest points
        for _ in range(1, n_seeds):
            probs = min_distances / min_distances.sum()
            new_seed = rng.choice(len(face_coords), p=probs)
            seed_indices.append(new_seed)
            
            new_distances = np.linalg.norm(face_coords - face_coords[new_seed], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
        
        return np.array(seed_indices)


class MeshSegmenter:
    """Main segmentation algorithm coordinator."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.graph_builder = GraphBuilder(config)
        self.seed_selector = SeedSelector()
    
    def segment(self, mesh_data: MeshData) -> SegmentationResult:
        """Perform complete mesh segmentation."""
        try:
            # Build adjacency graph
            sparse_matrix, face_coords, active_faces = self.graph_builder.build_adjacency_graph(mesh_data)
            
            # Select or use provided seeds
            if self.config.seed_indices:
                seed_indices = np.array(self.config.seed_indices, dtype=int)
            else:
                seed_indices = self.seed_selector.select_farthest_point_seeds(
                    face_coords, self.config.num_seeds
                )
            
            # Perform segmentation
            face_labels = self._dijkstra_segmentation(sparse_matrix, seed_indices)
            
            # Collect statistics
            stats = self._collect_stats(face_labels, sparse_matrix, active_faces)
            
            return SegmentationResult(
                face_labels=face_labels,
                seed_indices=seed_indices,
                active_faces=active_faces,
                stats=stats
            )
            
        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {e}")
    
    def _dijkstra_segmentation(self, sparse_matrix: sparse.csr_matrix, 
                              seed_indices: np.ndarray) -> dict:
        """Perform multi-source Dijkstra segmentation."""
        if len(seed_indices) == 0:
            return {}
        
        # Multi-source shortest path
        distances = csgraph.dijkstra(
            sparse_matrix, 
            indices=seed_indices,
            directed=False, 
            return_predecessors=False
        )
        
        # Handle ties by preferring earlier seeds
        eps = np.linspace(0.0, 1e-9, len(seed_indices), endpoint=False)[:, None]
        winners = (distances + eps).argmin(axis=0)
        
        # Create face labels for reachable faces
        face_labels = {}
        for i in range(sparse_matrix.shape[0]):
            if np.isfinite(distances[winners[i], i]):
                face_labels[i] = seed_indices[winners[i]]
        
        return face_labels
    
    def _collect_stats(self, face_labels: dict, sparse_matrix: sparse.csr_matrix, 
                      active_faces: np.ndarray) -> dict:
        """Collect segmentation statistics."""
        total_faces = sparse_matrix.shape[0]
        reachable_faces = len(face_labels)
        unreachable_faces = total_faces - reachable_faces
        
        return {
            'total_faces': total_faces,
            'reachable_faces': reachable_faces,
            'unreachable_faces': unreachable_faces,
            'coverage_ratio': reachable_faces / total_faces if total_faces > 0 else 0.0,
            'num_active_faces': len(active_faces)
        }
