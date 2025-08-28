"""
Clean mesh data model with proper encapsulation.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import trimesh


@dataclass
class MeshData:
    """Clean data container for mesh information."""
    vertices: np.ndarray
    faces: np.ndarray
    face_centers: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    
    @property
    def num_faces(self) -> int:
        return len(self.faces)
    
    @property
    def num_vertices(self) -> int:
        return len(self.vertices)


@dataclass
class SegmentationConfig:
    """Configuration for segmentation parameters."""
    curvature_penalty: float = 100.0
    max_normal_angle: float = np.radians(20)
    num_seeds: int = 10
    enhanced_mode: bool = False
    seed_indices: Optional[list] = None


@dataclass 
class SegmentationResult:
    """Results from mesh segmentation."""
    face_labels: Dict[int, int]
    seed_indices: np.ndarray
    active_faces: np.ndarray
    stats: Dict[str, Any]
    
    @property
    def num_segments(self) -> int:
        return len(self.seed_indices)
    
    @property
    def coverage_ratio(self) -> float:
        """Ratio of faces that were successfully segmented."""
        return len(self.face_labels) / len(self.active_faces) if len(self.active_faces) > 0 else 0.0


class MeshProcessor:
    """Clean mesh processing with proper error handling."""
    
    @staticmethod
    def load_mesh(file_path: str) -> MeshData:
        """Load and clean a mesh file."""
        try:
            mesh = trimesh.load(file_path, process=True)
            mesh.remove_unreferenced_vertices()
            mesh.remove_infinite_values()
            
            return MeshData(
                vertices=mesh.vertices,
                faces=mesh.faces,
                face_centers=mesh.triangles_center,
                face_normals=mesh.face_normals,
                face_areas=mesh.area_faces
            )
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {file_path}: {e}")
    
    @staticmethod
    def export_segments(mesh_data: MeshData, result: SegmentationResult, output_dir: str) -> None:
        """Export segmented mesh parts to separate files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create trimesh object for exporting
        mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces)
        
        # Group faces by segment
        segments = {}
        for face_idx, segment_id in result.face_labels.items():
            if segment_id not in segments:
                segments[segment_id] = []
            segments[segment_id].append(result.active_faces[face_idx])
        
        # Export each segment
        for segment_id, face_indices in segments.items():
            if face_indices:
                output_path = os.path.join(output_dir, f"segment_{segment_id}.obj")
                mesh.submesh([face_indices], append=True).export(output_path)
