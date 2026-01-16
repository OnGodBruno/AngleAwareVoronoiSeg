"""
Segmentation service providing high-level mesh processing operations.
"""
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional
import trimesh
import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(src_path))

from models.mesh import MeshData, SegmentationConfig, SegmentationResult
from core.segmentation import MeshSegmenter


class FileConverter:
    """Handles file format conversions for mesh files."""
    
    @staticmethod
    def convert_to_obj(input_path: str, output_path: str) -> None:
        """Convert GLB/GLTF files to OBJ format."""
        mesh = trimesh.load(input_path, process=True)
        mesh.export(output_path, file_type='obj')


class MeshProcessor:
    """Handles mesh loading and preprocessing."""
    
    @staticmethod
    def load_mesh(filepath: str) -> MeshData:
        """Load and preprocess mesh data."""
        mesh = trimesh.load(filepath, process=True)
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
        
        return MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_centers=mesh.triangles_center,
            face_normals=mesh.face_normals,
            face_areas=mesh.area_faces
        )


class MeshSegmentationService:
    """High-level service for mesh segmentation operations."""
    
    def __init__(self):
        self.processor = MeshProcessor()
        self.file_converter = FileConverter()
    
    def segment_mesh_file(self, mesh_path: str, config: SegmentationConfig, 
                         output_dir: str = 'output') -> SegmentationResult:
        """Segment a mesh file and save results."""
        # Load mesh
        mesh_data = self.processor.load_mesh(mesh_path)
        
        # Create segmenter and perform segmentation
        segmenter = MeshSegmenter(config)
        result = segmenter.segment(mesh_data)
        
        # Save segmentation results
        self._save_segmentation_results(mesh_data, result, output_dir)
        
        return result
    
    def create_segments_archive(self, output_dir: str) -> str:
        """Create a ZIP archive of segmentation results."""
        output_path = Path(output_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist")
        
        # Create temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_zip.close()
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_path.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(output_path))
        
        return temp_zip.name
    
    def _save_segmentation_results(self, mesh_data: MeshData, result: SegmentationResult, 
                                  output_dir: str) -> None:
        """Save segmentation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save face labels
        labels_file = output_path / 'face_labels.npy'
        np.save(labels_file, result.face_labels)
        
        # Save seed indices
        seeds_file = output_path / 'seed_indices.npy'
        np.save(seeds_file, result.seed_indices)
        
        # Save statistics
        stats_file = output_path / 'stats.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(result.stats, f, indent=2)
        
        # Optionally save segmented mesh (if trimesh supports it)
        # This would require additional implementation