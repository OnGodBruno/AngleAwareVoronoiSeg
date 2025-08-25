"""
Clean web interface for mesh segmentation.
"""
import os
import sys
from typing import Optional
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from models.mesh import SegmentationConfig
from services.segmentation_service import MeshSegmentationService, FileConverter


class MeshSegmentationApp:
    """Clean Flask application for mesh segmentation."""
    
    def __init__(self, upload_folder: str = 'uploads', max_file_size: int = 100 * 1024 * 1024):
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = max_file_size
        self.app.config['UPLOAD_FOLDER'] = upload_folder
        
        CORS(self.app)
        
        # Create upload directory
        os.makedirs(upload_folder, exist_ok=True)
        
        # Initialize services
        self.segmentation_service = MeshSegmentationService()
        self.file_converter = FileConverter()
        
        # Current session data (could be moved to session/database)
        self.current_mesh_path: Optional[str] = None
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index_simple.html')
        
        @self.app.route('/upload_mesh', methods=['POST'])
        def upload_mesh():
            return self._handle_upload_mesh()
        
        @self.app.route('/segment_mesh', methods=['POST'])
        def segment_mesh():
            return self._handle_segment_mesh()
        
        @self.app.route('/download_segments')
        def download_segments():
            return self._handle_download_segments()
    
    def _handle_upload_mesh(self):
        """Handle mesh file upload and conversion."""
        try:
            # Validate request
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            # Validate file type
            file_ext = file.filename.lower().split('.')[-1]
            if file_ext not in ['obj', 'glb', 'gltf']:
                return jsonify({'success': False, 'error': 'Only .obj, .glb, and .gltf files are supported'})
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Convert to OBJ if needed
            if file_ext in ['glb', 'gltf']:
                obj_filename = os.path.splitext(filename)[0] + '_converted.obj'
                obj_filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], obj_filename)
                self.file_converter.convert_to_obj(filepath, obj_filepath)
                filepath = obj_filepath
                filename = obj_filename
            
            # Store current mesh path
            self.current_mesh_path = filepath
            
            # Load mesh for preview
            mesh_data = self.segmentation_service.processor.load_mesh(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'vertices': mesh_data.vertices.tolist(),
                'faces': mesh_data.faces.tolist(),
                'face_centers': mesh_data.face_centers.tolist(),
                'total_faces': mesh_data.num_faces,
                'total_vertices': mesh_data.num_vertices
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    def _handle_segment_mesh(self):
        """Handle mesh segmentation request."""
        try:
            if not self.current_mesh_path:
                return jsonify({'success': False, 'error': 'No mesh loaded'})
            
            # Parse parameters
            data = request.json or {}
            config = SegmentationConfig(
                curvature_penalty=float(data.get('curvature_penalty', 100.0)),
                num_seeds=int(data.get('num_seeds', 10)),
                enhanced_mode=bool(data.get('enhanced_mode', False)),
                seed_indices=data.get('seed_indices')
            )
            
            output_dir = data.get('output_dir', 'output')
            
            # Perform segmentation
            result = self.segmentation_service.segment_mesh_file(
                self.current_mesh_path, config, output_dir
            )
            
            return jsonify({
                'success': True,
                'num_segments': result.num_segments,
                'coverage_ratio': result.coverage_ratio,
                'stats': result.stats
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    def _handle_download_segments(self):
        """Handle segment download request."""
        try:
            output_dir = request.args.get('output_dir', 'output')
            
            # Create zip archive
            zip_path = self.segmentation_service.create_segments_archive(output_dir)
            
            return send_file(zip_path, as_attachment=True, download_name='segments.zip')
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    def run(self, debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
        """Run the Flask application."""
        self.app.run(debug=debug, host=host, port=port)


def create_app() -> MeshSegmentationApp:
    """Factory function to create the application."""
    return MeshSegmentationApp()
