"""
Clean command-line interface for mesh segmentation.
"""
import argparse
import sys
import time
from pathlib import Path

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from models.mesh import SegmentationConfig
from services.segmentation_service import MeshSegmentationService


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="3D Mesh Segmentation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        "mesh_path",
        type=str,
        help="Path to input mesh file (.obj, .glb, .gltf)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory for segmented files"
    )
    
    # Segmentation parameters
    parser.add_argument(
        "--num-seeds", "-n",
        type=int,
        default=10,
        help="Number of segments to create"
    )
    parser.add_argument(
        "--curvature-penalty", "-c",
        type=float,
        default=100.0,
        help="Curvature penalty strength"
    )
    parser.add_argument(
        "--seed-indices",
        type=int,
        nargs="*",
        help="Manual seed face indices (optional)"
    )
    parser.add_argument(
        "--enhanced-mode",
        action="store_true",
        help="Enable enhanced segmentation features"
    )
    
    # Processing options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        mesh_path = Path(args.mesh_path)
        if not mesh_path.exists():
            print(f"Error: Input file {mesh_path} does not exist")
            sys.exit(1)
        
        if mesh_path.suffix.lower() not in ['.obj', '.glb', '.gltf']:
            print(f"Error: Unsupported file format {mesh_path.suffix}")
            sys.exit(1)
        
        # Create configuration
        config = SegmentationConfig(
            curvature_penalty=args.curvature_penalty,
            num_seeds=args.num_seeds,
            enhanced_mode=args.enhanced_mode,
            seed_indices=args.seed_indices
        )
        
        if args.verbose:
            print(f"Segmentation configuration:")
            print(f"  Mesh file: {mesh_path}")
            print(f"  Output directory: {args.output_dir}")
            print(f"  Number of seeds: {config.num_seeds}")
            print(f"  Curvature penalty: {config.curvature_penalty}")
            print(f"  Enhanced mode: {config.enhanced_mode}")
            if config.seed_indices:
                print(f"  Manual seeds: {config.seed_indices}")
            print()
        
        # Initialize service and perform segmentation
        service = MeshSegmentationService()
        
        start_time = time.perf_counter()
        
        print("Starting mesh segmentation...")
        result = service.segment_mesh_file(str(mesh_path), config, args.output_dir)
        
        elapsed_time = time.perf_counter() - start_time
        
        # Print results
        print(f"Segmentation completed in {elapsed_time:.2f} seconds")
        print(f"Results:")
        print(f"  Number of segments: {result.num_segments}")
        print(f"  Coverage ratio: {result.coverage_ratio:.2%}")
        print(f"  Total faces processed: {result.stats['total_faces']}")
        print(f"  Reachable faces: {result.stats['reachable_faces']}")
        print(f"  Unreachable faces: {result.stats['unreachable_faces']}")
        print(f"  Output directory: {args.output_dir}")
        
        if result.stats['unreachable_faces'] > 0:
            unreachable_ratio = result.stats['unreachable_faces'] / result.stats['total_faces']
            if unreachable_ratio > 0.05:  # More than 5% unreachable
                print(f"Warning: {unreachable_ratio:.1%} of faces were unreachable")
                print("Consider adjusting segmentation parameters")
        
        print("Segmentation complete!")
        
    except KeyboardInterrupt:
        print("\nSegmentation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
