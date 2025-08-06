# GLB/GLTF File Support

This mesh segmentation tool now supports GLB and GLTF files in addition to OBJ files.

## Supported File Formats

- **OBJ**: Native support, direct loading
- **GLB**: Converted to OBJ format automatically
- **GLTF**: Converted to OBJ format automatically

## How GLB/GLTF Conversion Works

1. **Upload**: Select a GLB or GLTF file using the file selector
2. **Automatic Conversion**: The server automatically:
   - Loads the GLB/GLTF file using Trimesh
   - Handles scenes with multiple geometries by combining them
   - Cleans up duplicate faces and unreferenced vertices
   - Exports as OBJ format for processing
3. **Processing**: The converted OBJ file is then processed normally for segmentation

## Features

- **Multi-geometry Support**: GLB files with multiple meshes are automatically combined
- **Scene Handling**: Properly extracts geometry from GLB scene objects
- **Error Handling**: Clear error messages for unsupported or corrupted files
- **Progress Feedback**: Visual indication during conversion process

## Technical Details

- Uses Trimesh library for GLB/GLTF loading and conversion
- Preserves mesh geometry while discarding materials and textures
- Automatically combines multiple geometries into a single mesh
- Performs mesh cleanup (duplicate removal, unreferenced vertex removal)
- Converted files are saved as `[filename]_converted.obj` in the uploads directory

## Usage Example

1. Click "Choose File" and select a GLB file
2. You'll see "GLB (will be converted to OBJ)" in the status
3. Click "Upload & Load Mesh" 
4. The system will show "Uploading and converting GLB/GLTF to OBJ format..."
5. Once converted, the mesh will be displayed and ready for segmentation

## Limitations

- Materials and textures are not preserved (only geometry)
- Very large GLB files may take longer to convert
- Complex scenes with many geometries are combined into a single mesh
- Animation data is discarded during conversion

## Troubleshooting

If you encounter issues:
- Ensure the GLB/GLTF file contains valid mesh geometry
- Check that the file is not corrupted
- Try with a simpler GLB file if conversion fails
- Check the browser console for detailed error messages
