// Global variables
        let scene, camera, renderer, controls;
        let meshObject = null;
        let selectedSeeds = [];
        let seedSelectionMode = true; // Always enabled
        let meshData = null;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();
        let mouseDown = false;
        let renderErrorCount = 0;
        let maxRenderErrors = 10;
        let currentSeedColor = 'red'; // Default seed color
        let segmentationMode = 'manual'; // 'manual', 'automatic', or 'geodesic'
        
        // Color mapping for seeds
        const seedColors = {
            'red': [1.0, 0.0, 0.0],
            'green': [0.0, 1.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'yellow': [1.0, 1.0, 0.0],
            'magenta': [1.0, 0.0, 1.0],
            'cyan': [0.0, 1.0, 1.0],
            'orange': [1.0, 0.5, 0.0],
            'purple': [0.5, 0.0, 1.0]
        };
        
        // Simple orbit controls implementation
        class SimpleOrbitControls {
            constructor(camera, domElement) {
                this.camera = camera;
                this.domElement = domElement;
                this.isUserInteracting = false;
                this.rotateSpeed = 1.0;
                this.zoomSpeed = 0.1; // Reduced zoom speed
                this.onMouseDown = this.onMouseDown.bind(this);
                this.onMouseMove = this.onMouseMove.bind(this);
                this.onMouseUp = this.onMouseUp.bind(this);
                this.onWheel = this.onWheel.bind(this);
                
                this.spherical = new THREE.Spherical();
                this.sphericalDelta = new THREE.Spherical();
                this.target = new THREE.Vector3();
                
                this.domElement.addEventListener('mousedown', this.onMouseDown);
                this.domElement.addEventListener('mousemove', this.onMouseMove);
                this.domElement.addEventListener('mouseup', this.onMouseUp);
                this.domElement.addEventListener('wheel', this.onWheel, { passive: false });
                
                this.update();
            }
            
            onMouseDown(event) {
                this.isUserInteracting = true;
                this.mouseX = event.clientX;
                this.mouseY = event.clientY;
            }
            
            onMouseMove(event) {
                if (!this.isUserInteracting) return;
                
                const deltaX = event.clientX - this.mouseX;
                const deltaY = event.clientY - this.mouseY;
                
                this.sphericalDelta.theta -= deltaX * 0.01;
                this.sphericalDelta.phi -= deltaY * 0.01;
                
                this.mouseX = event.clientX;
                this.mouseY = event.clientY;
                
                this.update();
            }
            
            onMouseUp() {
                this.isUserInteracting = false;
            }
            
            onWheel(event) {
                event.preventDefault(); // Prevent page scroll
                const distance = this.camera.position.distanceTo(this.target);
                const delta = event.deltaY * this.zoomSpeed * 0.001 * distance;
                
                // Limit zoom to prevent going too close or too far
                const newDistance = distance + delta;
                if (newDistance > 0.1 && newDistance < 100) {
                    this.camera.position.multiplyScalar((distance + delta) / distance);
                    this.update();
                }
            }
            
            update() {
                this.spherical.setFromVector3(this.camera.position.clone().sub(this.target));
                this.spherical.theta += this.sphericalDelta.theta;
                this.spherical.phi += this.sphericalDelta.phi;
                this.spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.spherical.phi));
                
                this.camera.position.setFromSpherical(this.spherical).add(this.target);
                this.camera.lookAt(this.target);
                
                this.sphericalDelta.set(0, 0, 0);
            }
        }
        
        // Debug function
        function updateDebugInfo(text) {
            const debugInfo = document.getElementById('debugInfo');
            if (debugInfo) {
                debugInfo.innerHTML = `<i class="fas fa-terminal"></i> ${text}`;
            }
            console.log(text);
        }
        
        // Initialize Three.js scene
        function initThreeJS() {
            try {
                updateDebugInfo('Initializing Three.js...');
                const viewer = document.getElementById('viewer');
                
                if (!viewer) {
                    throw new Error('Viewer element not found');
                }
                
                // Scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x222222);
                updateDebugInfo('Scene created');
                
                // Camera
                camera = new THREE.PerspectiveCamera(75, viewer.clientWidth / viewer.clientHeight, 0.1, 1000);
                camera.position.set(0, 0, 5);
                updateDebugInfo('Camera created');
                
                // Renderer
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(viewer.clientWidth, viewer.clientHeight);
                renderer.setClearColor(0x222222);
                viewer.appendChild(renderer.domElement);
                updateDebugInfo('Renderer created and added to DOM');
                
                // Simple controls
                controls = new SimpleOrbitControls(camera, renderer.domElement);
                updateDebugInfo('Controls initialized');
                
                // Lighting - more comprehensive setup
                const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
                scene.add(ambientLight);
                
                const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight1.position.set(1, 1, 1);
                scene.add(directionalLight1);
                
                const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
                directionalLight2.position.set(-1, -1, -1);
                scene.add(directionalLight2);
                
                updateDebugInfo('Enhanced lighting added');
                
                // Event listeners
                renderer.domElement.addEventListener('click', onMouseClick);
                window.addEventListener('resize', onWindowResize);
                
                // Animation loop
                animate();
                updateDebugInfo('Three.js initialization complete');
                
                // Add a test cube to verify rendering
                const geometry = new THREE.BoxGeometry(1, 1, 1);
                const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
                const testCube = new THREE.Mesh(geometry, material);
                scene.add(testCube);
                updateDebugInfo('Test cube added - you should see a green cube');
                
            } catch (error) {
                updateDebugInfo('Error initializing Three.js: ' + error.message);
                showStatus('Error initializing 3D viewer: ' + error.message, 'error');
            }
        }
        
        function restartAnimation() {
            renderErrorCount = 0;
            updateDebugInfo('Restarting animation loop');
            animate();
        }
        
        function clearScene() {
            if (meshObject) {
                scene.remove(meshObject);
                if (meshObject.geometry) {
                    meshObject.geometry.dispose();
                }
                if (meshObject.material) {
                    if (meshObject.material.map) meshObject.material.map.dispose();
                    meshObject.material.dispose();
                }
                meshObject = null;
                updateDebugInfo('Scene cleared');
            }
        }
        
        function animate() {
            try {
                if (renderer && scene && camera) {
                    renderer.render(scene, camera);
                    renderErrorCount = 0; // Reset error count on successful render
                }
            } catch (renderError) {
                renderErrorCount++;
                console.error('Render error:', renderError);
                updateDebugInfo(`Render error ${renderErrorCount}: ${renderError.message}`);
                
                // If too many errors, stop the animation loop and try to recover
                if (renderErrorCount >= maxRenderErrors) {
                    updateDebugInfo('Too many render errors, stopping animation loop');
                    return; // Stop the animation loop
                }
            }
            
            // Continue animation loop only if not too many errors
            if (renderErrorCount < maxRenderErrors) {
                requestAnimationFrame(animate);
            }
        }
        
        function selectSeedColor(color, element) {
            currentSeedColor = color;
            
            // Update UI to show active color
            document.querySelectorAll('.color-option').forEach(option => {
                option.classList.remove('active');
            });
            element.classList.add('active');
            
            showStatus(`Selected ${color} color for new seeds`, 'info');
        }
        
        function setSegmentationMode(mode) {
            segmentationMode = mode;
        
            const manualBtn = document.getElementById('manualModeBtn');
            const automaticBtn = document.getElementById('automaticModeBtn');
            const geodesicBtn = document.getElementById('geodesicModeBtn');
        
            const manualControls = document.getElementById('manualControls');
            const automaticControls = document.getElementById('automaticControls');
            const geodesicControls = document.getElementById('geodesicControls');
        
            // Reset all buttons and hide all panels
            [manualBtn, automaticBtn, geodesicBtn].forEach(btn => btn.classList.remove('active'));
            [manualControls, automaticControls, geodesicControls].forEach(panel => panel.style.display = 'none');
        
            if (mode === 'manual') {
                manualBtn.classList.add('active');
                manualControls.style.display = 'block';
                seedSelectionMode = true;
            } else if (mode === 'automatic') {
                automaticBtn.classList.add('active');
                automaticControls.style.display = 'block';
                seedSelectionMode = false;
                clearSeeds();
            } else if (mode === 'geodesic') {
                geodesicBtn.classList.add('active');
                geodesicControls.style.display = 'block';
                seedSelectionMode = false; // Disable seed placement
            }
        
            updateSegmentButtonState();
            showStatus(`Switched to ${mode} mode`, 'info');
        }

        async function autoPlaceSeeds() {
            console.log('autoPlaceSeeds function called!');
            console.log('meshObject:', meshObject);
            console.log('meshData:', meshData);
            
            if (!meshObject || !meshData) {
                console.log('No mesh loaded - showing error');
                showStatus('Please load a mesh first', 'error');
                return;
            }
            
            const autoSeedCount = parseInt(document.getElementById('autoSeedCount').value);
            
            if (autoSeedCount < 1 || autoSeedCount > 20) {
                showStatus('Number of seeds must be between 1 and 20', 'error');
                return;
            }
            
            showLoading(true);
            showStatus(`Automatically placing ${autoSeedCount} optimal seeds...`, 'info');
            updateDebugInfo('Running automatic seed placement algorithm...');
            
            try {
                const response = await fetch('/auto_place_seeds', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        num_seeds: autoSeedCount
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Clear existing seeds first
                    clearSeeds();
                    
                    // Add the automatically placed seeds
                    const colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple'];
                    
                    for (let i = 0; i < result.seed_positions.length; i++) {
                        const position = result.seed_positions[i];
                        const color = colors[i % colors.length];
                        
                        // Add seed with position and color
                        const seedData = {
                            position: position,
                            color: color,
                            colorRGB: seedColors[color]
                        };
                        selectedSeeds.push(seedData);
                        
                        // Add visual marker
                        const geometry = new THREE.SphereGeometry(0.04, 16, 16);
                        const colorRGB = seedColors[color];
                        const material = new THREE.MeshLambertMaterial({ 
                            color: new THREE.Color(colorRGB[0], colorRGB[1], colorRGB[2]),
                            opacity: 0.9,
                            transparent: true,
                            emissive: new THREE.Color(colorRGB[0] * 0.2, colorRGB[1] * 0.2, colorRGB[2] * 0.2),
                            emissiveIntensity: 0.3
                        });
                        const marker = new THREE.Mesh(geometry, material);
                        marker.position.set(position[0], position[1], position[2]);
                        marker.userData = { 
                            isSeedMarker: true, 
                            seedIndex: selectedSeeds.length - 1,
                            seedColor: color
                        };
                        scene.add(marker);
                    }
                    
                    updateSeedDisplay();
                    updateSegmentButtonState();
                    
                    showStatus(`✨ Successfully placed ${result.seed_positions.length} optimal seeds using curvature analysis and distance optimization!`, 'success');
                    updateDebugInfo(`Auto-placed ${result.seed_positions.length} seeds: ${result.algorithm_info}`);
                } else {
                    showStatus(`Error placing seeds: ${result.error}`, 'error');
                    updateDebugInfo('Auto-seed placement failed: ' + result.error);
                }
            } catch (error) {
                showStatus(`Network error: ${error.message}`, 'error');
                updateDebugInfo('Auto-seed placement network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function updateSegmentButtonState() {
            const segmentBtn = document.getElementById('segmentBtn');
            
            if (!meshObject) {
                segmentBtn.disabled = true;
                return;
            }
            
            if (segmentationMode === 'manual') {
                segmentBtn.disabled = selectedSeeds.length === 0;
            } else {
                segmentBtn.disabled = false;
            }
        }

        function onWindowResize() {
            const viewer = document.getElementById('viewer');
            camera.aspect = viewer.clientWidth / viewer.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewer.clientWidth, viewer.clientHeight);
        }
        
        function onMouseClick(event) {
            if (!meshObject || controls.isUserInteracting) {
                return;
            }
        
            const viewer = document.getElementById('viewer');
            const rect = viewer.getBoundingClientRect();
        
            mouse.x = ((event.clientX - rect.left) / viewer.clientWidth) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / viewer.clientHeight) * 2 + 1;
        
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(meshObject);
        
            if (intersects.length > 0) {
                const point = intersects[0].point;
        
                if (segmentationMode === 'manual') {
                    updateDebugInfo(`Seed click detected at: ${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)}`);
                    addSeed(point.x, point.y, point.z);
                } else if (segmentationMode === 'geodesic') {
                    updateDebugInfo(`Geodesic visualization click at: ${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)}`);
                    visualizeGeodesicDistance(point);
                }
            } else {
                updateDebugInfo('No intersection found on click');
            }
        }
        
        function handleFileSelect() {
            const fileInput = document.getElementById('meshFile');
            const uploadBtn = document.getElementById('uploadBtn');
            
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const fileName = file.name.toLowerCase();
                if (fileName.endsWith('.obj') || fileName.endsWith('.glb') || fileName.endsWith('.gltf')) {
                    uploadBtn.disabled = false;
                    uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload';
                    const fileType = fileName.endsWith('.obj') ? 'OBJ' : 
                                   fileName.endsWith('.glb') ? 'GLB (will be converted to OBJ)' : 
                                   'GLTF (will be converted to OBJ)';
                    showStatus(`📁 File ready: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) - ${fileType}`, 'info');
                } else {
                    uploadBtn.disabled = true;
                    uploadBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Invalid File Type';
                    showStatus('⚠️ Please select a .obj, .glb, or .gltf file', 'error');
                }
            } else {
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '<i class="fas fa-upload"></i> Upload';
            }
        }
        
        async function uploadMesh() {
            const fileInput = document.getElementById('meshFile');
            const curvaturePenalty = parseFloat(document.getElementById('curvaturePenalty').value);
            
            if (!fileInput.files.length) {
                showStatus('Please select a file first', 'error');
                return;
            }
            
            const file = fileInput.files[0];
            const fileName = file.name.toLowerCase();
            const isGlbGltf = fileName.endsWith('.glb') || fileName.endsWith('.gltf');
            
            showLoading(true);
            
            if (isGlbGltf) {
                showStatus('Uploading and converting GLB/GLTF to OBJ format...', 'info');
                updateDebugInfo('Converting GLB/GLTF file to OBJ...');
            } else {
                showStatus('Uploading and processing mesh...', 'info');
                updateDebugInfo('Uploading file to server...');
            }
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('curvature_penalty_strength', curvaturePenalty);
                
                const response = await fetch('/upload_mesh', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                updateDebugInfo('Server response received for upload');
                
                // Debug: Log the complete server response
                console.log('Complete server response:', result);
                updateDebugInfo(`Server response keys: ${Object.keys(result).join(', ')}`);
                
                if (result.success) {
                    // Debug: Check each data field
                    updateDebugInfo(`Response has vertices: ${!!result.vertices}, faces: ${!!result.faces}`);
                    if (result.vertices) updateDebugInfo(`Vertices length: ${result.vertices.length}`);
                    if (result.faces) updateDebugInfo(`Faces length: ${result.faces.length}`);
                    
                    meshData = result;
                    
                    // Debug: Validate mesh data
                    if (!result.vertices || !result.faces) {
                        updateDebugInfo('ERROR: Invalid mesh data - missing vertices or faces');
                        showStatus('Error: Invalid mesh data received', 'error');
                        return;
                    }
                    
                    if (result.vertices.length === 0 || result.faces.length === 0) {
                        updateDebugInfo('ERROR: Empty mesh data');
                        showStatus('Error: Mesh has no vertices or faces', 'error');
                        return;
                    }
                    
                    updateDebugInfo(`Valid mesh data: ${result.vertices.length} vertices, ${result.faces.length} faces`);
                    
                    displayMesh(result);
                    
                    // Display curvature penalty information if available
                    let statusMessage = `🚀 Mesh uploaded and loaded successfully! ${result.total_faces} faces ready for analysis.`;
                    if (result.curvature_stats) {
                        const stats = result.curvature_stats;
                        statusMessage += ` Curvature penalty applied (strength: ${stats.strength}) - Min: ${stats.min.toFixed(3)}, Max: ${stats.max.toFixed(3)}, Mean: ${stats.mean.toFixed(3)}`;
                    }
                    showStatus(statusMessage, 'success');
                    
                    clearSeeds();
                    updateSegmentButtonState();
                } else {
                    updateDebugInfo('Server error: ' + result.error);
                    showStatus(`Error processing mesh: ${result.error}`, 'error');
                }
            } catch (error) {
                updateDebugInfo('Network error: ' + error.message);
                showStatus(`Upload error: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function showStatus(message, type = 'info') {
            const statusContainer = document.getElementById('status');
            const statusDiv = document.createElement('div');
            statusDiv.className = `status ${type}`;
            
            // Add appropriate icon
            const icon = type === 'success' ? 'fa-check-circle' : 
                        type === 'error' ? 'fa-exclamation-circle' : 
                        'fa-info-circle';
            
            statusDiv.innerHTML = `<i class="fas ${icon}" style="margin-right: 8px;"></i>${message}`;
            statusContainer.appendChild(statusDiv);
            
            // Auto-remove after 5 seconds for non-error messages
            if (type !== 'error') {
                setTimeout(() => {
                    if (statusDiv.parentNode) {
                        statusDiv.style.animation = 'slideOut 0.3s ease forwards';
                        setTimeout(() => statusDiv.remove(), 300);
                    }
                }, 5000);
            }
        }
        
        function showLoading(show) {
            document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
        }
        
        async function loadMesh() {
            const meshPath = document.getElementById('meshPath').value;
            const curvaturePenalty = parseFloat(document.getElementById('curvaturePenalty').value);
            
            showLoading(true);
            showStatus('Loading mesh...', 'info');
            updateDebugInfo('Loading mesh from server...');
            
            try {
                const response = await fetch('/load_mesh', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        mesh_path: meshPath,
                        curvature_penalty_strength: curvaturePenalty
                    })
                });
                
                const result = await response.json();
                updateDebugInfo('Server response received');
                
                // Debug: Log the complete server response
                console.log('Complete server response:', result);
                updateDebugInfo(`Server response keys: ${Object.keys(result).join(', ')}`);
                
                if (result.success) {
                    // Debug: Check each data field
                    updateDebugInfo(`Response has vertices: ${!!result.vertices}, faces: ${!!result.faces}`);
                    if (result.vertices) updateDebugInfo(`Vertices length: ${result.vertices.length}`);
                    if (result.faces) updateDebugInfo(`Faces length: ${result.faces.length}`);
                    
                    meshData = result;
                    
                    // Debug: Validate mesh data
                    if (!result.vertices || !result.faces) {
                        updateDebugInfo('ERROR: Invalid mesh data - missing vertices or faces');
                        showStatus('Error: Invalid mesh data received', 'error');
                        return;
                    }
                    
                    if (result.vertices.length === 0 || result.faces.length === 0) {
                        updateDebugInfo('ERROR: Empty mesh data');
                        showStatus('Error: Mesh has no vertices or faces', 'error');
                        return;
                    }
                    
                    updateDebugInfo(`Valid mesh data: ${result.vertices.length} vertices, ${result.faces.length} faces`);
                    
                    displayMesh(result);
                    
                    // Display curvature penalty information if available
                    let statusMessage = `🚀 Mesh loaded successfully! ${result.total_faces} faces ready for analysis.`;
                    if (result.curvature_stats) {
                        const stats = result.curvature_stats;
                        statusMessage += ` Curvature penalty applied (strength: ${stats.strength}) - Min: ${stats.min.toFixed(3)}, Max: ${stats.max.toFixed(3)}, Mean: ${stats.mean.toFixed(3)}`;
                    }
                    showStatus(statusMessage, 'success');
                    
                    clearSeeds();
                    updateSegmentButtonState();
                } else {
                    updateDebugInfo('Server error: ' + result.error);
                    showStatus(`Error loading mesh: ${result.error}`, 'error');
                }
            } catch (error) {
                updateDebugInfo('Network error: ' + error.message);
                showStatus(`Network error: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function displayMesh(data) {
            try {
                updateDebugInfo(`Displaying mesh: ${data.vertices.length} vertices, ${data.faces.length} faces`);
                
                // Clear existing scene first
                clearScene();
                
                // Remove test cube
                scene.traverse((child) => {
                    if (child.geometry && child.geometry.type === 'BoxGeometry') {
                        scene.remove(child);
                    }
                });
                
                // Validate input data
                if (!data.vertices || !data.faces || data.vertices.length === 0 || data.faces.length === 0) {
                    updateDebugInfo('ERROR: Invalid or empty mesh data');
                    return;
                }
                
                // Create geometry
                const geometry = new THREE.BufferGeometry();
                
                // Convert vertices to flat array
                const vertices = [];
                for (let i = 0; i < data.vertices.length; i++) {
                    const vertex = data.vertices[i];
                    if (vertex && vertex.length >= 3) {
                        vertices.push(vertex[0], vertex[1], vertex[2]);
                    } else {
                        updateDebugInfo(`ERROR: Invalid vertex at index ${i}: ${vertex}`);
                        return;
                    }
                }
                
                // Convert faces to indices array and validate
                const indices = [];
                let maxVertexIndex = data.vertices.length - 1;
                
                for (let i = 0; i < data.faces.length; i++) {
                    const face = data.faces[i];
                    if (face && face.length >= 3) {
                        // Validate face indices
                        if (face[0] <= maxVertexIndex && face[1] <= maxVertexIndex && face[2] <= maxVertexIndex &&
                            face[0] >= 0 && face[1] >= 0 && face[2] >= 0) {
                            indices.push(face[0], face[1], face[2]);
                        } else {
                            updateDebugInfo(`ERROR: Invalid face indices at ${i}: [${face[0]}, ${face[1]}, ${face[2]}], max vertex: ${maxVertexIndex}`);
                            return;
                        }
                    } else {
                        updateDebugInfo(`ERROR: Invalid face at index ${i}: ${face}`);
                        return;
                    }
                }
                
                updateDebugInfo(`Processed: ${vertices.length/3} vertices, ${indices.length/3} faces`);
                
                // Debug: Check data ranges
                let minVert = [Infinity, Infinity, Infinity];
                let maxVert = [-Infinity, -Infinity, -Infinity];
                for (let i = 0; i < vertices.length; i += 3) {
                    minVert[0] = Math.min(minVert[0], vertices[i]);
                    minVert[1] = Math.min(minVert[1], vertices[i + 1]);
                    minVert[2] = Math.min(minVert[2], vertices[i + 2]);
                    maxVert[0] = Math.max(maxVert[0], vertices[i]);
                    maxVert[1] = Math.max(maxVert[1], vertices[i + 1]);
                    maxVert[2] = Math.max(maxVert[2], vertices[i + 2]);
                }
                
                updateDebugInfo(`Mesh bounds: min(${minVert[0].toFixed(2)}, ${minVert[1].toFixed(2)}, ${minVert[2].toFixed(2)}) max(${maxVert[0].toFixed(2)}, ${maxVert[1].toFixed(2)}, ${maxVert[2].toFixed(2)})`);
                
                // Create buffer geometry with proper buffer management
                const positionAttribute = new THREE.BufferAttribute(new Float32Array(vertices), 3);
                positionAttribute.needsUpdate = true;
                geometry.setAttribute('position', positionAttribute);
                
                // Only set indices if we have valid ones
                if (indices.length > 0) {
                    // Convert to appropriate array type based on vertex count
                    let indexAttribute;
                    if (maxVertexIndex < 65536) {
                        indexAttribute = new THREE.BufferAttribute(new Uint16Array(indices), 1);
                    } else {
                        indexAttribute = new THREE.BufferAttribute(new Uint32Array(indices), 1);
                    }
                    indexAttribute.needsUpdate = true;
                    geometry.setIndex(indexAttribute);
                    updateDebugInfo(`Set indices: ${indices.length} elements (${indexAttribute.array.constructor.name})`);
                } else {
                    updateDebugInfo('ERROR: No valid indices created');
                    return;
                }
                
                // Compute normals AFTER setting position and indices
                try {
                    geometry.computeVertexNormals();
                    updateDebugInfo('Computed vertex normals');
                } catch (normalError) {
                    updateDebugInfo(`Warning: Could not compute normals: ${normalError.message}`);
                }
                
                // Get bounding box
                geometry.computeBoundingBox();
                if (!geometry.boundingBox) {
                    updateDebugInfo('ERROR: Could not compute bounding box');
                    return;
                }
                
                const box = geometry.boundingBox;
                const center = new THREE.Vector3();
                box.getCenter(center);
                const size = new THREE.Vector3();
                box.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z);
                
                updateDebugInfo(`Bounding box: center(${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}) size(${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)}) maxDim: ${maxDim.toFixed(2)}`);
                
                // Create material - start with solid, not wireframe to avoid Three.js wireframe bug
                const material = new THREE.MeshLambertMaterial({
                    color: 0x888888,
                    side: THREE.DoubleSide
                });
                
                // Create mesh
                meshObject = new THREE.Mesh(geometry, material);
                
                // Reset position and scale
                meshObject.position.set(0, 0, 0);
                meshObject.scale.set(1, 1, 1);
                
                // Center the mesh
                meshObject.position.copy(center).negate();
                
                // Scale the mesh to fit in a 4-unit cube
                if (maxDim > 0) {
                    const scale = 4 / maxDim;
                    meshObject.scale.setScalar(scale);
                    updateDebugInfo(`Applied scale: ${scale.toFixed(4)}`);
                } else {
                    updateDebugInfo('Warning: mesh has zero size!');
                }
                
                // Add to scene
                scene.add(meshObject);
                updateDebugInfo('Mesh added to scene (solid mode)');
                
                // Position camera to see the mesh better
                camera.position.set(6, 6, 6);
                camera.lookAt(0, 0, 0);
                controls.target.set(0, 0, 0);
                controls.update();
                
                updateDebugInfo('Camera positioned - mesh should be visible');
                
                // Force a safe render
                try {
                    renderer.render(scene, camera);
                    updateDebugInfo('Initial render successful');
                } catch (renderError) {
                    updateDebugInfo('Initial render failed: ' + renderError.message);
                }
                
            } catch (error) {
                updateDebugInfo('Error displaying mesh: ' + error.message);
                showStatus('Error displaying mesh: ' + error.message, 'error');
                console.error('Full error:', error);
            }
        }
        
        function toggleWireframe() {
            if (meshObject && meshObject.material) {
                const isWireframe = meshObject.material.wireframe;
                
                if (isWireframe) {
                    // Switch to solid
                    meshObject.material = new THREE.MeshLambertMaterial({
                        color: 0x888888,
                        side: THREE.DoubleSide
                    });
                    updateDebugInfo('Switched to solid mode');
                } else {
                    // Switch to wireframe - create a new wireframe material
                    meshObject.material = new THREE.MeshBasicMaterial({
                        color: 0x00ff00,
                        wireframe: true,
                        side: THREE.DoubleSide
                    });
                    updateDebugInfo('Switched to wireframe mode');
                }
            } else {
                updateDebugInfo('No mesh object to toggle');
            }
        }
        
        function addSeed(x, y, z) {
            const seedData = {
                position: [x, y, z],
                color: currentSeedColor,
                colorRGB: seedColors[currentSeedColor]
            };
            selectedSeeds.push(seedData);
            updateSeedDisplay();
            
            // Add visual marker with the selected color
            const geometry = new THREE.SphereGeometry(0.04, 16, 16);
            const colorRGB = seedColors[currentSeedColor];
            const material = new THREE.MeshLambertMaterial({ 
                color: new THREE.Color(colorRGB[0], colorRGB[1], colorRGB[2]),
                opacity: 0.9,
                transparent: true,
                emissive: new THREE.Color(colorRGB[0] * 0.2, colorRGB[1] * 0.2, colorRGB[2] * 0.2),
                emissiveIntensity: 0.3
            });
            const marker = new THREE.Mesh(geometry, material);
            marker.position.set(x, y, z);
            marker.userData = { 
                isSeedMarker: true, 
                seedIndex: selectedSeeds.length - 1,
                seedColor: currentSeedColor
            };
            scene.add(marker);
            
            showStatus(`${currentSeedColor.charAt(0).toUpperCase() + currentSeedColor.slice(1)} seed ${selectedSeeds.length} added at (${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)})`, 'success');
        }
        
        function toggleSeedMarkers() {
            // Check if there are any seed markers visible
            let markersVisible = false;
            scene.traverse((child) => {
                if (child.userData && child.userData.isSeedMarker) {
                    markersVisible = child.visible;
                    return; // Exit early since we found one
                }
            });
            
            // Toggle visibility of all seed markers
            scene.traverse((child) => {
                if (child.userData && child.userData.isSeedMarker) {
                    child.visible = !markersVisible;
                }
            });
            
            const status = markersVisible ? 'hidden' : 'shown';
            showStatus(`Seed markers ${status}`, 'info');
        }
        
        function removeSeedMarkers() {
            // Remove visual markers from scene
            const markersToRemove = [];
            scene.traverse((child) => {
                if (child.userData && (child.userData.isSeedMarker || child.userData.isRingMarker)) {
                    markersToRemove.push(child);
                }
            });
            markersToRemove.forEach(marker => {
                scene.remove(marker);
                if (marker.geometry) marker.geometry.dispose();
                if (marker.material) marker.material.dispose();
            });
        }
        
        function updateSeedColors(seedColors) {
            // Update existing seed markers with their segment colors
            scene.traverse((child) => {
                if (child.userData && child.userData.isSeedMarker) {
                    const seedIndex = child.userData.seedIndex;
                    if (seedIndex < seedColors.length) {
                        const color = seedColors[seedIndex];
                        child.material.color.setRGB(color[0], color[1], color[2]);
                        // Add some emissive color for better visibility
                        child.material.emissive.setRGB(color[0] * 0.2, color[1] * 0.2, color[2] * 0.2);
                        // Make the seeds slightly larger and more visible
                        child.scale.setScalar(1.3);
                        child.material.opacity = 0.95;
                        updateDebugInfo(`Updated seed ${seedIndex} to color [${color[0].toFixed(2)}, ${color[1].toFixed(2)}, ${color[2].toFixed(2)}]`);
                    }
                }
            });
        }
        
        function clearSeeds() {
            selectedSeeds = [];
            updateSeedDisplay();
            
            // Remove visual markers
            removeSeedMarkers();
            
            // Reset mesh colors if segmented
            if (meshObject && meshObject.material.vertexColors) {
                meshObject.material = new THREE.MeshLambertMaterial({
                    color: 0x888888,
                    wireframe: false,
                    side: THREE.DoubleSide
                });
            }
            
            showStatus('All seeds cleared.', 'info');
        }
        
        function updateSeedDisplay() {
            const seedCount = document.getElementById('seedCount');
            const segmentBtn = document.getElementById('segmentBtn');
            const seedList = document.getElementById('seedList');
            
            seedCount.textContent = selectedSeeds.length;
            
            updateSegmentButtonState();

            if (selectedSeeds.length === 0) {
                seedList.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-style: italic;">No seeds selected</div>';
            } else {
                // Group seeds by color for display
                const seedsByColor = {};
                selectedSeeds.forEach((seed, index) => {
                    if (!seedsByColor[seed.color]) {
                        seedsByColor[seed.color] = [];
                    }
                    seedsByColor[seed.color].push({ seed, index });
                });
                
                let html = '';
                Object.keys(seedsByColor).forEach(color => {
                    const colorSeeds = seedsByColor[color];
                    const colorRGB = seedColors[color];
                    const colorStyle = `rgb(${Math.round(colorRGB[0] * 255)}, ${Math.round(colorRGB[1] * 255)}, ${Math.round(colorRGB[2] * 255)})`;
                    
                    html += `<div class="seed-item">
                        <div class="seed-color-indicator" style="background-color: ${colorStyle};"></div>
                        <div class="seed-info">
                            <strong>${color.charAt(0).toUpperCase() + color.slice(1)}</strong>: ${colorSeeds.length} seed${colorSeeds.length > 1 ? 's' : ''}
                        </div>
                    </div>`;
                });
                
                seedList.innerHTML = html;
            }
        }
        
        async function runSegmentation() {
            showLoading(true);
            
            if (segmentationMode === 'manual') {
                // Manual segmentation code (keep existing code)
                if (selectedSeeds.length === 0) {
                    showStatus('Please select at least one seed point by clicking on the mesh.', 'error');
                    showLoading(false);
                    return;
                }
                
                showStatus('Processing mesh segmentation with colored seeds...', 'info');
                
                try {
                    const seedData = selectedSeeds.map(seed => ({
                        position: seed.position,
                        color: seed.color
                    }));
                    
                    const response = await fetch('/segment_with_colored_seeds', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            colored_seeds: seedData,
                            output_dir: document.getElementById('outputDir').value
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displaySegmentedMesh(result.face_colors);
                        let statusMessage = `✨ Segmentation complete! ${result.segments_created} segments created`;
                        if (result.combined_segments) {
                            statusMessage += ` and combined into ${result.combined_segments} groups by color`;
                        }
                        statusMessage += '.';
                        
                        if (result.stats) {
                            const stats = result.stats;
                            statusMessage += ` (${stats.segmented_faces}/${stats.total_faces} faces segmented, ${((stats.segmented_faces/stats.total_faces)*100).toFixed(1)}%)`;
                            updateDebugInfo(`Segmentation stats: ${stats.segmented_faces}/${stats.total_faces} faces segmented`);
                        }
                        
                        showStatus(statusMessage, 'success');
                        document.getElementById('downloadBtn').disabled = false;
                    } else {
                        showStatus(`Segmentation error: ${result.error}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Network error: ${error.message}`, 'error');
                } finally {
                    showLoading(false);
                }
            } else {
                // Automatic segmentation
                const nSeeds = parseInt(document.getElementById('nSeeds').value);
                
                if (nSeeds < 2) {
                    showStatus('Number of segments must be at least 2', 'error');
                    showLoading(false);
                    return;
                }
                
                showStatus(`Running automatic segmentation with ${nSeeds} segments...`, 'info');
                
                try {
                    const response = await fetch('/segment_automatic', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            n_seeds: nSeeds,
                            output_dir: document.getElementById('outputDir').value
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displaySegmentedMesh(result.face_colors);
                        
                        let statusMessage = `✨ Automatic segmentation complete! ${result.segments_created} segments created.`;
                        
                        if (result.stats) {
                            const stats = result.stats;
                            statusMessage += ` (${stats.segmented_faces}/${stats.total_faces} faces segmented, ${((stats.segmented_faces/stats.total_faces)*100).toFixed(1)}%)`;
                            updateDebugInfo(`Automatic segmentation stats: ${stats.segmented_faces}/${stats.total_faces} faces segmented`);
                        }
                        
                        showStatus(statusMessage, 'success');
                        document.getElementById('downloadBtn').disabled = false;
                    } else {
                        showStatus(`Segmentation error: ${result.error}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Network error: ${error.message}`, 'error');
                } finally {
                    showLoading(false);
                }
            }
        }
        
        async function visualizeGeodesicDistance(point) {
            if (!meshObject) {
                showStatus('Please load a mesh first', 'error');
                return;
            }
        
            showLoading(true);
            showStatus('Calculating geodesic distances...', 'info');
        
            try {
                const response = await fetch('/visualize_geodesic_distance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        clicked_point: [point.x, point.y, point.z]
                    })
                });
        
                const result = await response.json();
        
                if (result.success) {
                    displaySegmentedMesh(result.face_colors);
                    showStatus('Geodesic distance visualization complete.', 'success');
                } else {
                    showStatus(`Error visualizing distance: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Network error: ${error.message}`, 'error');
            } finally {
                showLoading(false);
            }
        }

        function displaySegmentedMesh(faceColors) {
            if (!meshObject || !meshData) {
                updateDebugInfo('ERROR: No mesh object or data available for segmentation display');
                return;
            }
            
            updateDebugInfo(`[FIXED] Displaying segmented mesh with ${faceColors.length} face colors using smart vertex color averaging`);
            
            try {
                // Validate face colors array
                if (faceColors.length !== meshData.faces.length) {
                    updateDebugInfo(`WARNING: Face colors length (${faceColors.length}) != faces length (${meshData.faces.length})`);
                }
                
                // Keep the original indexed geometry but fix the color assignment
                const geometry = meshObject.geometry.clone();
                
                // Track vertex colors with proper averaging
                const positionAttribute = geometry.attributes.position;
                const vertexCount = positionAttribute.count;
                
                // Map to track all face colors for each vertex
                const vertexFaceColors = new Map();
                
                // Build mapping of vertices to all their face colors
                for (let faceIdx = 0; faceIdx < meshData.faces.length; faceIdx++) {
                    const face = meshData.faces[faceIdx];
                    const faceColor = faceColors[faceIdx] || [0.4, 0.4, 0.4];
                    
                    for (let j = 0; j < 3; j++) {
                        const vertexIdx = face[j];
                        if (vertexIdx < vertexCount) {
                            if (!vertexFaceColors.has(vertexIdx)) {
                                vertexFaceColors.set(vertexIdx, []);
                            }
                            vertexFaceColors.get(vertexIdx).push(faceColor);
                        }
                    }
                }
                
                // Create vertex colors by averaging all face colors for each vertex
                const colors = [];
                for (let i = 0; i < vertexCount; i++) {
                    const faceColors = vertexFaceColors.get(i) || [[0.4, 0.4, 0.4]];
                    
                    // Average all face colors for this vertex
                    let avgR = 0, avgG = 0, avgB = 0;
                    for (const color of faceColors) {
                        avgR += color[0];
                        avgG += color[1];
                        avgB += color[2];
                    }
                    const count = faceColors.length;
                    
                    colors.push(avgR / count, avgG / count, avgB / count);
                }
                
                updateDebugInfo(`Created averaged vertex colors: ${colors.length/3} vertices colored`);
                
                // Set color attribute
                const colorAttribute = new THREE.BufferAttribute(new Float32Array(colors), 3);
                colorAttribute.needsUpdate = true;
                geometry.setAttribute('color', colorAttribute);
                
                // Create new material with vertex colors
                const material = new THREE.MeshLambertMaterial({
                    vertexColors: true,
                    side: THREE.DoubleSide
                });
                
                // Update the existing mesh
                if (meshObject.material) {
                    if (meshObject.material.map) meshObject.material.map.dispose();
                    meshObject.material.dispose();
                }
                if (meshObject.geometry !== geometry) {
                    meshObject.geometry.dispose();
                }
                
                meshObject.geometry = geometry;
                meshObject.material = material;
                
                updateDebugInfo('Segmented mesh display updated successfully with averaged vertex colors');
                
                // Force render update
                renderer.render(scene, camera);
                
            } catch (error) {
                updateDebugInfo('Error in displaySegmentedMesh: ' + error.message);
                console.error('Error in displaySegmentedMesh:', error);
            }
        }
        
        async function updatePenaltyStrength() {
            if (!meshObject) {
                return; // No mesh loaded, nothing to update
            }

            const newPenalty = parseFloat(document.getElementById('curvaturePenalty').value);
            showStatus(`Updating curvature penalty to ${newPenalty}...`, 'info');

            try {
                const response = await fetch('/update_penalty', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        curvature_penalty_strength: newPenalty
                    })
                });

                const result = await response.json();

                if (result.success) {
                    showStatus(`✅ Penalty strength updated. The graph has been rebuilt.`, 'success');
                    updateDebugInfo(`Graph rebuilt with new penalty: ${newPenalty}`);
                } else {
                    showStatus(`Error updating penalty: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Network error while updating penalty: ${error.message}`, 'error');
            }
        }

        function downloadSegments() {
            const outputDir = document.getElementById('outputDir').value;
            window.open(`/download_segments?output_dir=${encodeURIComponent(outputDir)}`, '_blank');
        }
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            updateDebugInfo('DOM loaded, initializing...');

            // Add listener for curvature penalty changes
            const penaltyInput = document.getElementById('curvaturePenalty');
            if (penaltyInput) {
                penaltyInput.addEventListener('change', updatePenaltyStrength);
            }

            // Update n_seeds display when input changes
            const nSeedsInput = document.getElementById('nSeeds');
            const nSeedsDisplay = document.getElementById('nSeedsDisplay');

            if (nSeedsInput) {
                nSeedsInput.addEventListener('input', function() {
                    if (nSeedsDisplay) {
                        nSeedsDisplay.textContent = this.value;
                    }
                });
            }
            
            // Wait a bit for Three.js to load
            setTimeout(() => {
                if (typeof THREE !== 'undefined') {
                    updateDebugInfo('Three.js available, version: ' + THREE.REVISION);
                    initThreeJS();
                    
                    // Auto-load the example mesh after Three.js is initialized
                    setTimeout(() => {
                        updateDebugInfo('Auto-loading example mesh...');
                        loadMesh();
                    }, 500);
                } else {
                    updateDebugInfo('Three.js not loaded, retrying...');
                    setTimeout(() => {
                        if (typeof THREE !== 'undefined') {
                            initThreeJS();
                            // Auto-load after retry
                            setTimeout(() => {
                                updateDebugInfo('Auto-loading example mesh...');
                                loadMesh();
                            }, 500);
                        } else {
                            updateDebugInfo('Failed to load Three.js');
                        }
                    }, 1000);
                }
            }, 100);
        });
        
        async function facilityPlaceSeeds() {
            console.log('facilityPlaceSeeds function called!');
            
            if (!meshObject || !meshData) {
                showStatus('Please load a mesh first', 'error');
                return;
            }
            
            const numSeeds = parseInt(document.getElementById('facilitySeeds').value);
            const strategy = document.getElementById('facilityStrategy').value;
            
            if (numSeeds < 1 || numSeeds > 20) {
                showStatus('Number of seeds must be between 1 and 20', 'error');
                return;
            }
            
            showLoading(true);
            showStatus(`Running facility placement algorithm with ${numSeeds} seeds using ${strategy} strategy...`, 'info');
            updateDebugInfo(`Facility placement: ${numSeeds} seeds, strategy: ${strategy}`);
            
            try {
                const response = await fetch('/facility_place_seeds', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        num_seeds: numSeeds,
                        strategy: strategy
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Clear existing seeds first
                    clearSeeds();
                    
                    // Add the facility-placed seeds with their assigned colors
                    for (let i = 0; i < result.seeds.length; i++) {
                        const seedData = result.seeds[i];
                        const position = seedData.position;
                        const color = seedData.color;
                        
                        // Add seed with position and color
                        const seed = {
                            position: position,
                            color: color,
                            colorRGB: seedColors[color] || [0.5, 0.5, 0.5]
                        };
                        selectedSeeds.push(seed);
                        
                        // Add visual marker
                        const geometry = new THREE.SphereGeometry(0.05, 20, 20);
                        const colorRGB = seed.colorRGB;
                        
                        // Create a more prominent material for facility-placed seeds
                        const material = new THREE.MeshLambertMaterial({ 
                            color: new THREE.Color(colorRGB[0], colorRGB[1], colorRGB[2]),
                            opacity: 0.95,
                            transparent: true,
                            emissive: new THREE.Color(colorRGB[0] * 0.3, colorRGB[1] * 0.3, colorRGB[2] * 0.3),
                            emissiveIntensity: 0.4
                        });
                        
                        const marker = new THREE.Mesh(geometry, material);
                        marker.position.set(position[0], position[1], position[2]);
                        
                        // Add a subtle ring around facility-placed seeds to distinguish them
                        const ringGeometry = new THREE.RingGeometry(0.06, 0.08, 16);
                        const ringMaterial = new THREE.MeshBasicMaterial({ 
                            color: new THREE.Color(colorRGB[0], colorRGB[1], colorRGB[2]),
                            opacity: 0.6,
                            transparent: true,
                            side: THREE.DoubleSide
                        });
                        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
                        ring.position.set(position[0], position[1], position[2]);
                        ring.lookAt(camera.position); // Face the camera
                        scene.add(ring);
                        
                        marker.userData = { 
                            isSeedMarker: true, 
                            seedIndex: selectedSeeds.length - 1,
                            seedColor: color,
                            isFacilityPlaced: true,
                            ringMarker: ring
                        };
                        ring.userData = {
                            isRingMarker: true,
                            parentSeed: marker
                        };
                        
                        scene.add(marker);
                    }
                    
                    updateSeedDisplay();
                    updateSegmentButtonState();
                    
                    showStatus(`🎯 Facility placement complete! ${result.num_seeds_placed} seeds optimally placed using ${result.strategy_used} algorithm. ${result.algorithm_info}`, 'success');
                    updateDebugInfo(`Facility placement successful: ${result.num_seeds_placed} seeds using ${result.strategy_used}`);
                } else {
                    showStatus(`Error in facility placement: ${result.error}`, 'error');
                    updateDebugInfo('Facility placement failed: ' + result.error);
                }
            } catch (error) {
                showStatus(`Network error: ${error.message}`, 'error');
                updateDebugInfo('Facility placement network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        // Make functions globally accessible for debugging
        window.autoPlaceSeeds = autoPlaceSeeds;
        window.facilityPlaceSeeds = facilityPlaceSeeds;
        window.loadMesh = loadMesh;
        window.uploadMesh = uploadMesh;