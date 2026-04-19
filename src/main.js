import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import GUI from 'lil-gui';

/**
 * TSDF Demo - Step-by-Step Educational Version
 */

const params = {
    resolution: 64,
    mu: 0.15,
    noise: 0.05,
    outliers: 0.02,
    gaussianCount: 500,
    showGaussians: true,
    showPoints: true,
    showVoxels: true,
    showMesh: true,
    showSlice: false,
    showCamera: true,
    showKnowledge: () => toggleModal(true),
    cameraZoom: 5.2, // initial distance = length of (3,3,3)
    slicePos: 0.0,
    type: 'sphere',

    // Step Buttons
    step1: () => triggerStep(1),
    step2: () => triggerStep(2),
    step3: () => triggerStep(3),
    step4: () => triggerStep(4),
    step5: () => triggerStep(5),
    reset: () => resetDemo(),
};

let scene, camera, renderer, controls;
let marchingCubes, marchingWireframe, pointCloud, voxelCloud, slicePlane;
let gaussianCloud = null;
let cameraHelpers = [];
let worker;
let currentStep = 0;
let distancesBuffer = null;
let pendingPoints = null; // point cloud from step 1, shown in step 2

const DEPTH_RES = 48; // must match worker.js

/**
 * Build a cloud of semi-transparent colored ellipsoids that visually represents
 * a 3D Gaussian Splatting scene. Each instance is a flattened ellipsoid (thin
 * in the surface-normal direction, wide tangentially) — the hallmark look of GSplat.
 * @param {'sphere'|'torus'} type
 */
function createGaussianCloud(type, N = params.gaussianCount) {
    // Low-poly sphere as the base ellipsoid shape
    const geo = new THREE.SphereGeometry(1, 5, 4);
    const mat = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.40,
        depthWrite: false,
        // Additive blending makes dense overlapping regions brighter,
        // matching the appearance of Gaussian splats in a real renderer.
        blending: THREE.AdditiveBlending,
    });

    const cloud = new THREE.InstancedMesh(geo, mat, N);

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const normal = new THREE.Vector3();

    for (let i = 0; i < N; i++) {
        let x, y, z, nx, ny, nz;

        if (type === 'sphere') {
            const phi   = Math.random() * Math.PI * 2;
            const theta = Math.acos(2 * Math.random() - 1);
            nx = Math.sin(theta) * Math.cos(phi);
            ny = Math.sin(theta) * Math.sin(phi);
            nz = Math.cos(theta);
            x = nx; y = ny; z = nz;
        } else {
            const phi   = Math.random() * Math.PI * 2;
            const theta = Math.random() * Math.PI * 2;
            const R = 1.0, r = 0.4;
            x  = (R + r * Math.cos(theta)) * Math.cos(phi);
            y  = (R + r * Math.cos(theta)) * Math.sin(phi);
            z  = r * Math.sin(theta);
            nx = Math.cos(theta) * Math.cos(phi);
            ny = Math.cos(theta) * Math.sin(phi);
            nz = Math.sin(theta);
        }

        // Small position jitter to simulate imperfect 3DGS reconstruction
        x += (Math.random() - 0.5) * 0.06;
        y += (Math.random() - 0.5) * 0.06;
        z += (Math.random() - 0.5) * 0.06;

        // Anisotropic scale: flat in normal dir (thin), wide tangentially
        const tang = 0.05 + Math.random() * 0.10;
        const norm = 0.015 + Math.random() * 0.025;
        dummy.position.set(x, y, z);
        dummy.scale.set(tang, norm, tang);

        // Align the ellipsoid's Y-axis (thin axis) to the surface normal
        normal.set(nx, ny, nz);
        dummy.quaternion.setFromUnitVectors(yAxis, normal);
        // Random spin around normal for tangential variety
        dummy.rotateOnWorldAxis(normal, Math.random() * Math.PI);

        dummy.updateMatrix();
        cloud.setMatrixAt(i, dummy.matrix);

        // Color: hue from azimuth angle, saturation/lightness with slight randomness.
        // Adjacent Gaussians share similar hues — mimics view-dependent SH color.
        const hue = (Math.atan2(nz, nx) / (Math.PI * 2) + 0.5 + ny * 0.15) % 1.0;
        color.setHSL(hue, 0.65 + Math.random() * 0.25, 0.55 + Math.random() * 0.2);
        cloud.setColorAt(i, color);
    }

    cloud.instanceMatrix.needsUpdate = true;
    if (cloud.instanceColor) cloud.instanceColor.needsUpdate = true;

    return cloud;
}

/**
 * COLMAP-style camera icon: a simple pyramid (apex + image plane rectangle).
 * Apex at the given position, pyramid opens toward the scene origin.
 * @param {THREE.Vector3} position
 */
function createCamIcon(position) {
    const w = 0.10;
    const h = 0.07;
    const d = 0.18;

    // Pyramid outline (apex at origin, image plane at z = -d in camera-local space)
    /* eslint-disable no-multi-spaces */
    const verts = new Float32Array([
        0,  0,  0,  -w,  h, -d,
        0,  0,  0,   w,  h, -d,
        0,  0,  0,   w, -h, -d,
        0,  0,  0,  -w, -h, -d,
        -w,  h, -d,   w,  h, -d,
         w,  h, -d,   w, -h, -d,
         w, -h, -d,  -w, -h, -d,
        -w, -h, -d,  -w,  h, -d,
    ]);
    /* eslint-enable no-multi-spaces */

    const outline = new THREE.LineSegments(
        new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(verts, 3)),
        new THREE.LineBasicMaterial({ color: 0x88c0d0 }),
    );

    // Depth map image plane — initially dark, updated per camera during step 1
    const canvas = document.createElement('canvas');
    canvas.width = DEPTH_RES;
    canvas.height = DEPTH_RES;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, DEPTH_RES, DEPTH_RES);
    const texture = new THREE.CanvasTexture(canvas);

    const imgPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(w * 2, h * 2),
        new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }),
    );
    imgPlane.position.set(0, 0, -d);

    const group = new THREE.Group();
    group.add(outline);
    group.add(imgPlane);

    // Stored for texture updates from the depth_progress handler
    group.depthCtx = ctx;
    group.depthTexture = texture;

    group.position.copy(position);

    // Copy quaternion from a PerspectiveCamera so -Z faces the origin
    const cam = new THREE.PerspectiveCamera();
    cam.position.copy(position);
    cam.lookAt(0, 0, 0);
    group.quaternion.copy(cam.quaternion);

    return group;
}

init();

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(3, 3, 3);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting (Nord-themed)
    const ambientLight = new THREE.AmbientLight(0x434c5e); 
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xeceff4, 1);
    dirLight.position.set(1, 1, 1);
    scene.add(dirLight);

    // Helpers
    const gridHelper = new THREE.GridHelper(4, 20, 0x4c566a, 0x3b4252);
    scene.add(gridHelper);

    // Gaussian cloud — represents the 3DGS scene the virtual cameras will scan
    gaussianCloud = createGaussianCloud(params.type);
    scene.add(gaussianCloud);

    // Virtual Cameras — ring of cameras around the object, each pointing inward
    // This matches the 3DGS → TSDF pipeline: render depth/opacity maps from N viewpoints
    const CAM_COUNT = 8;
    const CAM_RADIUS = 2.5;
    const CAM_HEIGHT = 1.0;
    for (let i = 0; i < CAM_COUNT; i++) {
        const angle = (i / CAM_COUNT) * Math.PI * 2;
        const pos = new THREE.Vector3(
            CAM_RADIUS * Math.cos(angle),
            CAM_HEIGHT,
            CAM_RADIUS * Math.sin(angle)
        );
        const icon = createCamIcon(pos);
        icon.visible = params.showCamera;
        scene.add(icon);
        cameraHelpers.push(icon);
    }

    // Step 4: Mesh (Marching Cubes)
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x88c0d0, // nord8 (Frost)
        specular: 0x2e3440,
        shininess: 2,
        side: THREE.DoubleSide,
        flatShading: false,
        transparent: true,
        opacity: 0.8
    });
    
    marchingCubes = new MarchingCubes(params.resolution, material, true, true, 100000);
    marchingCubes.scale.set(2, 2, 2);
    marchingCubes.visible = false;
    scene.add(marchingCubes);

    // Step 4 wireframe overlay — shares geometry with marchingCubes so it updates automatically
    marchingWireframe = new THREE.Mesh(
        marchingCubes.geometry,
        new THREE.MeshBasicMaterial({ color: 0x4c566a, wireframe: true, transparent: true, opacity: 0.35 })
    );
    marchingWireframe.scale.set(2, 2, 2);
    marchingWireframe.visible = false;
    scene.add(marchingWireframe);

    // Step 1: Point Cloud
    const pcMaterial = new THREE.PointsMaterial({ 
        color: 0xbf616a, // nord11 (Aurora Red)
        size: 0.02,
        transparent: true,
        opacity: 0.8
    });
    pointCloud = new THREE.Points(new THREE.BufferGeometry(), pcMaterial);
    pointCloud.visible = false;
    scene.add(pointCloud);

    // Step 2: Voxelization
    const voxelMaterial = new THREE.PointsMaterial({
        color: 0xebcb8b, // nord13 (Yellow)
        size: 0.03,
        transparent: true,
        opacity: 0.4
    });
    voxelCloud = new THREE.Points(new THREE.BufferGeometry(), voxelMaterial);
    voxelCloud.visible = false;
    scene.add(voxelCloud);

    // Slicing Plane
    setupSlicePlane();

    setupGUI();
    initWorker();

    // Modal Event Listeners
    document.getElementById('close-modal').addEventListener('click', () => toggleModal(false));
    document.getElementById('lang-toggle').addEventListener('click', () => toggleLanguage());
    document.getElementById('modal-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'modal-overlay') toggleModal(false);
    });

    window.addEventListener('resize', onWindowResize);
    animate();
}

let currentLang = 'en';

function applyLanguage(lang) {
    document.querySelectorAll('.lang-en').forEach(el => el.classList.toggle('hidden', lang !== 'en'));
    document.querySelectorAll('.lang-zh').forEach(el => el.classList.toggle('hidden', lang !== 'zh'));
    renderMath();
}

function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'zh' : 'en';
    applyLanguage(currentLang);
}

function toggleModal(show) {
    const modal = document.getElementById('modal-overlay');
    if (show) {
        modal.classList.remove('hidden');
        applyLanguage(currentLang);
    } else {
        modal.classList.add('hidden');
    }
}

function renderMath() {
    const modalBody = document.getElementById('modal-body');
    if (typeof renderMathInElement === 'function') {
        renderMathInElement(modalBody, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    }

    // Still render the specific formulas if needed, or let auto-render handle them
    // Our HTML has IDs formula-d and formula-w with math content inside or we can just put math in them
}

function setupSlicePlane() {
    const canvas = document.createElement('canvas');
    canvas.width = params.resolution;
    canvas.height = params.resolution;
    const texture = new THREE.CanvasTexture(canvas);
    texture.interpolation = THREE.NearestFilter;

    const sliceMat = new THREE.MeshBasicMaterial({ 
        map: texture, 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.9 
    });
    slicePlane = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), sliceMat);
    slicePlane.rotation.y = Math.PI / 2;
    slicePlane.visible = false;
    scene.add(slicePlane);
}

function updateSliceTexture(distances) {
    if (!slicePlane.visible) return;
    
    const canvas = slicePlane.material.map.image;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    
    const sliceIdx = Math.floor(((params.slicePos + 2.0) / 4.0) * params.resolution);
    const clampedSlice = Math.max(0, Math.min(params.resolution - 1, sliceIdx));
    
    const res = params.resolution;
    for (let y = 0; y < res; y++) {
        for (let z = 0; z < res; z++) {
            const idx = clampedSlice + y * res + z * res * res;
            const d = distances[idx];
            
            // Map d (-mu to mu) to color
            // Blue for negative (inside), Red for positive (outside), White for zero
            const val = d / params.mu; // -1 to 1
            const r = val > 0 ? 255 : 255 * (1 + val);
            const b = val < 0 ? 255 : 255 * (1 - val);
            const g = 255 * (1 - Math.abs(val));

            const pixelIdx = (y + z * res) * 4;
            imgData.data[pixelIdx] = r;
            imgData.data[pixelIdx + 1] = g;
            imgData.data[pixelIdx + 2] = b;
            imgData.data[pixelIdx + 3] = 255;
        }
    }
    
    ctx.putImageData(imgData, 0, 0);
    slicePlane.material.map.needsUpdate = true;
    slicePlane.position.x = params.slicePos;
}

function setupGUI() {
    const gui = new GUI();
    
    gui.add(params, 'showKnowledge').name('💡 Knowledge Base');
    gui.add(params, 'cameraZoom', 1.5, 12, 0.1).name('🔍 Zoom').onChange(v => {
        const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
        offset.setLength(v);
        camera.position.copy(controls.target).add(offset);
        controls.update();
    });

    const settings = gui.addFolder('Settings');
    settings.add(params, 'type', ['sphere', 'torus']).name('Surface Type').onChange(v => {
        scene.remove(gaussianCloud);
        gaussianCloud = createGaussianCloud(v);
        scene.add(gaussianCloud);
        resetDemo();
    });
    settings.add(params, 'gaussianCount', 100, 1000, 50).name('Gaussian Count').onChange(() => {
        scene.remove(gaussianCloud);
        gaussianCloud = createGaussianCloud(params.type);
        gaussianCloud.visible = params.showGaussians && currentStep < 5;
        scene.add(gaussianCloud);
    });
    settings.add(params, 'mu', 0.05, 0.4).name('Truncation (μ)');
    settings.add(params, 'noise', 0, 0.2).name('Noise');

    const steps = gui.addFolder('Steps');
    steps.add(params, 'step1').name('1. Render Depth Maps');
    steps.add(params, 'step2').name('2. Show Point Cloud');
    steps.add(params, 'step3').name('3. Voxelize');
    steps.add(params, 'step4').name('4. Fuse Logic');
    steps.add(params, 'step5').name('5. Extract Mesh');
    steps.add(params, 'reset').name('Reset Everything');

    const view = gui.addFolder('Visibility');
    view.add(params, 'showGaussians').name('Show Gaussians').onChange(v => {
        gaussianCloud.visible = v && currentStep < 5;
    });
    view.add(params, 'showPoints').name('Show Points').onChange(v => pointCloud.visible = v && currentStep >= 2);
    view.add(params, 'showVoxels').name('Show Voxels').onChange(v => voxelCloud.visible = v && currentStep >= 3);
    view.add(params, 'showMesh').name('Show Mesh').onChange(v => {
        marchingCubes.visible = v && currentStep === 5;
        marchingWireframe.visible = v && currentStep === 5;
    });
    view.add(params, 'showCamera').name('Show Camera').onChange(v => cameraHelpers.forEach(h => h.visible = v));
    view.add(params, 'showSlice').name('Show Slice').onChange(v => {
        slicePlane.visible = v && currentStep >= 4;
        if (v && distancesBuffer) updateSliceTexture(distancesBuffer);
    });
    view.add(params, 'slicePos', -1.9, 1.9).name('Slice Position').onChange(() => {
        slicePlane.position.x = params.slicePos;
        if (distancesBuffer) updateSliceTexture(distancesBuffer);
    });
    
    steps.open();
}

function initWorker() {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    sendInitToWorker();

    worker.onmessage = (e) => {
        const { type, data } = e.data;
        switch (type) {
            case 'ready': updateStatus('System Ready. Start with Step 1.'); break;

            // Step 1: one depth map per camera arrives progressively
            case 'depth_progress': {
                const { camIndex, depthMap } = data;
                const icon = cameraHelpers[camIndex];
                if (!icon?.depthCtx) break;

                const ctx = icon.depthCtx;
                const imgData = ctx.createImageData(DEPTH_RES, DEPTH_RES);
                let minD = Infinity, maxD = -Infinity;
                for (let i = 0; i < depthMap.length; i++) {
                    if (depthMap[i] >= 0) { minD = Math.min(minD, depthMap[i]); maxD = Math.max(maxD, depthMap[i]); }
                }
                for (let i = 0; i < depthMap.length; i++) {
                    const d = depthMap[i];
                    const idx = i * 4;
                    if (d < 0) {
                        imgData.data[idx] = 13; imgData.data[idx + 1] = 17; imgData.data[idx + 2] = 23; imgData.data[idx + 3] = 200;
                    } else {
                        // near = bright, far = dark (standard depth map convention)
                        const t = maxD > minD ? 1 - (d - minD) / (maxD - minD) : 0.5;
                        const v = Math.floor(t * 255);
                        imgData.data[idx] = v; imgData.data[idx + 1] = v; imgData.data[idx + 2] = v; imgData.data[idx + 3] = 255;
                    }
                }
                ctx.putImageData(imgData, 0, 0);
                icon.depthTexture.needsUpdate = true;
                updateStatus(`Step 1: Rendering depth maps... camera ${camIndex + 1} / ${cameraHelpers.length}`);
                break;
            }

            // Step 1 complete: store points, do NOT show yet (step 2 reveals them)
            case 'points':
                pendingPoints = data;
                updateStatus('Step 1 Complete: All depth maps rendered. Run Step 2 to reveal point cloud.');
                break;

            case 'voxels': renderVoxels(data); updateStatus('Step 3 Complete: Affected Voxels Identified.'); break;

            case 'fusing_progress': {
                const percent = (data.progress * 100).toFixed(0);
                updateStatus(`Step 4: Fusing Logic... ${percent}%`);

                // Step 4 focuses on the TSDF distance field — force slice plane, hide mesh
                marchingCubes.visible = false;
                marchingWireframe.visible = false;
                slicePlane.visible = true;
                updateSliceTexture(data.distances);

                if (data.isDone) {
                    distancesBuffer = data.distances;
                    updateStatus(`Step 4 Complete: TSDF field built in ${data.time.toFixed(2)}ms. Run Step 5 to extract mesh.`);
                }
                break;
            }

            case 'extracted':
                distancesBuffer = data;
                // Step 5: Gaussians → Mesh — the key visual transformation
                gaussianCloud.visible = false; // always hide regardless of showGaussians toggle
                slicePlane.visible = false;
                renderMesh(data);
                marchingCubes.visible = params.showMesh;
                marchingWireframe.visible = params.showMesh;
                updateStatus('Step 5 Complete: Mesh Extracted via Marching Cubes.');
                break;

            case 'reset_done':
                currentStep = 0;
                pendingPoints = null;
                scene.remove(gaussianCloud);
                gaussianCloud = createGaussianCloud(params.type);
                gaussianCloud.visible = params.showGaussians;
                scene.add(gaussianCloud);
                pointCloud.visible = false;
                voxelCloud.visible = false;
                marchingCubes.visible = false;
                marchingWireframe.visible = false;
                slicePlane.visible = false;
                cameraHelpers.forEach(h => {
                    h.visible = params.showCamera;
                    // Clear depth map texture
                    if (h.depthCtx) {
                        h.depthCtx.fillStyle = '#0d1117';
                        h.depthCtx.fillRect(0, 0, DEPTH_RES, DEPTH_RES);
                        h.depthTexture.needsUpdate = true;
                    }
                });
                distancesBuffer = null;
                updateStatus('Reset Complete. Ready for Step 1.');
                break;
        }
    };
}

function triggerStep(step) {
    if (step === 1) {
        currentStep = 1;
        sendInitToWorker();
        worker.postMessage({ type: 'step1_render' });
    } else if (step === 2) {
        if (currentStep < 1) return updateStatus('Error: Complete Step 1 first.');
        if (!pendingPoints) return updateStatus('Error: Step 1 not finished yet.');
        currentStep = 2;
        renderPoints(pendingPoints);
        updateStatus('Step 2 Complete: Point cloud from depth map back-projection.');
    } else if (step === 3) {
        if (currentStep < 2) return updateStatus('Error: Complete Step 2 first.');
        currentStep = 3;
        worker.postMessage({ type: 'step3_voxelize' });
    } else if (step === 4) {
        if (currentStep < 3) return updateStatus('Error: Complete Step 3 first.');
        currentStep = 4;
        marchingCubes.visible = false;
        marchingWireframe.visible = false;
        worker.postMessage({ type: 'step4_fuse' });
    } else if (step === 5) {
        if (currentStep < 4) return updateStatus('Error: Complete Step 4 first.');
        currentStep = 5;
        worker.postMessage({ type: 'step5_extract' });
    }
}

function renderPoints(data) {
    const numPoints = data.length / 7;
    const positions = new Float32Array(numPoints * 3);
    for (let i = 0; i < numPoints; i++) {
        positions[i * 3] = data[i * 7];
        positions[i * 3 + 1] = data[i * 7 + 1];
        positions[i * 3 + 2] = data[i * 7 + 2];
    }
    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointCloud.visible = params.showPoints;
}

function renderVoxels(data) {
    voxelCloud.geometry.setAttribute('position', new THREE.BufferAttribute(data, 3));
    voxelCloud.visible = params.showVoxels;
}

function renderMesh(distances) {
    marchingCubes.field.set(distances);
    marchingCubes.isolation = 0.0;
    marchingCubes.update();
    // marchingWireframe shares the same geometry and updates automatically
}

function resetDemo() {
    worker.postMessage({ type: 'reset' });
}

function sendInitToWorker() {
    worker.postMessage({
        type: 'init',
        data: {
            size: params.resolution,
            extent: 2.0,
            mu: params.mu,
            type: params.type,
            noise: params.noise,
            outliers: params.outliers,
        },
    });
}

function updateStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
