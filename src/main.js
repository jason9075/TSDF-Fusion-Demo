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
    numPoints: 5000,
    showPoints: true,
    showVoxels: true,
    showMesh: true,
    showSlice: false,
    showKnowledge: () => toggleModal(true),
    slicePos: 0.0,
    type: 'sphere',
    
    // Step Buttons
    step1: () => triggerStep(1),
    step2: () => triggerStep(2),
    step3: () => triggerStep(3),
    step4: () => triggerStep(4),
    reset: () => resetDemo()
};

let scene, camera, renderer, controls;
let marchingCubes, pointCloud, voxelCloud, slicePlane;
let worker;
let currentStep = 0;
let distancesBuffer = null;

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

function toggleLanguage() {
    currentLang = currentLang === 'en' ? 'zh' : 'en';
    const enElements = document.querySelectorAll('.lang-en');
    const zhElements = document.querySelectorAll('.lang-zh');
    
    // Use a helper to toggle visibility
    const setVisibility = (elements, show) => {
        elements.forEach(el => {
            if (show) el.classList.remove('hidden');
            else el.classList.add('hidden');
        });
    };

    setVisibility(enElements, currentLang === 'en');
    setVisibility(zhElements, currentLang === 'zh');
    
    // Re-render math to ensure any newly visible math is processed
    renderMath();
}

function toggleModal(show) {
    const modal = document.getElementById('modal-overlay');
    if (show) {
        modal.classList.remove('hidden');
        // Ensure only one language is visible on start
        const enElements = document.querySelectorAll('.lang-en');
        const zhElements = document.querySelectorAll('.lang-zh');
        enElements.forEach(el => el.classList.toggle('hidden', currentLang !== 'en'));
        zhElements.forEach(el => el.classList.toggle('hidden', currentLang !== 'zh'));
        
        renderMath();
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
    const geometry = new THREE.PlaneGeometry(4, 4, params.resolution, params.resolution);
    const material = new THREE.ShaderMaterial({
        side: THREE.DoubleSide,
        transparent: true,
        uniforms: {
            u_res: { value: params.resolution },
            u_slice: { value: 0.5 },
            u_mu: { value: params.mu },
            u_distances: { value: new Float32Array(params.resolution**3) }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform float u_res;
            uniform float u_slice;
            uniform float u_mu;
            uniform float u_distances[4096]; // Limited by shader uniform limits
            varying vec2 vUv;

            void main() {
                // This is a simplified demo slicing. In a real production app, 
                // we would use a 3D Texture for efficiency.
                gl_FragColor = vec4(vUv, 0.5, 0.4);
            }
        `
    });

    // Actually, passing a huge array to uniforms is bad. 
    // Let's use a simpler approach for the slice: a Canvas texture.
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

    const settings = gui.addFolder('Settings');
    settings.add(params, 'type', ['sphere', 'torus']).name('Surface Type').onChange(resetDemo);
    settings.add(params, 'mu', 0.05, 0.4).name('Truncation (μ)');
    settings.add(params, 'noise', 0, 0.2).name('Noise');
    settings.add(params, 'numPoints', 1000, 20000).name('Points Count');
    
    const steps = gui.addFolder('Steps');
    steps.add(params, 'step1').name('1. Generate Points');
    steps.add(params, 'step2').name('2. Voxelize');
    steps.add(params, 'step3').name('3. Fuse Logic');
    steps.add(params, 'step4').name('4. Extract Mesh');
    steps.add(params, 'reset').name('Reset Everything');

    const view = gui.addFolder('Visibility');
    view.add(params, 'showPoints').name('Show Points').onChange(v => pointCloud.visible = v && currentStep >= 1);
    view.add(params, 'showVoxels').name('Show Voxels').onChange(v => voxelCloud.visible = v && currentStep >= 2);
    view.add(params, 'showMesh').name('Show Mesh').onChange(v => marchingCubes.visible = v && currentStep >= 4);
    view.add(params, 'showSlice').name('Show Slice').onChange(v => {
        slicePlane.visible = v && currentStep >= 3;
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
            case 'points': renderPoints(data); updateStatus('Step 1 Complete: Points Generated.'); break;
            case 'voxels': renderVoxels(data); updateStatus('Step 2 Complete: Affected Voxels Identified.'); break;
            case 'fused': 
                updateStatus(`Step 3 Complete: TSDF Fusion done in ${data.time.toFixed(2)}ms.`); 
                worker.postMessage({ type: 'step4_extract' }); // Get data for slice but don't show mesh yet
                break;
            case 'extracted':
                distancesBuffer = data;
                if (currentStep === 4) {
                    renderMesh(data);
                    updateStatus('Step 4 Complete: Mesh Extracted via Marching Cubes.');
                }
                if (params.showSlice) updateSliceTexture(data);
                break;
            case 'reset_done':
                currentStep = 0;
                pointCloud.visible = false;
                voxelCloud.visible = false;
                marchingCubes.visible = false;
                slicePlane.visible = false;
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
        worker.postMessage({ type: 'step1_generate' });
    } else if (step === 2) {
        if (currentStep < 1) return updateStatus('Error: Please generate points first (Step 1).');
        currentStep = 2;
        worker.postMessage({ type: 'step2_voxelize' });
    } else if (step === 3) {
        if (currentStep < 2) return updateStatus('Error: Please run voxelization first (Step 2).');
        currentStep = 3;
        worker.postMessage({ type: 'step3_fuse' });
    } else if (step === 4) {
        if (currentStep < 3) return updateStatus('Error: Please run fusion first (Step 3).');
        currentStep = 4;
        worker.postMessage({ type: 'step4_extract' });
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
    for (let i = 0; i < distances.length; i++) {
        marchingCubes.field[i] = distances[i];
    }
    marchingCubes.isolation = 0.0;
    marchingCubes.update();
    marchingCubes.visible = params.showMesh;
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
            numPoints: params.numPoints,
            noise: params.noise,
            outliers: params.outliers
        } 
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
