import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import GUI from 'lil-gui';

/**
 * TSDF Demo - Step-by-Step Educational Version
 */

const params = {
    resolution: 64,
    muVoxels: 4,
    observationWeight: 1.0,
    maxTSDFWeight: 32,
    noise: 0.05,
    outliers: 0.02,
    gaussianCount: 500,
    gsplatNoise: 0.5,
    showGaussians: true,
    showPoints: true,
    showVoxels: true,
    showMesh: true,
    showCamera: false,
    showGrids: true,
    showKnowledge: () => toggleModal(true),
    cameraZoom: 5.2, // initial distance = length of (3,3,3)
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
let marchingCubes, marchingWireframe, pointCloud, voxelCloud;
let gaussianClouds = [];
let gridHelper = null;
let cameraHelpers = [];
let scanLines = []; // animated rays during step 1

// Auto-play state
const autoPlayState = { playing: false, paused: false };
let resumeResolve = null;
const workerResolvers = new Map(); // type → resolve, for async worker communication

// What the current step/play WANTS to show — independent of user toggle permissions.
// gaussians: -1 = hide all, 'all' = show all sectors, 0–7 = show specific sector
const stepWants = { gaussians: 'all', points: false, voxels: false, mesh: false, camera: false };

// Accumulated point positions from all previously completed cameras (auto-play only)
let accumulatedPointPositions = null;

let worker;
let currentStep = 0;
let distancesBuffer = null;
let pendingPoints = null; // point cloud from step 1, shown in step 2

const POINT_REVEAL_DURATION_MS = 1100;
const VOXEL_REVEAL_DURATION_MS = 850;
const pointRevealState = {
    active: false,
    startTime: 0,
    durationMs: POINT_REVEAL_DURATION_MS,
    totalPoints: 0,
    currentCount: 0,
    prevCount: 0, // static portion from accumulated cameras (auto-play)
};
const voxelRevealState = {
    active: false,
    startTime: 0,
    durationMs: VOXEL_REVEAL_DURATION_MS,
    totalPoints: 0,
    currentCount: 0,
};
const voxelIntegrationState = {
    active: false,
    progress: 0,
    pulsePhase: 0,
};

const DEPTH_RES = 48; // must match worker.js

/**
 * Build 8 sector meshes representing the 3DGS scene, one per camera viewing angle.
 * Each mesh contains the gaussians whose XZ angle falls in that camera's field of view
 * (sector c = the 45° wedge opposite camera c).  This matches the angle-based assignment
 * in the worker so visual sectors and computed point clouds are always consistent.
 * @returns {THREE.InstancedMesh[]} Array of 8 meshes
 */
function createGaussianCloud(type, N = params.gaussianCount, noise = params.gsplatNoise) {
    const geo = new THREE.SphereGeometry(1, 5, 4);
    const baseMat = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.40,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
    });

    const SECTOR_COUNT = CAM_COUNT_MAIN;
    const sectorSize = (Math.PI * 2) / SECTOR_COUNT;

    const sectors = Array.from({ length: SECTOR_COUNT }, () => ({
        matrices: [],
        colors: [],
        rawPoints: [],
    }));

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const normal = new THREE.Vector3();

    for (let i = 0; i < N; i++) {
        let x, y, z, nx, ny, nz;
        if (type === 'sphere') {
            const phi   = Math.random() * Math.PI * 2;
            const theta = Math.acos(2 * Math.random() - 1);
            // Y-up convention: equator in XZ plane, poles at ±Y — matches camera ring
            nx = Math.sin(theta) * Math.cos(phi);
            ny = Math.cos(theta);
            nz = Math.sin(theta) * Math.sin(phi);
            x = nx; y = ny; z = nz;
        } else {
            const phi   = Math.random() * Math.PI * 2;
            const theta = Math.random() * Math.PI * 2;
            const R = 1.0, r = 0.4;
            // Y-up convention: torus ring in XZ plane — matches sdTorus in worker
            x  = (R + r * Math.cos(theta)) * Math.cos(phi);
            y  = r * Math.sin(theta);
            z  = (R + r * Math.cos(theta)) * Math.sin(phi);
            nx = Math.cos(theta) * Math.cos(phi);
            ny = Math.sin(theta);
            nz = Math.cos(theta) * Math.sin(phi);
        }

        const baseHue = (Math.atan2(nz, nx) / (Math.PI * 2) + 0.5 + ny * 0.15) % 1.0;
        const roll = Math.random();
        let sx, sy, sz, hue, sat, lig;

        if (roll < 0.08 * noise) {
            const drift = (0.2 + Math.random() * 0.6) * noise;
            x += nx * drift + (Math.random() - 0.5) * 0.25 * noise;
            y += ny * drift + (Math.random() - 0.5) * 0.25 * noise;
            z += nz * drift + (Math.random() - 0.5) * 0.25 * noise;
            sx = sz = 0.025 + Math.random() * 0.05;
            sy = 0.025 + Math.random() * 0.05;
            hue = Math.random();
            sat = 0.25; lig = 0.20 + Math.random() * 0.12;
        } else if (roll < (0.08 + 0.10) * noise) {
            x += (Math.random() - 0.5) * 0.04 * noise;
            y += (Math.random() - 0.5) * 0.04 * noise;
            z += (Math.random() - 0.5) * 0.04 * noise;
            sx = 0.18 + Math.random() * 0.18;
            sy = 0.006 + Math.random() * 0.006;
            sz = 0.008 + Math.random() * 0.01;
            hue = baseHue;
            sat = 0.55; lig = 0.35 + Math.random() * 0.2;
        } else if (roll < (0.08 + 0.10 + 0.07) * noise) {
            x += (Math.random() - 0.5) * 0.05 * noise;
            y += (Math.random() - 0.5) * 0.05 * noise;
            z += (Math.random() - 0.5) * 0.05 * noise;
            sx = 0.14 + Math.random() * 0.14;
            sy = 0.05 + Math.random() * 0.05;
            sz = 0.14 + Math.random() * 0.14;
            hue = (baseHue + 0.04) % 1.0;
            sat = 0.35; lig = 0.25 + Math.random() * 0.12;
        } else {
            x += (Math.random() - 0.5) * 0.06 * noise;
            y += (Math.random() - 0.5) * 0.06 * noise;
            z += (Math.random() - 0.5) * 0.06 * noise;
            sx = 0.05 + Math.random() * 0.10;
            sy = 0.012 + Math.random() * 0.022;
            sz = 0.03  + Math.random() * 0.08;
            hue = baseHue;
            sat = 0.65 + Math.random() * 0.25;
            lig = 0.55 + Math.random() * 0.20;
        }

        // Sector assignment — must match worker's getAngleSector formula
        const normalizedAngle = ((Math.atan2(z, x) + Math.PI * 2) % (Math.PI * 2));
        const sectorIdx = Math.floor(normalizedAngle / sectorSize);

        sectors[sectorIdx].rawPoints.push(x, y, z, nx, ny, nz);

        dummy.position.set(x, y, z);
        dummy.scale.set(sx, sy, sz);
        normal.set(nx, ny, nz).normalize();
        dummy.quaternion.setFromUnitVectors(yAxis, normal);
        dummy.rotateOnWorldAxis(normal, Math.random() * Math.PI * 2);
        dummy.updateMatrix();
        sectors[sectorIdx].matrices.push(dummy.matrix.clone());

        color.setHSL(hue, sat, lig);
        sectors[sectorIdx].colors.push(color.clone());
    }

    return sectors.map(sector => {
        const count = sector.matrices.length;
        const mesh = new THREE.InstancedMesh(geo, baseMat.clone(), count || 1);
        mesh.count = count;
        if (count > 0) {
            sector.matrices.forEach((mat, i) => mesh.setMatrixAt(i, mat));
            sector.colors.forEach((c, i) => mesh.setColorAt(i, c));
            mesh.instanceMatrix.needsUpdate = true;
            if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        }
        mesh.rawPoints = new Float32Array(sector.rawPoints);
        return mesh;
    });
}

/** Merge all sector rawPoints into a single Float32Array for the worker. */
function getGaussianRawPoints() {
    const totalLen = gaussianClouds.reduce((s, m) => s + m.rawPoints.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const m of gaussianClouds) { merged.set(m.rawPoints, off); off += m.rawPoints.length; }
    return merged;
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

// ---------------------------------------------------------------------------
// Scan-ray helpers (Step 1 animation)
// ---------------------------------------------------------------------------

const CAM_COUNT_MAIN = 8;
const CAM_RADIUS_MAIN = 2.5;
const CAM_HEIGHT_MAIN = 1.0;

/** Remove all scan rays from the scene and restore camera icon colours. */
function clearScanRays() {
    scanLines.forEach(r => scene.remove(r.line));
    scanLines = [];
    cameraHelpers.forEach(h => {
        if (h.children[0]) h.children[0].material.color.setHex(0x88c0d0);
    });
}

/**
 * Compute N target points on the unit sphere that are visible from camera camIdx.
 * Uses analytic ray-sphere intersection so points land exactly on the surface.
 */
function getSurfaceTargets(camIdx, n) {
    const angle = (camIdx / CAM_COUNT_MAIN) * Math.PI * 2;
    const camPos = new THREE.Vector3(
        CAM_RADIUS_MAIN * Math.cos(angle),
        CAM_HEIGHT_MAIN,
        CAM_RADIUS_MAIN * Math.sin(angle),
    );
    const fwd   = camPos.clone().negate().normalize();
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x).normalize();
    const up    = new THREE.Vector3().crossVectors(right, fwd).normalize();

    const targets = [];
    for (let i = 0; i < n; i++) {
        const phi    = (i / n) * Math.PI * 2;
        const spread = i === 0 ? 0 : 0.38; // centre ray + ring
        const dir = fwd.clone()
            .addScaledVector(right, Math.cos(phi) * spread)
            .addScaledVector(up,    Math.sin(phi) * spread)
            .normalize();

        // Ray-sphere intersection (unit sphere at origin)
        const b    = camPos.dot(dir);
        const disc = b * b - (camPos.lengthSq() - 1);
        if (disc >= 0) {
            const t = -b - Math.sqrt(disc);
            if (t > 0) { targets.push(camPos.clone().addScaledVector(dir, t)); continue; }
        }
        targets.push(dir.clone()); // fallback: unit-distance point in ray direction
    }
    return targets;
}

/**
 * Spawn animated scan rays from camera camIdx toward the target surface.
 * Previous camera rays are kept visible (building up the scan picture).
 */
function createScanRays(camIdx) {
    // Highlight active camera in yellow; leave done cameras frost-blue
    cameraHelpers.forEach((h, i) => {
        if (h.children[0]) {
            h.children[0].material.color.setHex(i === camIdx ? 0xebcb8b : 0x88c0d0);
        }
    });

    const camPos  = cameraHelpers[camIdx].position.clone();
    const targets = getSurfaceTargets(camIdx, 7);

    targets.forEach(endPos => {
        const positions = new Float32Array([
            camPos.x, camPos.y, camPos.z, // start (fixed)
            camPos.x, camPos.y, camPos.z, // end   (animates toward endPos)
        ]);
        const geo  = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const line = new THREE.Line(
            geo,
            new THREE.LineBasicMaterial({ color: 0xa3be8c, transparent: true, opacity: 0.75 }),
        );
        scene.add(line);
        scanLines.push({ line, startPos: camPos.clone(), endPos, progress: 0 });
    });
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

    // Helpers — grid divisions match the voxel resolution so each cell = one voxel column
    gridHelper = new THREE.GridHelper(4, params.resolution, 0x4c566a, 0x3b4252);
    gridHelper.visible = params.showGrids;
    scene.add(gridHelper);

    // Gaussian cloud — represents the 3DGS scene the virtual cameras will scan
    gaussianClouds = createGaussianCloud(params.type);
    gaussianClouds.forEach(m => scene.add(m));

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

/** Actual truncation distance in world units: muVoxels × voxelSize. */
function getMu() {
    return params.muVoxels * (4.0 / params.resolution);
}

function easeOutCubic(t) {
    return 1 - (1 - t) ** 3;
}

function stopPointReveal({ showAll = true } = {}) {
    pointRevealState.active = false;
    pointRevealState.startTime = 0;
    if (showAll && pointRevealState.totalPoints > 0) {
        pointCloud.geometry.setDrawRange(0, pointRevealState.totalPoints);
        pointRevealState.currentCount = pointRevealState.totalPoints;
    }
    pointCloud.material.opacity = 0.8;
    pointCloud.material.size = 0.02;
}

function stopVoxelReveal({ showAll = true } = {}) {
    voxelRevealState.active = false;
    voxelRevealState.startTime = 0;
    if (showAll && voxelRevealState.totalPoints > 0) {
        voxelCloud.geometry.setDrawRange(0, voxelRevealState.totalPoints);
        voxelRevealState.currentCount = voxelRevealState.totalPoints;
    }
    voxelCloud.material.opacity = params.showVoxels ? 0.4 : voxelCloud.material.opacity;
    voxelCloud.material.size = 0.03;
}

function stopVoxelIntegration() {
    voxelIntegrationState.active = false;
    voxelIntegrationState.progress = 0;
    voxelIntegrationState.pulsePhase = 0;
    voxelCloud.material.color.setHex(0xebcb8b);
    voxelCloud.material.opacity = params.showVoxels ? 0.4 : voxelCloud.material.opacity;
    voxelCloud.material.size = 0.03;
}

/**
 * Color each active voxel by its TSDF scalar value using a diverging colormap:
 *   blue  (t = -1) → white (t = 0, zero-crossing / surface) → red (t = +1)
 *   near-black = unobserved (d ≈ 1.0)
 * PointsMaterial has no per-vertex alpha, so brightness encodes importance.
 */
function updateVoxelColors(distances) {
    const posAttr = voxelCloud.geometry.attributes.position;
    if (!posAttr) return;

    const N = posAttr.count;
    const extent = 2.0;
    const size = params.resolution;
    const voxelSize = (extent * 2) / size;
    const mu = getMu();
    const sizeSq = size * size;

    const colorData = new Float32Array(N * 3);

    for (let i = 0; i < N; i++) {
        const wx = posAttr.getX(i);
        const wy = posAttr.getY(i);
        const wz = posAttr.getZ(i);

        const ix = Math.floor((wx + extent) / voxelSize);
        const iy = Math.floor((wy + extent) / voxelSize);
        const iz = Math.floor((wz + extent) / voxelSize);
        const idx = ix + iy * size + iz * sizeSq;
        const d = (idx >= 0 && idx < distances.length) ? distances[idx] : 1.0;

        let r, g, b;
        if (d >= 0.99) {
            // Unobserved — near-black so it fades into the background
            r = 0.12; g = 0.13; b = 0.16;
        } else {
            const t = Math.max(-1, Math.min(1, d / mu));
            if (t <= 0) {
                // Negative SDF (behind surface): white → Nord10 blue #5e81ac
                const s = -t;
                r = 1.0 - s * (1.0 - 0.369);
                g = 1.0 - s * (1.0 - 0.506);
                b = 1.0 - s * (1.0 - 0.675);
            } else {
                // Positive SDF (in front of surface): white → Nord11 red #bf616a
                r = 1.0 - t * (1.0 - 0.749);
                g = 1.0 - t * (1.0 - 0.380);
                b = 1.0 - t * (1.0 - 0.416);
            }
        }
        colorData[i * 3]     = r;
        colorData[i * 3 + 1] = g;
        colorData[i * 3 + 2] = b;
    }

    voxelCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));
    voxelCloud.material.vertexColors = true;
    voxelCloud.material.color.set(0xffffff); // must be white so it doesn't tint vertex colors
    voxelCloud.material.opacity = 0.80;
    voxelCloud.material.size = 0.042;
    voxelCloud.material.needsUpdate = true;
}

function updateVoxelIntegration(progress) {
    voxelIntegrationState.active = progress < 1;
    voxelIntegrationState.progress = progress;
    voxelCloud.material.opacity = 0.22 + progress * 0.28;
    voxelCloud.material.size = 0.03 + progress * 0.012;
    voxelCloud.material.color.setRGB(
        0.92 + progress * 0.08,
        0.80 + progress * 0.14,
        0.30 + progress * 0.18,
    );
}

function setupGUI() {
    const gui = new GUI();
    
    gui.add(params, 'showKnowledge').name('💡 Knowledge Base');
    gui.add(params, 'cameraZoom', 0.8, 6, 0.1).name('🔍 Zoom').onChange(v => {
        const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
        offset.setLength(v);
        camera.position.copy(controls.target).add(offset);
        controls.update();
    });
    let playPauseCtrl;
    params.togglePlay = () => {
        if (!autoPlayState.playing) startAutoPlay();
        else togglePause();
    };
    playPauseCtrl = gui.add(params, 'togglePlay').name('▶ Auto Play');
    window._playPauseCtrl = playPauseCtrl;

    // Shared voxel grid resolution — must be identical for TSDF volume and MarchingCubes
    // because MC reads the TSDF distances array directly (marchingCubes.field.set(distances)).
    params.voxelSizeDisplay = `${(4.0 / params.resolution).toFixed(4)}`;
    gui.add(params, 'resolution', 32, 96, 16).name('Grid Resolution').onChange(() => {
        params.voxelSizeDisplay = `${(4.0 / params.resolution).toFixed(4)}`;
        voxelSizeCtrl.updateDisplay();

        // Rebuild grid so divisions = resolution (each cell = one voxel column)
        scene.remove(gridHelper);
        gridHelper = new THREE.GridHelper(4, params.resolution, 0x4c566a, 0x3b4252);
        gridHelper.visible = params.showGrids;
        scene.add(gridHelper);

        marchingCubes.reset();
        const nextScale = marchingCubes.scale.clone();
        scene.remove(marchingCubes);
        scene.remove(marchingWireframe);

        const material = marchingCubes.material;
        marchingCubes = new MarchingCubes(params.resolution, material, true, true, 100000);
        marchingCubes.scale.copy(nextScale);
        marchingCubes.visible = params.showMesh;
        scene.add(marchingCubes);

        marchingWireframe = new THREE.Mesh(
            marchingCubes.geometry,
            new THREE.MeshBasicMaterial({ color: 0x4c566a, wireframe: true, transparent: true, opacity: 0.35 })
        );
        marchingWireframe.scale.copy(nextScale);
        marchingWireframe.visible = params.showMesh;
        scene.add(marchingWireframe);
    });
    const voxelSizeCtrl = gui.add(params, 'voxelSizeDisplay').name('↳ Voxel Size').disable();

    // --- Gaussian Splat (affects Step 1) ---
    const gsplatFolder = gui.addFolder('Gaussian Splat (Step 1)');
    gsplatFolder.add(params, 'type', ['sphere', 'torus']).name('Surface Type').onChange(v => {
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(v);
        gaussianClouds.forEach(m => scene.add(m));
        applyVis();
    });
    gsplatFolder.add(params, 'gaussianCount', 100, 1000, 50).name('Gaussian Count').onChange(() => {
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(params.type);
        gaussianClouds.forEach(m => scene.add(m));
        applyVis();
    });
    gsplatFolder.add(params, 'gsplatNoise', 0.0, 1.0, 0.05).name('Splat Noise').onChange(() => {
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(params.type);
        gaussianClouds.forEach(m => scene.add(m));
        applyVis();
    });

    // --- TSDF Volume (Step 4 params; mu also governs Step 3 active region) ---
    const tsdfFolder = gui.addFolder('TSDF Volume (Steps 3–4)');
    tsdfFolder.add(params, 'muVoxels', 1, 10, 0.5).name('Truncation μ');
    tsdfFolder.add(params, 'observationWeight', 0.1, 4.0, 0.1).name('Observation Weight');
    tsdfFolder.add(params, 'maxTSDFWeight', 1, 128, 1).name('Max TSDF Weight');

    const steps = gui.addFolder('Steps');
    const stepCtrls = [
        steps.add(params, 'step1').name('1. Render Depth Maps'),
        steps.add(params, 'step2').name('2. Reconstruct Point Cloud'),
        steps.add(params, 'step3').name('3. Mark Active Voxels'),
        steps.add(params, 'step4').name('4. Integrate TSDF'),
        steps.add(params, 'step5').name('5. Extract Mesh (Marching Cubes)'),
    ];
    steps.add(params, 'reset').name('Reset Everything');
    window._stepCtrls = stepCtrls;
    window._setStepButtonsEnabled = (enabled) => stepCtrls.forEach(c => enabled ? c.enable() : c.disable());

    const view = gui.addFolder('Visibility');
    view.add(params, 'showGrids').name('Show Grids')
        .onChange(v => { gridHelper.visible = v; });
    const ctrlGaussians = view.add(params, 'showGaussians').name('Show Gaussians')
        .onChange(() => applyVis());
    const ctrlCamera = view.add(params, 'showCamera').name('Show Camera')
        .onChange(() => applyVis());
    const ctrlPoints = view.add(params, 'showPoints').name('Show Points')
        .onChange(() => applyVis());
    const ctrlVoxels = view.add(params, 'showVoxels').name('Show Voxels')
        .onChange(() => applyVis());
    const ctrlMesh = view.add(params, 'showMesh').name('Show Mesh')
        .onChange(() => applyVis());


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
                createScanRays(camIndex);
                updateStatus(`Step 1: Rendering depth maps... camera ${camIndex + 1} / ${cameraHelpers.length}`);
                break;
            }

            // Step 1 complete: store points, restore camera colours, keep rays visible
            case 'points':
                pendingPoints = data;
                cameraHelpers.forEach(h => {
                    if (h.children[0]) h.children[0].material.color.setHex(0x88c0d0);
                });
                updateStatus('Step 1 Complete: All depth maps rendered. Run Step 2 to reconstruct the point cloud.');
                break;

            case 'voxels':
                renderVoxels(data);
                if (workerResolvers.has('voxels')) {
                    workerResolvers.get('voxels')(data);
                    workerResolvers.delete('voxels');
                }
                break;

            case 'fusing_progress': {
                const percent = (data.progress * 100).toFixed(0);
                distancesBuffer = data.distances;

                // Highlight cameras: green = fused, yellow = active, blue = pending
                if (data.cameraIndex !== undefined) {
                    cameraHelpers.forEach((h, i) => {
                        if (h.children[0]) {
                            const color = i < data.cameraIndex ? 0xa3be8c  // Nord14 green — done
                                        : i === data.cameraIndex ? 0xebcb8b // Nord13 yellow — active
                                        : 0x88c0d0;                          // Nord8 blue — pending
                            h.children[0].material.color.setHex(color);
                        }
                    });
                }

                if (data.isDone) {
                    // Restore all camera icons to default colour
                    cameraHelpers.forEach(h => {
                        if (h.children[0]) h.children[0].material.color.setHex(0x88c0d0);
                    });
                    stopVoxelIntegration();
                    updateVoxelColors(data.distances);
                    updateStatus(`Step 4 Complete: TSDF field integrated in ${data.time.toFixed(2)}ms. Run Step 5 to extract the mesh.`);
                } else {
                    updateVoxelIntegration(data.progress);
                    updateStatus(`Step 4: Integrating TSDF... camera ${data.cameraIndex + 1} / ${cameraHelpers.length} (${percent}%)`);
                }
                break;
            }

            case 'extracted': {
                distancesBuffer = data;
                renderMesh(data);
                setVis('gaussians', false);
                setVis('points', false);
                setVis('voxels', false);
                setVis('mesh', true);
                updateStatus('Step 5 Complete: Mesh Extracted via Marching Cubes.');
                break;
            }

            case 'play_ready':
                if (workerResolvers.has('play_ready')) {
                    workerResolvers.get('play_ready')(data);
                    workerResolvers.delete('play_ready');
                }
                break;

            case 'play_render_done':
                if (workerResolvers.has('play_render_done')) {
                    workerResolvers.get('play_render_done')(data);
                    workerResolvers.delete('play_render_done');
                }
                break;

            case 'play_fuse_done':
                distancesBuffer = data.distances;
                updateVoxelColors(data.distances);
                if (workerResolvers.has('play_fuse_done')) {
                    workerResolvers.get('play_fuse_done')(data);
                    workerResolvers.delete('play_fuse_done');
                }
                break;

            case 'reset_done': {
                currentStep = 0;
                pendingPoints = null;
                distancesBuffer = null;
                accumulatedPointPositions = null;
                stopPointReveal({ showAll: false });
                stopVoxelReveal({ showAll: false });
                stopVoxelIntegration();
                highlightStep(0); // clears all highlights
                clearScanRays();
                gaussianClouds.forEach(m => scene.remove(m));
                gaussianClouds = createGaussianCloud(params.type);
                gaussianClouds.forEach(m => scene.add(m));
                setVis('gaussians', true);
                setVis('camera', false);
                setVis('points', false);
                setVis('voxels', false);
                setVis('mesh', false);
                cameraHelpers.forEach(h => {
                    if (h.depthCtx) {
                        h.depthCtx.fillStyle = '#0d1117';
                        h.depthCtx.fillRect(0, 0, DEPTH_RES, DEPTH_RES);
                        h.depthTexture.needsUpdate = true;
                    }
                });
                updateStatus('Reset Complete. Ready for Step 1.');
                // Abort any in-progress auto-play
                autoPlayState.playing = false;
                autoPlayState.paused = false;
                if (resumeResolve) { resumeResolve(); resumeResolve = null; }
                updatePlayButton();
                window._setStepButtonsEnabled?.(true);
                break;
            }
        }
    };
}


/**
 * Apply visibility: actual = user toggle (params.showX) AND step wants.
 * Called whenever either side changes.
 */
function applyVis() {
    gaussianClouds.forEach((m, i) => {
        const g = stepWants.gaussians;
        const show = g === 'all' || (Array.isArray(g) ? g.includes(i) : g === i);
        m.visible = params.showGaussians && show;
    });
    pointCloud.visible = params.showPoints && stepWants.points;
    voxelCloud.visible = params.showVoxels && stepWants.voxels;
    marchingCubes.visible = params.showMesh && stepWants.mesh;
    marchingWireframe.visible = params.showMesh && stepWants.mesh;
    cameraHelpers.forEach(h => { h.visible = params.showCamera && stepWants.camera; });
}

/** Set what the current step wants to show, then reapply. key matches stepWants fields. */
function setVis(key, value) {
    if (key === 'gaussians') {
        stepWants.gaussians = value ? 'all' : -1;
    } else {
        stepWants[key] = value;
    }
    applyVis();
}

function highlightStep(step) {
    window._stepCtrls.forEach((c, i) => {
        c.domElement.classList.toggle('active-step', i === step - 1);
    });
}

function triggerStep(step) {
    if (step === 1) {
        currentStep = 1;
        highlightStep(1);
        clearScanRays();
        setVis('gaussians', true);
        setVis('camera', true);
        setVis('points', false);
        setVis('voxels', false);
        setVis('mesh', false);
        sendInitToWorker();
        worker.postMessage({ type: 'step1_render' });

    } else if (step === 2) {
        if (currentStep < 1) return updateStatus('Error: Complete Step 1 first.');
        if (!pendingPoints) return updateStatus('Error: Step 1 not finished yet.');
        currentStep = 2;
        highlightStep(2);
        clearScanRays();
        setVis('gaussians', false);
        setVis('camera', false);
        setVis('points', true);
        setVis('voxels', false);
        setVis('mesh', false);
        renderPoints(pendingPoints);
        updateStatus('Step 2: Reconstructing point cloud...');

    } else if (step === 3) {
        if (currentStep < 2) return updateStatus('Error: Complete Step 2 first.');
        currentStep = 3;
        highlightStep(3);
        updateStatus('Step 3: Marking active TSDF voxels...');
        setVis('gaussians', false);
        setVis('camera', false);
        setVis('points', true);
        setVis('voxels', true);
        setVis('mesh', false);
        worker.postMessage({
            type: 'step3_voxelize',
            data: { resolution: params.resolution, mu: getMu() },
        });

    } else if (step === 4) {
        if (currentStep < 3) return updateStatus('Error: Complete Step 3 first.');
        currentStep = 4;
        highlightStep(4);
        stopVoxelReveal();
        updateStatus('Step 4: Integrating TSDF...');
        setVis('gaussians', false);
        setVis('camera', true);
        setVis('points', false);
        setVis('voxels', true);
        setVis('mesh', false);
        worker.postMessage({
            type: 'step4_fuse',
            data: {
                mu: getMu(),
                observationWeight: params.observationWeight,
                maxTSDFWeight: params.maxTSDFWeight,
            },
        });

    } else if (step === 5) {
        if (currentStep < 4) return updateStatus('Error: Complete Step 4 first.');
        currentStep = 5;
        highlightStep(5);
        worker.postMessage({ type: 'step5_extract' });
    }
}

function renderPoints(data) {
    const newCount = data.length / 7;
    const newPositions = new Float32Array(newCount * 3);
    for (let i = 0; i < newCount; i++) {
        newPositions[i * 3]     = data[i * 7];
        newPositions[i * 3 + 1] = data[i * 7 + 1];
        newPositions[i * 3 + 2] = data[i * 7 + 2];
    }

    // In auto-play, prepend accumulated positions from previous cameras
    const prevCount = accumulatedPointPositions ? accumulatedPointPositions.length / 3 : 0;
    let positions;
    if (prevCount > 0) {
        positions = new Float32Array(prevCount * 3 + newCount * 3);
        positions.set(accumulatedPointPositions, 0);
        positions.set(newPositions, prevCount * 3);
    } else {
        positions = newPositions;
    }

    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // Show accumulated portion instantly; new points animate in
    pointCloud.geometry.setDrawRange(0, prevCount);
    pointCloud.material.opacity = prevCount > 0 ? 0.8 : 0.18;
    pointCloud.material.size = prevCount > 0 ? 0.02 : 0.028;

    pointRevealState.active = true;
    pointRevealState.startTime = performance.now();
    pointRevealState.totalPoints = prevCount + newCount;
    pointRevealState.prevCount = prevCount;
    pointRevealState.currentCount = prevCount;
}

function renderVoxels(data) {
    voxelCloud.geometry.deleteAttribute('color');
    voxelCloud.material.vertexColors = false;
    voxelCloud.material.needsUpdate = true;
    voxelCloud.geometry.setAttribute('position', new THREE.BufferAttribute(data, 3));
    voxelCloud.geometry.setDrawRange(0, 0);
    voxelCloud.material.opacity = 0.08;
    voxelCloud.material.size = 0.055;
    voxelCloud.material.color.setHex(0xebcb8b);
    voxelRevealState.active = true;
    voxelRevealState.startTime = performance.now();
    voxelRevealState.totalPoints = data.length / 3;
    voxelRevealState.currentCount = 0;
    stopVoxelIntegration();
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

/** Show only sector c's gaussian mesh; hide all others. c = -1 hides all. */
function showSector(c) {
    stepWants.gaussians = c;
    applyVis();
}

/** Add sector c to the currently visible set (cumulative — previous sectors stay). */
function addSector(c) {
    const g = stepWants.gaussians;
    if (g === 'all') return;
    if (g === -1) {
        stepWants.gaussians = [c];
    } else if (Array.isArray(g)) {
        if (!g.includes(c)) g.push(c);
    } else {
        stepWants.gaussians = g === c ? [c] : [g, c];
    }
    applyVis();
}

/** Returns a Promise that resolves once a specific worker message type arrives. */
function waitForWorkerMsg(type) {
    return new Promise(resolve => workerResolvers.set(type, resolve));
}

/** If paused, waits until resume is clicked. */
function checkPause() {
    if (!autoPlayState.paused) return Promise.resolve();
    return new Promise(r => { resumeResolve = r; });
}

/** Wait until at least `ms` milliseconds have elapsed since `startTime`. */
function waitAtLeast(ms, startTime = performance.now()) {
    const remaining = ms - (performance.now() - startTime);
    return new Promise(r => setTimeout(r, Math.max(0, remaining)));
}

function updatePlayButton() {
    const ctrl = window._playPauseCtrl;
    if (!ctrl) return;
    if (!autoPlayState.playing) ctrl.name('▶ Auto Play');
    else if (autoPlayState.paused) ctrl.name('▶ Resume');
    else ctrl.name('⏸ Pause');
}

function togglePause() {
    if (!autoPlayState.playing) return;
    autoPlayState.paused = !autoPlayState.paused;
    if (!autoPlayState.paused && resumeResolve) {
        resumeResolve();
        resumeResolve = null;
    }
    updatePlayButton();
}

async function startAutoPlay() {
    if (autoPlayState.playing) return;
    autoPlayState.playing = true;
    autoPlayState.paused = false;
    updatePlayButton();
    window._setStepButtonsEnabled?.(false);

    accumulatedPointPositions = null;

    // Reset visuals
    setVis('gaussians', false);
    setVis('camera', false);
    setVis('points', false);
    setVis('voxels', false);
    setVis('mesh', false);
    clearScanRays();
    currentStep = 0;
    highlightStep(0);

    // Init worker: reset volume + build per-camera point clouds
    updateStatus('Auto Play: Initialising...');
    sendInitToWorker();
    worker.postMessage({ type: 'play_init' });
    const cameraPointsData = await waitForWorkerMsg('play_ready');

    // Pre-populate voxel cloud so SDF colours can update progressively (keep hidden until fuse step)
    worker.postMessage({ type: 'step3_voxelize', data: { resolution: params.resolution, mu: getMu() } });
    await waitForWorkerMsg('voxels');

    for (let c = 0; c < CAM_COUNT_MAIN; c++) {
        if (!autoPlayState.playing) break;

        // ── Step 1: sector gaussian + depth map ──────────────────────────────
        highlightStep(1);
        updateStatus(`Camera ${c + 1}/8 — Step 1: Rendering depth map...`);
        addSector(c);
        setVis('camera', true);
        cameraHelpers.forEach((h, i) => {
            if (h.children[0]) h.children[0].material.color.setHex(i === c ? 0xebcb8b : 0x88c0d0);
        });
        const t1 = performance.now();
        worker.postMessage({ type: 'play_render', data: { cameraIndex: c } });
        await waitForWorkerMsg('play_render_done');
        await waitAtLeast(3000, t1);
        await checkPause();

        // ── Step 2: point cloud ───────────────────────────────────────────────
        highlightStep(2);
        updateStatus(`Camera ${c + 1}/8 — Step 2: Point cloud`);
        renderPoints(cameraPointsData[c]);
        setVis('points', true);
        await waitAtLeast(3000);
        await checkPause();

        // Accumulate this camera's points for the next camera's display
        accumulatedPointPositions = pointCloud.geometry.attributes.position.array.slice();

        // ── Step 3: fuse ──────────────────────────────────────────────────────
        highlightStep(3);
        updateStatus(`Camera ${c + 1}/8 — Step 3: Fusing into TSDF volume...`);
        setVis('voxels', true);
        voxelIntegrationState.active = true;
        const t3 = performance.now();
        worker.postMessage({ type: 'play_fuse', data: { cameraIndex: c } });
        await waitForWorkerMsg('play_fuse_done');
        stopVoxelIntegration();
        await waitAtLeast(3000, t3);
        await checkPause();

        setVis('voxels', false); // hide voxels after fuse step
        cameraHelpers.forEach(h => {
            if (h.children[0]) h.children[0].material.color.setHex(0x88c0d0);
        });
    }

    if (autoPlayState.playing) {
        // Extract mesh
        highlightStep(5);
        updateStatus('All cameras fused. Extracting mesh...');
        worker.postMessage({ type: 'step5_extract' });
        // extracted handler will show the mesh
    }

    autoPlayState.playing = false;
    updatePlayButton();
    window._setStepButtonsEnabled?.(true);
}

function sendInitToWorker() {
    worker.postMessage({
        type: 'init',
        data: {
            size: params.resolution,
            extent: 2.0,
            mu: getMu(),
            observationWeight: params.observationWeight,
            maxTSDFWeight: params.maxTSDFWeight,
            type: params.type,
            noise: params.noise,
            outliers: params.outliers,
            // Pass gaussian positions so the worker can build a point cloud that
            // reflects the gsplat distribution (artifacts included).
            gaussianPoints: getGaussianRawPoints(),
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

    // Grow scan rays toward their target (step 1 animation)
    for (const r of scanLines) {
        if (r.progress >= 1) continue;
        r.progress = Math.min(1, r.progress + 0.06);
        const pos = r.line.geometry.attributes.position;
        pos.setXYZ(1,
            r.startPos.x + (r.endPos.x - r.startPos.x) * r.progress,
            r.startPos.y + (r.endPos.y - r.startPos.y) * r.progress,
            r.startPos.z + (r.endPos.z - r.startPos.z) * r.progress,
        );
        pos.needsUpdate = true;
    }

    if (pointCloud.visible && pointRevealState.active) {
        const elapsed = performance.now() - pointRevealState.startTime;
        const t = Math.min(1, elapsed / pointRevealState.durationMs);
        const eased = easeOutCubic(t);
        const prev = pointRevealState.prevCount;
        const newTotal = pointRevealState.totalPoints - prev;
        const nextCount = prev + Math.floor(newTotal * eased);

        if (nextCount !== pointRevealState.currentCount) {
            pointCloud.geometry.setDrawRange(0, nextCount);
            pointRevealState.currentCount = nextCount;
        }

        if (prev > 0) {
            // Accumulated points already at full opacity; just reveal new ones via draw range
            pointCloud.material.opacity = 0.8;
            pointCloud.material.size = 0.02;
        } else {
            pointCloud.material.opacity = 0.18 + eased * 0.62;
            pointCloud.material.size = 0.028 - eased * 0.008;
        }

        if (t >= 1) {
            stopPointReveal();
            updateStatus('Step 2 Complete: Point cloud reconstructed from the depth maps.');
        }
    }

    if (voxelCloud.visible && voxelRevealState.active) {
        const elapsed = performance.now() - voxelRevealState.startTime;
        const t = Math.min(1, elapsed / voxelRevealState.durationMs);
        const eased = easeOutCubic(t);
        const nextCount = Math.floor(voxelRevealState.totalPoints * eased);

        if (nextCount !== voxelRevealState.currentCount) {
            voxelCloud.geometry.setDrawRange(0, nextCount);
            voxelRevealState.currentCount = nextCount;
        }

        voxelCloud.material.opacity = 0.08 + eased * 0.32;
        voxelCloud.material.size = 0.055 - eased * 0.025;

        if (t >= 1) {
            stopVoxelReveal();
            updateStatus('Step 3 Complete: Active TSDF voxels identified.');
        }
    }

    if (voxelCloud.visible && voxelIntegrationState.active) {
        voxelIntegrationState.pulsePhase += 0.16;
        const pulse = 0.5 + 0.5 * Math.sin(voxelIntegrationState.pulsePhase);
        voxelCloud.material.size = 0.036 + voxelIntegrationState.progress * 0.006 + pulse * 0.004;
        voxelCloud.material.opacity = Math.max(voxelCloud.material.opacity, 0.30 + voxelIntegrationState.progress * 0.12 + pulse * 0.05);
    }

    renderer.render(scene, camera);
}
