import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import GUI from 'lil-gui';
import { DEPTH_RES } from './constants.js';

/**
 * TSDF Demo - Step-by-Step Educational Version
 */

const params = {
    resolution: 64,
    muVoxels: 4,
    observationWeight: 1.0,
    maxTSDFWeight: 8,
    mcWeightThreshold: 1,
    noise: 0.05,
    gaussianCount: 500,
    gsplatNoise: 0.5,
    showGaussians: true,
    showPoints: true,
    showFrameVoxels: true,
    showGlobalVoxels: true,
    showMesh: true,
    showCamera: true,
    showGrids: true,
    showKnowledge: () => toggleModal(true),
    cameraZoom: 4, // initial distance = length of (3,3,3)
    observationCount: 8,
    observationProgressText: 'Observation 0 / 8',
    fastForward: false,
    type: 'sphere',

    // Step display only
    step1: () => {},
    step2: () => {},
    step3: () => {},
    step4: () => {},
    stopAutoPlay: () => stopAutoPlay(),
    nextAutoStep: () => requestNextAutoStep(),
};

let scene, camera, renderer, controls;
let marchingCubes, marchingWireframe, pointCloud, frameVoxelCloud, globalVoxelCloud;
let gaussianClouds = [];
let gridHelper = null;
let cameraHelpers = [];
let scanLines = []; // animated rays during step 1
let voxelTooltip = null;

// Auto-play state
const autoPlayState = { playing: false, paused: false };
let resumeResolve = null;
let stepAdvanceCredits = 0;
const workerResolvers = new Map(); // type → resolve, for async worker communication

// What the current step/play WANTS to show — independent of user toggle permissions.
// gaussians: -1 = hide all, 'all' = show all sectors, 0–7 = show specific sector
const stepWants = { gaussians: 'all', points: true, frameVoxels: false, globalVoxels: false, mesh: false, camera: true };
const visibilityControllers = {};
let observationProgressCtrl = null;

let worker;
let workerNeedsInit = true;
let currentStep = 0;
let distancesBuffer = null;
let weightsBuffer = null;
let pendingPoints = null;
let activeDepthMap = null;
let activeCameraIndex = 0;
let activeObservationIndex = 0;
let frameDistancesBuffer = null;
let globalDistancesBuffer = null;
let frameWeightsBuffer = null;
let globalWeightsBuffer = null;

const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2(2, 2);
const pointerScreen = { x: 0, y: 0, active: false };
let hoveredVoxel = null;

const POINT_REVEAL_DURATION_MS = 1100;
const VOXEL_REVEAL_DURATION_MS = 850;
const AUTO_PLAY_STEP_HOLD_MS = 5000;
const pointRevealState = {
    active: false,
    startTime: 0,
    durationMs: POINT_REVEAL_DURATION_MS,
    totalPoints: 0,
    currentCount: 0,
    prevCount: 0,
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
const stepProgressState = {
    activeStep: 0,
    activeProgress: 0,
    nextStep: 0,
    completedStep: 0,
};


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

        // Sector assignment — shared with the worker's angular camera partitioning.
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

function syncPointCloudToGaussians() {
    renderRawPoints(getGaussianRawPoints());
    pendingPoints = null;
    applyVis();
}

function renderGaussianSectorPoints(cameraIndex) {
    const rawPoints = gaussianClouds[cameraIndex]?.rawPoints ?? new Float32Array(0);
    renderRawPoints(rawPoints);
    pendingPoints = null;
    applyVis();
}

function regenerateGaussianScene() {
    gaussianClouds.forEach(m => scene.remove(m));
    gaussianClouds = createGaussianCloud(params.type);
    gaussianClouds.forEach(m => scene.add(m));
    highlightCamera(activeCameraIndex);
}

function renderRawPoints(rawPoints) {
    const pointCount = rawPoints.length / 6;
    const positions = new Float32Array(pointCount * 3);

    for (let i = 0; i < pointCount; i++) {
        positions[i * 3] = rawPoints[i * 6];
        positions[i * 3 + 1] = rawPoints[i * 6 + 1];
        positions[i * 3 + 2] = rawPoints[i * 6 + 2];
    }

    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointCloud.geometry.setDrawRange(0, pointCount);
    pointCloud.material.opacity = 0.8;
    pointCloud.material.size = 0.02;
    pointRevealState.active = false;
    pointRevealState.totalPoints = pointCount;
    pointRevealState.currentCount = pointCount;
    pointRevealState.prevCount = 0;
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
    camera.position.set(2, 2, 2);

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
    highlightCamera(activeCameraIndex);

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
    syncPointCloudToGaussians();

    // Step 2: Voxelization
    const voxelMaterial = new THREE.PointsMaterial({
        color: 0xebcb8b, // nord13 (Yellow)
        size: 0.03,
        transparent: true,
        opacity: 0.4
    });
    frameVoxelCloud = new THREE.Points(new THREE.BufferGeometry(), voxelMaterial.clone());
    frameVoxelCloud.visible = false;
    scene.add(frameVoxelCloud);

    globalVoxelCloud = new THREE.Points(new THREE.BufferGeometry(), voxelMaterial.clone());
    globalVoxelCloud.visible = false;
    scene.add(globalVoxelCloud);

    voxelTooltip = document.createElement('div');
    voxelTooltip.className = 'voxel-tooltip hidden';
    document.body.appendChild(voxelTooltip);

    setupGUI();
    initWorker();

    // Modal Event Listeners
    document.getElementById('close-modal').addEventListener('click', () => toggleModal(false));
    document.getElementById('lang-toggle').addEventListener('click', () => toggleLanguage());
    document.getElementById('modal-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'modal-overlay') toggleModal(false);
    });

    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerleave', onPointerLeave);
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
        frameVoxelCloud.geometry.setDrawRange(0, voxelRevealState.totalPoints);
        voxelRevealState.currentCount = voxelRevealState.totalPoints;
    }
    frameVoxelCloud.material.opacity = params.showFrameVoxels ? 0.4 : frameVoxelCloud.material.opacity;
    frameVoxelCloud.material.size = 0.03;
}

function stopVoxelIntegration({ preserveColors = false } = {}) {
    voxelIntegrationState.active = false;
    voxelIntegrationState.progress = 0;
    voxelIntegrationState.pulsePhase = 0;
    if (!preserveColors) {
        globalVoxelCloud.material.color.setHex(0xebcb8b);
        globalVoxelCloud.material.opacity = params.showGlobalVoxels ? 0.4 : globalVoxelCloud.material.opacity;
        globalVoxelCloud.material.size = 0.03;
    }
}

/**
 * Color each active voxel by its TSDF scalar value using a diverging colormap:
 *   blue  (t = -1) → white (t = 0, zero-crossing / surface) → red (t = +1)
 *   near-black = unobserved (d ≈ 1.0)
 * PointsMaterial has no per-vertex alpha, so brightness encodes importance.
 */
function updateVoxelColors(targetCloud, distances) {
    const posAttr = targetCloud.geometry.attributes.position;
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

    targetCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colorData, 3));
    targetCloud.material.vertexColors = true;
    targetCloud.material.color.set(0xffffff);
    targetCloud.material.opacity = 0.80;
    targetCloud.material.size = 0.042;
    targetCloud.material.needsUpdate = true;
}

function updateVoxelIntegration(progress) {
    voxelIntegrationState.active = progress < 1;
    voxelIntegrationState.progress = progress;
    globalVoxelCloud.material.opacity = 0.22 + progress * 0.28;
    globalVoxelCloud.material.size = 0.03 + progress * 0.012;
    globalVoxelCloud.material.color.setRGB(
        0.92 + progress * 0.08,
        0.80 + progress * 0.14,
        0.30 + progress * 0.18,
    );
}

function setupGUI() {
    const gui = new GUI();
    
    gui.add(params, 'showKnowledge').name('💡 Knowledge Base');
    gui.add(params, 'cameraZoom', 0.8, 5, 0.1).name('🔍 Zoom').onChange(v => {
        const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
        offset.setLength(v);
        camera.position.copy(controls.target).add(offset);
        controls.update();
    });
    gui.add(params, 'observationCount', 1, 800, 1).name('Observation Count').onChange(() => {
        updatePlaybackInfo(autoPlayState.playing ? activeObservationIndex : 0, Math.max(1, Math.floor(params.observationCount)));
    });
    gui.add(params, 'fastForward').name('Fast Forward');
    let playPauseCtrl;
    params.togglePlay = () => {
        if (!autoPlayState.playing) startAutoPlay();
        else togglePause();
    };
    playPauseCtrl = gui.add(params, 'togglePlay').name('▶ Auto Play');
    window._playPauseCtrl = playPauseCtrl;
    gui.add(params, 'stopAutoPlay').name('■ Stop');
    gui.add(params, 'nextAutoStep').name('⏭ Next Step');

    // Shared voxel grid resolution — must be identical for TSDF volume and MarchingCubes
    // because MC reads the TSDF distances array directly (marchingCubes.field.set(distances)).
    params.voxelSizeDisplay = `${(4.0 / params.resolution).toFixed(4)}`;
    gui.add(params, 'resolution', 32, 96, 16).name('Grid Resolution').onChange(() => {
        workerNeedsInit = true;
        globalDistancesBuffer = null;
        resetManualProgress();
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
        workerNeedsInit = true;
        globalDistancesBuffer = null;
        resetManualProgress();
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(v);
        gaussianClouds.forEach(m => scene.add(m));
        syncPointCloudToGaussians();
        applyVis();
    });
    gsplatFolder.add(params, 'gaussianCount', 100, 1000, 50).name('Gaussian Count').onChange(() => {
        workerNeedsInit = true;
        globalDistancesBuffer = null;
        resetManualProgress();
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(params.type);
        gaussianClouds.forEach(m => scene.add(m));
        syncPointCloudToGaussians();
        applyVis();
    });
    gsplatFolder.add(params, 'gsplatNoise', 0.0, 1.0, 0.05).name('Splat Noise').onChange(() => {
        workerNeedsInit = true;
        globalDistancesBuffer = null;
        resetManualProgress();
        gaussianClouds.forEach(m => scene.remove(m));
        gaussianClouds = createGaussianCloud(params.type);
        gaussianClouds.forEach(m => scene.add(m));
        syncPointCloudToGaussians();
        applyVis();
    });

    // --- TSDF Volume (Generate + Fuse params) ---
    const tsdfFolder = gui.addFolder('TSDF Volume (Steps 2–3)');
    tsdfFolder.add(params, 'muVoxels', 1, 10, 0.5).name('Truncation μ');
    tsdfFolder.add(params, 'observationWeight', 0.1, 2.0, 0.1).name('Observation Weight');
    tsdfFolder.add(params, 'maxTSDFWeight', 1, 8, 1).name('Max TSDF Weight');
    tsdfFolder.add(params, 'mcWeightThreshold', 1, 8, 1).name('MC Weight Threshold');

    const steps = gui.addFolder('Steps');
    observationProgressCtrl = steps.add(params, 'observationProgressText').name('Observation').disable();
    const stepCtrls = [
        steps.add(params, 'step1').name('1. Acquire Depth'),
        steps.add(params, 'step2').name('2. Generate TSDF'),
        steps.add(params, 'step3').name('3. TSDF Fuse'),
        steps.add(params, 'step4').name('4. Marching Cubes'),
    ];
    stepCtrls.forEach(ctrl => {
        ctrl.domElement.classList.add('step-controller');
        ctrl.domElement.style.setProperty('--step-progress', '0%');
        ctrl.disable();
    });
    window._stepCtrls = stepCtrls;
    updateStepIndicators({ activeStep: 0, activeProgress: 0, nextStep: 1, completedStep: 0 });

    const view = gui.addFolder('Visibility');
    view.add(params, 'showGrids').name('Show Grids')
        .onChange(v => { gridHelper.visible = v; });
    const ctrlGaussians = view.add(params, 'showGaussians').name('Show Gaussians')
        .onChange(() => applyVis());
    const ctrlCamera = view.add(params, 'showCamera').name('Show Camera')
        .onChange(() => applyVis());
    const ctrlPoints = view.add(params, 'showPoints').name('Show Points')
        .onChange(() => applyVis());
    const ctrlFrameVoxels = view.add(params, 'showFrameVoxels').name('Show Frame Voxels')
        .onChange(() => applyVis());
    const ctrlGlobalVoxels = view.add(params, 'showGlobalVoxels').name('Show Global Voxels')
        .onChange(() => applyVis());
    const ctrlMesh = view.add(params, 'showMesh').name('Show Mesh')
        .onChange(() => applyVis());
    visibilityControllers.showGaussians = ctrlGaussians;
    visibilityControllers.showCamera = ctrlCamera;
    visibilityControllers.showPoints = ctrlPoints;
    visibilityControllers.showFrameVoxels = ctrlFrameVoxels;
    visibilityControllers.showGlobalVoxels = ctrlGlobalVoxels;
    visibilityControllers.showMesh = ctrlMesh;


    steps.open();
}

function initWorker() {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    sendInitToWorker();

    worker.onmessage = (e) => {
        const { type, data } = e.data;
        switch (type) {
            case 'ready':
                workerNeedsInit = false;
                highlightCamera(activeCameraIndex);
                activeObservationIndex = 0;
                updatePlaybackInfo(0, Math.max(1, Math.floor(params.observationCount)));
                updateStatus('System Ready. Use Auto Play to start the reconstruction pipeline.');
                break;

            case 'depth_acquired': {
                const { cameraIndex, depthMap, points } = data;
                activeCameraIndex = cameraIndex;
                activeDepthMap = depthMap;
                pendingPoints = points;
                showSingleDepthMap(cameraIndex, depthMap);
                clearScanRays();
                createScanRays(cameraIndex);
                highlightCamera(cameraIndex);
                markStepComplete(1, { nextStep: 2 });
                resolveWorker('depth_acquired', data);
                updateStatus(`Step 1 Complete: Camera ${cameraIndex + 1} depth captured. Run Step 2 to build the frame TSDF.`);
                break;
            }

            case 'gaussians_updated':
                resolveWorker('gaussians_updated', data);
                break;

            case 'frame_tsdf_generated':
                activeCameraIndex = data.cameraIndex;
                frameDistancesBuffer = data.distances;
                frameWeightsBuffer = data.weights;
                distancesBuffer = data.distances;
                weightsBuffer = data.weights;
                renderVoxels(frameVoxelCloud, data.voxelCenters);
                updateVoxelColors(frameVoxelCloud, data.distances);
                markStepComplete(2, { nextStep: 3 });
                resolveWorker('frame_tsdf_generated', data);
                console.debug('frame_tsdf_generated', data.debug);
                updateStatus(
                    `Step 2 Complete: Frame TSDF generated for camera ${data.cameraIndex + 1}. `
                    + `[debug: ${formatVolumeDebug(data.debug)}]`,
                );
                break;

            case 'tsdf_fused':
                activeCameraIndex = data.cameraIndex;
                globalDistancesBuffer = data.distances;
                globalWeightsBuffer = data.weights;
                distancesBuffer = data.distances;
                weightsBuffer = data.weights;
                renderVoxels(globalVoxelCloud, data.voxelCenters, { animate: false });
                updateVoxelColors(globalVoxelCloud, data.distances);
                stopVoxelIntegration({ preserveColors: true });
                markStepComplete(3, { nextStep: 4 });
                resolveWorker('tsdf_fused', data);
                console.debug('tsdf_fused', data.debug);
                updateStatus(
                    data.skipped
                        ? `Step 3: Camera ${data.cameraIndex + 1} initialized the global TSDF. [debug: ${formatVolumeDebug(data.debug)}]`
                        : `Step 3 Complete: Camera ${data.cameraIndex + 1} fused into the global TSDF. [debug: ${formatVolumeDebug(data.debug)}]`,
                );
                break;

            case 'mesh_ready': {
                if (data.distances) {
                    distancesBuffer = data.distances;
                    renderMesh(data.distances);
                }
                markStepComplete(4, { nextStep: 0 });
                resolveWorker('mesh_ready', data);
                updateStatus(`Step 4 Complete: Mesh extracted from the ${data.source ?? 'latest'} TSDF.`);
                break;
            }

            case 'reset_done': {
                resetManualProgress();
                globalDistancesBuffer = null;
                globalWeightsBuffer = null;
                distancesBuffer = null;
                weightsBuffer = null;
                workerNeedsInit = true;
                clearPointCloud();
                stopPointReveal({ showAll: false });
                stopVoxelReveal({ showAll: false });
                stopVoxelIntegration();
                updateStepIndicators({ activeStep: 0, activeProgress: 0, nextStep: 1, completedStep: 0 });
                clearScanRays();
                highlightCamera(activeCameraIndex);
                gaussianClouds.forEach(m => scene.remove(m));
                gaussianClouds = createGaussianCloud(params.type);
                gaussianClouds.forEach(m => scene.add(m));
                syncPointCloudToGaussians();
                setVis('gaussians', true);
                setVis('camera', true);
                setVis('points', true);
                setVis('frameVoxels', false);
                setVis('globalVoxels', false);
                setVis('mesh', false);
                cameraHelpers.forEach(h => {
                    if (h.depthCtx) {
                        h.depthCtx.fillStyle = '#0d1117';
                        h.depthCtx.fillRect(0, 0, DEPTH_RES, DEPTH_RES);
                        h.depthTexture.needsUpdate = true;
                    }
                });
                updateStatus('Reset Complete. Use Auto Play to start the reconstruction pipeline.');
                // Abort any in-progress auto-play
                autoPlayState.playing = false;
                autoPlayState.paused = false;
                stepAdvanceCredits = 0;
                activeObservationIndex = 0;
                if (resumeResolve) { resumeResolve(); resumeResolve = null; }
                updatePlaybackInfo(0, Math.max(1, Math.floor(params.observationCount)));
                updatePlayButton();
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
    if (pointCloud) {
        const pointCount = pointCloud.geometry.attributes.position?.count ?? 0;
        pointCloud.visible = params.showPoints && stepWants.points && pointCount > 0;
    }
    if (frameVoxelCloud) frameVoxelCloud.visible = params.showFrameVoxels && stepWants.frameVoxels;
    if (globalVoxelCloud) globalVoxelCloud.visible = params.showGlobalVoxels && stepWants.globalVoxels;
    if (marchingCubes) {
        marchingCubes.visible = params.showMesh && stepWants.mesh;
    }
    if (marchingWireframe) {
        marchingWireframe.visible = params.showMesh && stepWants.mesh;
    }
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

function setVisibilityToggle(key, value) {
    params[key] = value;
    visibilityControllers[key]?.updateDisplay();
}

function setManualVisibilityState({
    showGaussians = params.showGaussians,
    showCamera = params.showCamera,
    showPoints = params.showPoints,
    showFrameVoxels = params.showFrameVoxels,
    showGlobalVoxels = params.showGlobalVoxels,
    showMesh = params.showMesh,
} = {}) {
    stepWants.gaussians = 'all';
    stepWants.camera = true;
    stepWants.points = true;
    stepWants.frameVoxels = true;
    stepWants.globalVoxels = true;
    stepWants.mesh = true;

    setVisibilityToggle('showGaussians', showGaussians);
    setVisibilityToggle('showCamera', showCamera);
    setVisibilityToggle('showPoints', showPoints);
    setVisibilityToggle('showFrameVoxels', showFrameVoxels);
    setVisibilityToggle('showGlobalVoxels', showGlobalVoxels);
    setVisibilityToggle('showMesh', showMesh);
    applyVis();
}

function formatVolumeDebug(debug) {
    if (!debug) return '';
    const parts = [];

    if (debug.frame) {
        parts.push(
            `frame vox=${debug.frame.observedVoxels}`,
            `frame wSum=${debug.frame.weightSum.toFixed(1)}`,
            `frame wMax=${debug.frame.maxWeight.toFixed(1)}`,
        );
    } else {
        parts.push(
            `vox=${debug.observedVoxels}`,
            `wSum=${debug.weightSum.toFixed(1)}`,
            `wMax=${debug.maxWeight.toFixed(1)}`,
        );
    }

    if (debug.global) {
        parts.push(
            `global vox=${debug.global.observedVoxels}`,
            `global wSum=${debug.global.weightSum.toFixed(1)}`,
            `changed=${debug.changedVoxels}`,
        );
    }

    return parts.join(', ');
}

function resolveWorker(type, data) {
    if (!workerResolvers.has(type)) return;
    workerResolvers.get(type)(data);
    workerResolvers.delete(type);
}

function highlightCamera(cameraIndex) {
    cameraHelpers.forEach((h, i) => {
        if (h.children[0]) h.children[0].material.color.setHex(i === cameraIndex ? 0xebcb8b : 0x88c0d0);
    });
}

function showSingleDepthMap(cameraIndex, depthMap) {
    cameraHelpers.forEach((icon, idx) => {
        if (!icon?.depthCtx) return;
        const ctx = icon.depthCtx;
        if (idx !== cameraIndex) {
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, DEPTH_RES, DEPTH_RES);
            icon.depthTexture.needsUpdate = true;
            return;
        }

        const imgData = ctx.createImageData(DEPTH_RES, DEPTH_RES);
        let minD = Infinity;
        let maxD = -Infinity;
        for (let i = 0; i < depthMap.length; i++) {
            if (depthMap[i] >= 0) {
                minD = Math.min(minD, depthMap[i]);
                maxD = Math.max(maxD, depthMap[i]);
            }
        }
        for (let i = 0; i < depthMap.length; i++) {
            const d = depthMap[i];
            const idx4 = i * 4;
            if (d < 0) {
                imgData.data[idx4] = 13;
                imgData.data[idx4 + 1] = 17;
                imgData.data[idx4 + 2] = 23;
                imgData.data[idx4 + 3] = 200;
            } else {
                const t = maxD > minD ? 1 - (d - minD) / (maxD - minD) : 0.5;
                const v = Math.floor(t * 255);
                imgData.data[idx4] = v;
                imgData.data[idx4 + 1] = v;
                imgData.data[idx4 + 2] = v;
                imgData.data[idx4 + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);
        icon.depthTexture.needsUpdate = true;
    });
}

function resetManualProgress() {
    currentStep = 0;
    pendingPoints = null;
    activeDepthMap = null;
    frameDistancesBuffer = null;
    frameWeightsBuffer = null;
    distancesBuffer = globalDistancesBuffer;
    weightsBuffer = globalWeightsBuffer;
    activeCameraIndex = 0;
    hoveredVoxel = null;
    hideVoxelTooltip();
}

function clearPointCloud() {
    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    pointCloud.geometry.setDrawRange(0, 0);
    stopPointReveal({ showAll: false });
    pendingPoints = null;
}

function hideVoxelTooltip() {
    voxelTooltip?.classList.add('hidden');
}

function onPointerMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointerScreen.x = event.clientX;
    pointerScreen.y = event.clientY;
    pointerNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    pointerScreen.active = true;
}

function onPointerLeave() {
    pointerScreen.active = false;
    pointerNdc.set(2, 2);
    hoveredVoxel = null;
    hideVoxelTooltip();
}

function updateVoxelHover() {
    const candidates = [];
    if (frameVoxelCloud.visible && frameDistancesBuffer && frameWeightsBuffer) {
        candidates.push({
            kind: 'frame',
            cloud: frameVoxelCloud,
            distances: frameDistancesBuffer,
            weights: frameWeightsBuffer,
        });
    }
    if (globalVoxelCloud.visible && globalDistancesBuffer && globalWeightsBuffer) {
        candidates.push({
            kind: 'global',
            cloud: globalVoxelCloud,
            distances: globalDistancesBuffer,
            weights: globalWeightsBuffer,
        });
    }

    if (!pointerScreen.active || candidates.length === 0) {
        hoveredVoxel = null;
        hideVoxelTooltip();
        return;
    }

    raycaster.setFromCamera(pointerNdc, camera);
    let bestHit = null;
    for (const candidate of candidates) {
        raycaster.params.Points.threshold = Math.max(0.12, candidate.cloud.material.size * 3.5);
        const hit = raycaster.intersectObject(candidate.cloud, false)[0];
        if (!hit || hit.index === undefined) continue;
        if (!bestHit || hit.distance < bestHit.hit.distance) {
            bestHit = { candidate, hit };
        }
    }

    if (!bestHit) {
        hoveredVoxel = null;
        hideVoxelTooltip();
        return;
    }

    const { candidate, hit } = bestHit;
    const posAttr = candidate.cloud.geometry.attributes.position;
    const wx = posAttr.getX(hit.index);
    const wy = posAttr.getY(hit.index);
    const wz = posAttr.getZ(hit.index);
    const idx = getVoxelIndexFromWorld(wx, wy, wz);
    if (idx < 0 || idx >= candidate.distances.length || idx >= candidate.weights.length) {
        hoveredVoxel = null;
        hideVoxelTooltip();
        return;
    }

    hoveredVoxel = {
        kind: candidate.kind,
        world: [wx, wy, wz],
        index: idx,
        distance: candidate.distances[idx],
        weight: candidate.weights[idx],
        mcThreshold: params.mcWeightThreshold,
        weightPassesThreshold: candidate.weights[idx] >= params.mcWeightThreshold,
        mcCellEligible: hasEligibleMarchingCubesCell(idx, candidate.weights, params.resolution, params.mcWeightThreshold),
    };
    renderVoxelTooltip();
}

function getVoxelIndexFromWorld(wx, wy, wz) {
    const extent = 2.0;
    const size = params.resolution;
    const voxelSize = (extent * 2) / size;
    const ix = Math.floor((wx + extent) / voxelSize);
    const iy = Math.floor((wy + extent) / voxelSize);
    const iz = Math.floor((wz + extent) / voxelSize);
    if (ix < 0 || ix >= size || iy < 0 || iy >= size || iz < 0 || iz >= size) return -1;
    return ix + iy * size + iz * size * size;
}

function getVoxelCoordsFromIndex(idx, size) {
    const sizeSq = size * size;
    const iz = Math.floor(idx / sizeSq);
    const iy = Math.floor((idx % sizeSq) / size);
    const ix = idx % size;
    return { ix, iy, iz };
}

function hasEligibleMarchingCubesCell(idx, weights, size, threshold) {
    if (threshold <= 0) return true;

    const { ix, iy, iz } = getVoxelCoordsFromIndex(idx, size);
    for (let dz = -1; dz <= 0; dz++) {
        for (let dy = -1; dy <= 0; dy++) {
            for (let dx = -1; dx <= 0; dx++) {
                const x = ix + dx;
                const y = iy + dy;
                const z = iz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= size - 1 || y >= size - 1 || z >= size - 1) continue;
                if (cellPassesWeightThreshold(x, y, z, weights, size, threshold)) return true;
            }
        }
    }
    return false;
}

function cellPassesWeightThreshold(x, y, z, weights, size, threshold) {
    const sizeSq = size * size;
    const i000 = x + y * size + z * sizeSq;
    const i100 = i000 + 1;
    const i010 = i000 + size;
    const i110 = i010 + 1;
    const i001 = i000 + sizeSq;
    const i101 = i001 + 1;
    const i011 = i001 + size;
    const i111 = i011 + 1;
    return weights[i000] >= threshold
        && weights[i100] >= threshold
        && weights[i010] >= threshold
        && weights[i110] >= threshold
        && weights[i001] >= threshold
        && weights[i101] >= threshold
        && weights[i011] >= threshold
        && weights[i111] >= threshold;
}

function renderVoxelTooltip() {
    if (!hoveredVoxel || !voxelTooltip) return;
    const mu = getMu();
    voxelTooltip.innerHTML = [
        `kind: ${hoveredVoxel.kind}`,
        `idx: ${hoveredVoxel.index}`,
        `distance: ${hoveredVoxel.distance.toFixed(4)}`,
        `weight: ${hoveredVoxel.weight.toFixed(2)}`,
        `mc threshold: ${hoveredVoxel.mcThreshold.toFixed(2)}`,
        `voxel passes: ${hoveredVoxel.weightPassesThreshold ? 'yes' : 'no'}`,
        `mc cell eligible: ${hoveredVoxel.mcCellEligible ? 'yes' : 'no'}`,
        `distance/mu: ${(hoveredVoxel.distance / mu).toFixed(3)}`,
    ].join('<br>');
    voxelTooltip.style.left = `${pointerScreen.x + 14}px`;
    voxelTooltip.style.top = `${pointerScreen.y + 14}px`;
    voxelTooltip.classList.remove('hidden');
}

function highlightStep(step) {
    updateStepIndicators({ activeStep: step, nextStep: step > 0 ? Math.min(step + 1, window._stepCtrls.length) : 0 });
}

function updateStepIndicators({ activeStep = stepProgressState.activeStep, activeProgress = stepProgressState.activeProgress, nextStep = stepProgressState.nextStep, completedStep = stepProgressState.completedStep } = {}) {
    stepProgressState.activeStep = activeStep;
    stepProgressState.activeProgress = activeProgress;
    stepProgressState.nextStep = nextStep;
    stepProgressState.completedStep = completedStep;

    window._stepCtrls.forEach((c, i) => {
        const stepNum = i + 1;
        c.domElement.classList.toggle('active-step', stepNum === activeStep);
        c.domElement.classList.toggle('next-step', stepNum === nextStep);
        c.domElement.style.setProperty('--step-progress', `${stepNum === activeStep ? Math.max(0, Math.min(1, activeProgress)) * 100 : 0}%`);
    });
}

function setStepProgress(step, progress, { nextStep = step < window._stepCtrls.length ? step + 1 : 0, completedStep = Math.max(stepProgressState.completedStep, step - 1) } = {}) {
    updateStepIndicators({
        activeStep: step,
        activeProgress: progress,
        nextStep,
        completedStep,
    });
}

function markStepComplete(step, { nextStep = step < window._stepCtrls.length ? step + 1 : 0 } = {}) {
    updateStepIndicators({
        activeStep: step,
        activeProgress: 1,
        nextStep,
        completedStep: Math.max(stepProgressState.completedStep, step),
    });
}

async function waitWithStepProgress(step, baseDurationMs, startTime = performance.now(), { nextStep = step < window._stepCtrls.length ? step + 1 : 0, completedStep = Math.max(stepProgressState.completedStep, step) } = {}) {
    let adjustedStart = startTime;
    while (autoPlayState.playing) {
        if (autoPlayState.paused) {
            if (stepAdvanceCredits > 0) {
                stepAdvanceCredits -= 1;
            } else {
                const pauseStarted = performance.now();
                await new Promise(r => { resumeResolve = r; });
                adjustedStart += performance.now() - pauseStarted;
                continue;
            }
        }

        const durationMs = params.fastForward ? 0 : baseDurationMs;
        if (durationMs <= 0) {
            markStepComplete(step, { nextStep });
            return;
        }

        const progress = Math.min(1, (performance.now() - adjustedStart) / durationMs);
        setStepProgress(step, progress, { nextStep, completedStep: Math.max(stepProgressState.completedStep, step - 1) });
        if (progress >= 1) {
            markStepComplete(step, { nextStep });
            return;
        }
        await new Promise(r => setTimeout(r, 50));
    }
}

function triggerStep(step) {
    if (step === 1) {
        activeCameraIndex = 0;
        currentStep = 1;
        setStepProgress(1, 0, { nextStep: 2, completedStep: 0 });
        clearScanRays();
        setManualVisibilityState({
            showGaussians: true,
            showCamera: true,
            showPoints: true,
            showFrameVoxels: false,
            showGlobalVoxels: false,
            showMesh: false,
        });
        if (workerNeedsInit) sendInitToWorker();
        worker.postMessage({ type: 'acquire_depth', data: { cameraIndex: activeCameraIndex } });

    } else if (step === 2) {
        if (currentStep < 1) return updateStatus('Error: Complete Step 1 first.');
        if (!pendingPoints) return updateStatus('Error: Step 1 not finished yet.');
        currentStep = 2;
        setStepProgress(2, 0, { nextStep: 3, completedStep: 1 });
        setManualVisibilityState({
            showGaussians: false,
            showCamera: true,
            showPoints: true,
            showFrameVoxels: true,
            showGlobalVoxels: false,
            showMesh: false,
        });
        renderPoints(pendingPoints);
        updateStatus('Step 2: Generating the frame TSDF...');
        worker.postMessage({
            type: 'generate_tsdf',
            data: {
                cameraIndex: activeCameraIndex,
                size: params.resolution,
                extent: 2.0,
                mu: getMu(),
                observationWeight: params.observationWeight,
                maxTSDFWeight: params.maxTSDFWeight,
            },
        });

    } else if (step === 3) {
        if (currentStep < 2) return updateStatus('Error: Complete Step 2 first.');
        currentStep = 3;
        setStepProgress(3, 0, { nextStep: 4, completedStep: 2 });
        updateStatus('Step 3: Fusing frame TSDF into the global TSDF...');
        setManualVisibilityState({
            showGaussians: false,
            showCamera: true,
            showPoints: true,
            showFrameVoxels: false,
            showGlobalVoxels: true,
            showMesh: false,
        });
        voxelIntegrationState.active = true;
        updateVoxelIntegration(0.5);
        worker.postMessage({
            type: 'fuse_tsdf',
            data: {
                cameraIndex: activeCameraIndex,
                size: params.resolution,
                extent: 2.0,
                mu: getMu(),
                observationWeight: params.observationWeight,
                maxTSDFWeight: params.maxTSDFWeight,
            },
        });

    } else if (step === 4) {
        if (currentStep < 3) return updateStatus('Error: Complete Step 3 first.');
        currentStep = 4;
        setStepProgress(4, 0, { nextStep: 0, completedStep: 3 });
        stopVoxelReveal();
        updateStatus('Step 4: Running Marching Cubes on the latest TSDF...');
        setManualVisibilityState({
            showGaussians: false,
            showCamera: false,
            showPoints: false,
            showFrameVoxels: false,
            showGlobalVoxels: false,
            showMesh: true,
        });
        worker.postMessage({ type: 'extract_mesh', data: { weightThreshold: params.mcWeightThreshold } });
    }
}

function renderPoints(data) {
    const newCount = data.length / 7;
    const positions = new Float32Array(newCount * 3);
    for (let i = 0; i < newCount; i++) {
        positions[i * 3]     = data[i * 7];
        positions[i * 3 + 1] = data[i * 7 + 1];
        positions[i * 3 + 2] = data[i * 7 + 2];
    }

    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointCloud.geometry.setDrawRange(0, 0);
    pointCloud.material.opacity = 0.18;
    pointCloud.material.size = 0.028;

    pointRevealState.active = true;
    pointRevealState.startTime = performance.now();
    pointRevealState.totalPoints = newCount;
    pointRevealState.prevCount = 0;
    pointRevealState.currentCount = 0;
}

function renderVoxels(targetCloud, data, { animate = true } = {}) {
    targetCloud.geometry.deleteAttribute('color');
    targetCloud.material.vertexColors = false;
    targetCloud.material.needsUpdate = true;
    targetCloud.geometry.setAttribute('position', new THREE.BufferAttribute(data, 3));
    targetCloud.geometry.setDrawRange(0, animate ? 0 : data.length / 3);
    targetCloud.material.opacity = 0.08;
    targetCloud.material.size = 0.055;
    targetCloud.material.color.setHex(0xebcb8b);

    if (targetCloud === frameVoxelCloud) {
        voxelRevealState.active = animate;
        voxelRevealState.startTime = performance.now();
        voxelRevealState.totalPoints = data.length / 3;
        voxelRevealState.currentCount = animate ? 0 : data.length / 3;
        stopVoxelIntegration();
    }
}

function showActiveVoxelPreview() {
    frameVoxelCloud.geometry.deleteAttribute('color');
    frameVoxelCloud.material.vertexColors = false;
    frameVoxelCloud.material.needsUpdate = true;
    frameVoxelCloud.material.color.setHex(0xebcb8b);
    frameVoxelCloud.material.opacity = 0.4;
    frameVoxelCloud.material.size = 0.03;
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
    if (stepAdvanceCredits > 0) {
        stepAdvanceCredits -= 1;
        return Promise.resolve();
    }
    return new Promise(r => { resumeResolve = r; });
}

/** Wait until at least `ms` milliseconds have elapsed since `startTime`. */
function waitAtLeast(ms, startTime = performance.now()) {
    if (ms <= 0) return Promise.resolve();
    const remaining = ms - (performance.now() - startTime);
    return new Promise(r => setTimeout(r, Math.max(0, remaining)));
}

function getAutoPlayHoldMs() {
    return params.fastForward ? 0 : AUTO_PLAY_STEP_HOLD_MS;
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
    if (!autoPlayState.paused) stepAdvanceCredits = 0;
    updatePlayButton();
}

function stopAutoPlay() {
    resetDemo();
}

function requestNextAutoStep() {
    if (!autoPlayState.playing || !autoPlayState.paused) return;
    stepAdvanceCredits += 1;
    if (resumeResolve) {
        resumeResolve();
        resumeResolve = null;
    }
}

async function startAutoPlay() {
    if (autoPlayState.playing) return;
    autoPlayState.playing = true;
    autoPlayState.paused = false;
    updatePlayButton();

    // Reset visuals
    setManualVisibilityState({
        showGaussians: false,
        showCamera: true,
        showPoints: true,
        showFrameVoxels: false,
        showGlobalVoxels: false,
        showMesh: false,
    });
    clearScanRays();
    resetManualProgress();
    updateStepIndicators({ activeStep: 0, activeProgress: 0, nextStep: 1, completedStep: 0 });

    updateStatus('Auto Play: Initialising KinectFusion-style pipeline...');
    sendInitToWorker();

    const totalObservations = Math.max(1, Math.floor(params.observationCount));
    for (let observationIndex = 0; observationIndex < totalObservations; observationIndex++) {
        if (!autoPlayState.playing) break;
        const c = observationIndex % CAM_COUNT_MAIN;
        activeCameraIndex = c;
        activeObservationIndex = observationIndex + 1;
        updatePlaybackInfo(activeObservationIndex, totalObservations);

        // ── Step 1: Acquire Depth ────────────────────────────────────────────
        setStepProgress(1, 0, { nextStep: 2, completedStep: 0 });
        updateStatus(`Observation ${observationIndex + 1}/${totalObservations} — Camera ${c + 1}/8 — Step 1: Acquiring depth...`);
        setManualVisibilityState({
            showGaussians: true,
            showCamera: true,
            showPoints: true,
            showFrameVoxels: false,
            showGlobalVoxels: false,
            showMesh: false,
        });
        regenerateGaussianScene();
        const waitForGaussiansUpdated = waitForWorkerMsg('gaussians_updated');
        worker.postMessage({
            type: 'update_gaussians',
            data: {
                type: params.type,
                noise: params.noise,
                gsplatNoise: params.gsplatNoise,
                gaussianPoints: getGaussianRawPoints(),
            },
        });
        await waitForGaussiansUpdated;
        await checkPause();
        addSector(c);
        renderGaussianSectorPoints(c);
        clearScanRays();
        highlightCamera(c);
        const t1 = performance.now();
        worker.postMessage({ type: 'acquire_depth', data: { cameraIndex: c } });
        await waitForWorkerMsg('depth_acquired');
        await checkPause();
        await waitWithStepProgress(1, AUTO_PLAY_STEP_HOLD_MS, t1, { nextStep: 2 });
        if (!autoPlayState.playing) break;

        // ── Step 2: Generate TSDF ────────────────────────────────────────────
        setStepProgress(2, 0, { nextStep: 3, completedStep: 1 });
        updateStatus(`Observation ${observationIndex + 1}/${totalObservations} — Camera ${c + 1}/8 — Step 2: Generating frame TSDF...`);
        setManualVisibilityState({
            showGaussians: false,
            showCamera: true,
            showPoints: true,
            showFrameVoxels: true,
            showGlobalVoxels: false,
            showMesh: false,
        });
        renderPoints(pendingPoints);
        worker.postMessage({
            type: 'generate_tsdf',
            data: {
                cameraIndex: c,
                size: params.resolution,
                extent: 2.0,
                mu: getMu(),
                observationWeight: params.observationWeight,
                maxTSDFWeight: params.maxTSDFWeight,
            },
        });
        await waitForWorkerMsg('frame_tsdf_generated');
        await checkPause();
        await waitWithStepProgress(2, AUTO_PLAY_STEP_HOLD_MS, performance.now(), { nextStep: 3 });
        if (!autoPlayState.playing) break;

        if (observationIndex === 0) {
            markStepComplete(3, { nextStep: 4 });
            updateStatus(`Observation 1/${totalObservations} — Camera 1/8 — Step 3 skipped: first frame initializes the global TSDF.`);
            await checkPause();
            await waitAtLeast(getAutoPlayHoldMs(), performance.now());
        } else {
            // ── Step 3: TSDF Fuse ────────────────────────────────────────────
            setStepProgress(3, 0, { nextStep: 4, completedStep: 2 });
            updateStatus(`Observation ${observationIndex + 1}/${totalObservations} — Camera ${c + 1}/8 — Step 3: Fusing frame TSDF...`);
            setManualVisibilityState({
                showGaussians: false,
                showCamera: true,
                showPoints: true,
                showFrameVoxels: false,
                showGlobalVoxels: true,
                showMesh: false,
            });
            voxelIntegrationState.active = true;
            updateVoxelIntegration(0.5);
            const t3 = performance.now();
            worker.postMessage({
                type: 'fuse_tsdf',
                data: {
                    cameraIndex: c,
                    size: params.resolution,
                    extent: 2.0,
                    mu: getMu(),
                    observationWeight: params.observationWeight,
                    maxTSDFWeight: params.maxTSDFWeight,
                },
            });
            await waitForWorkerMsg('tsdf_fused');
            await checkPause();
            await waitWithStepProgress(3, AUTO_PLAY_STEP_HOLD_MS, t3, { nextStep: 4 });
            if (!autoPlayState.playing) break;
        }

        if (observationIndex === 0) {
            worker.postMessage({
                type: 'fuse_tsdf',
                data: {
                    cameraIndex: c,
                    size: params.resolution,
                    extent: 2.0,
                    mu: getMu(),
                    observationWeight: params.observationWeight,
                    maxTSDFWeight: params.maxTSDFWeight,
                },
            });
            await waitForWorkerMsg('tsdf_fused');
            await checkPause();
        }

        // ── Step 4: Marching Cubes ───────────────────────────────────────────
        setStepProgress(4, 0, { nextStep: observationIndex < totalObservations - 1 ? 1 : 0, completedStep: 3 });
        updateStatus(`Observation ${observationIndex + 1}/${totalObservations} — Camera ${c + 1}/8 — Step 4: Running Marching Cubes...`);
        setManualVisibilityState({
            showGaussians: false,
            showCamera: false,
            showPoints: false,
            showFrameVoxels: false,
            showGlobalVoxels: false,
            showMesh: true,
        });
        const t4 = performance.now();
        worker.postMessage({ type: 'extract_mesh', data: { weightThreshold: params.mcWeightThreshold } });
        await waitForWorkerMsg('mesh_ready');
        await checkPause();
        await waitWithStepProgress(4, AUTO_PLAY_STEP_HOLD_MS, t4, { nextStep: observationIndex < totalObservations - 1 ? 1 : 0 });
        cameraHelpers.forEach(h => {
            if (h.children[0]) h.children[0].material.color.setHex(0x88c0d0);
        });
    }

    autoPlayState.playing = false;
    stepAdvanceCredits = 0;
    activeObservationIndex = 0;
    updatePlaybackInfo(0, Math.max(1, Math.floor(params.observationCount)));
    updatePlayButton();
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
            gsplatNoise: params.gsplatNoise,
            // Pass gaussian positions so the worker can synthesize depth maps
            // from the gsplat distribution before back-projecting them.
            gaussianPoints: getGaussianRawPoints(),
        },
    });
}

function updateStatus(msg) {
    document.getElementById('status-message').textContent = msg;
}

function updatePlaybackInfo(currentObservation, totalObservations) {
    params.observationProgressText = `Observation ${currentObservation} / ${totalObservations}`;
    observationProgressCtrl?.updateDisplay();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    hideVoxelTooltip();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    updateVoxelHover();

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
        }
    }

    if (frameVoxelCloud.visible && voxelRevealState.active) {
        const elapsed = performance.now() - voxelRevealState.startTime;
        const t = Math.min(1, elapsed / voxelRevealState.durationMs);
        const eased = easeOutCubic(t);
        const nextCount = Math.floor(voxelRevealState.totalPoints * eased);

        if (nextCount !== voxelRevealState.currentCount) {
            frameVoxelCloud.geometry.setDrawRange(0, nextCount);
            voxelRevealState.currentCount = nextCount;
        }

        frameVoxelCloud.material.opacity = 0.08 + eased * 0.32;
        frameVoxelCloud.material.size = 0.055 - eased * 0.025;

        if (t >= 1) {
            stopVoxelReveal();
        }
    }

    if (globalVoxelCloud.visible && voxelIntegrationState.active) {
        voxelIntegrationState.pulsePhase += 0.16;
        const pulse = 0.5 + 0.5 * Math.sin(voxelIntegrationState.pulsePhase);
        globalVoxelCloud.material.size = 0.036 + voxelIntegrationState.progress * 0.006 + pulse * 0.004;
        globalVoxelCloud.material.opacity = Math.max(globalVoxelCloud.material.opacity, 0.30 + voxelIntegrationState.progress * 0.12 + pulse * 0.05);
    }

    renderer.render(scene, camera);
}
