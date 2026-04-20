import { TSDFVolume } from './tsdf.js';

let volume = null;
let currentPoints = null;
let sceneType = 'sphere';
let noiseStd = 0.02;
let outlierRate = 0.05;
let observationWeight = 1.0;
let gaussianPoints = null; // Float32Array [x,y,z,nx,ny,nz] × N from main thread

/** Must match the constants in main.js */
const CAM_COUNT = 8;
const CAM_RADIUS = 2.5;
const CAM_HEIGHT = 1.0;
const CAM_FOV_DEG = 45;
export const DEPTH_RES = 48; // exported so main.js can import for canvas size

self.onmessage = (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            volume = new TSDFVolume(data);
            sceneType = data.type || 'sphere';
            noiseStd = data.noise || 0.02;
            outlierRate = data.outliers || 0.05;
            observationWeight = data.observationWeight ?? 1.0;
            gaussianPoints = data.gaussianPoints ?? null;
            currentPoints = null;
            self.postMessage({ type: 'ready' });
            break;

        case 'step1_render':
            renderDepthMapsProgressive();
            break;

        case 'step3_voxelize':
            if (volume && currentPoints) {
                if (data) {
                    // Rebuild volume if resolution changed so voxelSize/range are correct.
                    if (data.resolution !== undefined && data.resolution !== volume.size) {
                        volume = new TSDFVolume({
                            size: data.resolution,
                            extent: volume.extent,
                            mu: data.mu ?? volume.mu,
                            maxTSDFWeight: volume.maxTSDFWeight,
                        });
                    } else if (data.mu !== undefined) {
                        volume.mu = data.mu;
                        volume.range = Math.ceil(data.mu / volume.voxelSize);
                    }
                }
                self.postMessage({ type: 'voxels', data: getAffectedVoxels(volume, currentPoints) });
            }
            break;

        case 'step4_fuse':
            if (volume && currentPoints) {
                // Apply latest params so re-running step 4 after changing mu/weights
                // always reflects the current GUI settings.
                if (data) {
                    if (data.mu !== undefined) {
                        volume.mu = data.mu;
                        volume.range = Math.ceil(data.mu / volume.voxelSize);
                    }
                    if (data.observationWeight !== undefined) observationWeight = data.observationWeight;
                    if (data.maxTSDFWeight    !== undefined) volume.maxTSDFWeight = data.maxTSDFWeight;
                }
                // Reset the volume so each run starts fresh — otherwise repeated
                // runs accumulate weights and changing params has no visible effect.
                volume.distances.fill(1.0);
                volume.weights.fill(0.0);
                fuseProgressive(volume, currentPoints);
            }
            break;

        case 'step5_extract':
            if (volume) {
                self.postMessage({ type: 'extracted', data: volume.getGridData() });
            }
            break;

        case 'reset':
            if (volume) {
                volume.distances.fill(1.0);
                volume.weights.fill(0.0);
            }
            currentPoints = null;
            gaussianPoints = null;
            self.postMessage({ type: 'reset_done' });
            break;
    }
};

// ---------------------------------------------------------------------------
// Step 1: depth map rendering + back-projection
// ---------------------------------------------------------------------------

/**
 * Build { pos, fwd, right, up } axes for camera at ring index camIdx.
 * All vectors are plain arrays [x, y, z].
 */
function getCameraAxes(camIdx) {
    const angle = (camIdx / CAM_COUNT) * Math.PI * 2;
    const pos = [
        CAM_RADIUS * Math.cos(angle),
        CAM_HEIGHT,
        CAM_RADIUS * Math.sin(angle),
    ];
    const L = Math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2);
    const fwd = [-pos[0] / L, -pos[1] / L, -pos[2] / L]; // toward origin

    // right = normalize(fwd × worldUp), worldUp = (0,1,0)
    // fwd × (0,1,0) = (-fwd[2], 0, fwd[0])
    const rx = -fwd[2], rz = fwd[0];
    const rLen = Math.sqrt(rx * rx + rz * rz);
    const right = [rx / rLen, 0, rz / rLen];

    // up = right × fwd (always unit length given our geometry)
    const ux = right[1] * fwd[2] - right[2] * fwd[1];
    const uy = right[2] * fwd[0] - right[0] * fwd[2];
    const uz = right[0] * fwd[1] - right[1] * fwd[0];
    const uLen = Math.sqrt(ux * ux + uy * uy + uz * uz);

    return { pos, fwd, right, up: [ux / uLen, uy / uLen, uz / uLen] };
}

async function renderDepthMapsProgressive() {
    const halfFOV = (CAM_FOV_DEG * Math.PI / 180) / 2;
    const tanFOV = Math.tan(halfFOV);

    // Render depth maps camera-by-camera — purely for the visual animation.
    // The actual point cloud is sampled from the gaussian data below.
    for (let camIdx = 0; camIdx < CAM_COUNT; camIdx++) {
        const { pos, fwd, right, up } = getCameraAxes(camIdx);
        const depthMap = new Float32Array(DEPTH_RES * DEPTH_RES).fill(-1);

        for (let py = 0; py < DEPTH_RES; py++) {
            for (let px = 0; px < DEPTH_RES; px++) {
                const ndcX = (px + 0.5) / DEPTH_RES * 2 - 1;
                const ndcY = 1 - (py + 0.5) / DEPTH_RES * 2;
                const rdx = fwd[0] + ndcX * tanFOV * right[0] + ndcY * tanFOV * up[0];
                const rdy = fwd[1] + ndcX * tanFOV * right[1] + ndcY * tanFOV * up[1];
                const rdz = fwd[2] + ndcX * tanFOV * right[2] + ndcY * tanFOV * up[2];
                const rdLen = Math.sqrt(rdx ** 2 + rdy ** 2 + rdz ** 2);
                const dir = [rdx / rdLen, rdy / rdLen, rdz / rdLen];
                const t = sceneType === 'sphere' ? raySphere(pos, dir) : rayTorus(pos, dir);
                if (t >= 0) depthMap[py * DEPTH_RES + px] = t;
            }
        }

        self.postMessage({ type: 'depth_progress', data: { camIndex: camIdx, depthMap } });
        await new Promise(r => setTimeout(r, 60));
    }

    // Build the point cloud to match the gsplat visual distribution.
    currentPoints = gaussianPoints
        ? samplePointCloudFromGaussians()
        : samplePointCloudFromSurface();

    self.postMessage({ type: 'points', data: currentPoints });
}

/**
 * Build a point cloud directly from gaussian positions (1:1, no noise added).
 * Artifact positions (floaters, needles, blobs) are already baked into gaussianPoints
 * by createGaussianCloud() in main.js, so the distribution mirrors the visual splat.
 */
function samplePointCloudFromGaussians() {
    const N = gaussianPoints.length / 6; // [x,y,z,nx,ny,nz] per entry
    const points = new Float32Array(N * 7);
    for (let i = 0; i < N; i++) {
        points[i * 7]     = gaussianPoints[i * 6];
        points[i * 7 + 1] = gaussianPoints[i * 6 + 1];
        points[i * 7 + 2] = gaussianPoints[i * 6 + 2];
        points[i * 7 + 3] = gaussianPoints[i * 6 + 3];
        points[i * 7 + 4] = gaussianPoints[i * 6 + 4];
        points[i * 7 + 5] = gaussianPoints[i * 6 + 5];
        points[i * 7 + 6] = observationWeight;
    }
    return points;
}

/**
 * Fallback: build point cloud by back-projecting depth map hits against the
 * analytical surface. Used when no gaussian data is available.
 */
function samplePointCloudFromSurface() {
    const halfFOV = (CAM_FOV_DEG * Math.PI / 180) / 2;
    const tanFOV  = Math.tan(halfFOV);
    const allPoints = [];

    for (let camIdx = 0; camIdx < CAM_COUNT; camIdx++) {
        const { pos, fwd, right, up } = getCameraAxes(camIdx);
        for (let py = 0; py < DEPTH_RES; py++) {
            for (let px = 0; px < DEPTH_RES; px++) {
                const ndcX = (px + 0.5) / DEPTH_RES * 2 - 1;
                const ndcY = 1 - (py + 0.5) / DEPTH_RES * 2;
                const rdx = fwd[0] + ndcX * tanFOV * right[0] + ndcY * tanFOV * up[0];
                const rdy = fwd[1] + ndcX * tanFOV * right[1] + ndcY * tanFOV * up[1];
                const rdz = fwd[2] + ndcX * tanFOV * right[2] + ndcY * tanFOV * up[2];
                const rdLen = Math.sqrt(rdx ** 2 + rdy ** 2 + rdz ** 2);
                const dir = [rdx / rdLen, rdy / rdLen, rdz / rdLen];
                const t = sceneType === 'sphere' ? raySphere(pos, dir) : rayTorus(pos, dir);
                if (t < 0) continue;
                const hx = pos[0] + t * dir[0], hy = pos[1] + t * dir[1], hz = pos[2] + t * dir[2];
                let nx, ny, nz;
                if (sceneType === 'sphere') {
                    const nLen = Math.sqrt(hx ** 2 + hy ** 2 + hz ** 2);
                    nx = hx / nLen; ny = hy / nLen; nz = hz / nLen;
                } else {
                    const eps = 0.001;
                    const gx = sdTorus([hx + eps, hy, hz]) - sdTorus([hx - eps, hy, hz]);
                    const gy = sdTorus([hx, hy + eps, hz]) - sdTorus([hx, hy - eps, hz]);
                    const gz = sdTorus([hx, hy, hz + eps]) - sdTorus([hx, hy, hz - eps]);
                    const gLen = Math.sqrt(gx ** 2 + gy ** 2 + gz ** 2);
                    nx = gx / gLen; ny = gy / gLen; nz = gz / gLen;
                }
                if (nx * (-dir[0]) + ny * (-dir[1]) + nz * (-dir[2]) < 0) { nx = -nx; ny = -ny; nz = -nz; }
                allPoints.push(
                    hx + gaussianRandom() * noiseStd,
                    hy + gaussianRandom() * noiseStd,
                    hz + gaussianRandom() * noiseStd,
                    nx, ny, nz, observationWeight,
                );
            }
        }
    }

    const numOutliers = Math.floor(allPoints.length / 7 * outlierRate);
    for (let i = 0; i < numOutliers; i++) {
        const ox = (Math.random() - 0.5) * 4, oy = (Math.random() - 0.5) * 4, oz = (Math.random() - 0.5) * 4;
        let onx = Math.random() - 0.5, ony = Math.random() - 0.5, onz = Math.random() - 0.5;
        const onLen = Math.sqrt(onx ** 2 + ony ** 2 + onz ** 2);
        allPoints.push(ox, oy, oz, onx / onLen, ony / onLen, onz / onLen, observationWeight);
    }
    return new Float32Array(allPoints);
}

// ---------------------------------------------------------------------------
// Ray intersection helpers
// ---------------------------------------------------------------------------

/** Analytic ray-sphere intersection. Returns t ≥ 0, or -1 on miss. */
function raySphere(ro, rd, radius = 1.0) {
    const b = 2 * (ro[0] * rd[0] + ro[1] * rd[1] + ro[2] * rd[2]);
    const c = ro[0] ** 2 + ro[1] ** 2 + ro[2] ** 2 - radius ** 2;
    const disc = b * b - 4 * c;
    if (disc < 0) return -1;
    const sq = Math.sqrt(disc);
    const t1 = (-b - sq) / 2;
    const t2 = (-b + sq) / 2;
    if (t2 < 0) return -1;
    return t1 >= 0 ? t1 : t2;
}

/** Torus SDF: major radius R=1.0, minor radius r=0.4 */
function sdTorus(p) {
    const q = Math.sqrt(p[0] ** 2 + p[2] ** 2) - 1.0;
    return Math.sqrt(q ** 2 + p[1] ** 2) - 0.4;
}

/** Ray-march against torus SDF. Returns t ≥ 0, or -1 on miss. */
function rayTorus(ro, rd) {
    let t = 0;
    for (let i = 0; i < 150; i++) {
        const p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
        const d = sdTorus(p);
        if (Math.abs(d) < 0.001) return t;
        t += d * 0.9; // slight under-stepping for robustness
        if (t > 12) return -1;
    }
    return -1;
}

/** Box-Muller Gaussian noise */
function gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ---------------------------------------------------------------------------
// Step 3: Voxelization
// ---------------------------------------------------------------------------

function getAffectedVoxels(vol, points) {
    const numPoints = points.length / 7;
    const affectedIndices = new Set();
    const { voxelSize, extent, size, range } = vol;

    for (let i = 0; i < numPoints; i++) {
        const px = points[i * 7];
        const py = points[i * 7 + 1];
        const pz = points[i * 7 + 2];

        const nix = Math.floor((px + extent) / voxelSize);
        const niy = Math.floor((py + extent) / voxelSize);
        const niz = Math.floor((pz + extent) / voxelSize);

        for (let dx = -range; dx <= range; dx++) {
            for (let dy = -range; dy <= range; dy++) {
                for (let dz = -range; dz <= range; dz++) {
                    const ix = nix + dx;
                    const iy = niy + dy;
                    const iz = niz + dz;
                    if (ix < 0 || ix >= size || iy < 0 || iy >= size || iz < 0 || iz >= size) continue;
                    affectedIndices.add(ix + iy * size + iz * vol.sizeSq);
                }
            }
        }
    }

    const centers = new Float32Array(affectedIndices.size * 3);
    let count = 0;
    for (const idx of affectedIndices) {
        const iz = Math.floor(idx / vol.sizeSq);
        const iy = Math.floor((idx % vol.sizeSq) / size);
        const ix = idx % size;
        centers[count * 3]     = (ix + 0.5) * voxelSize - extent;
        centers[count * 3 + 1] = (iy + 0.5) * voxelSize - extent;
        centers[count * 3 + 2] = (iz + 0.5) * voxelSize - extent;
        count++;
    }
    return centers;
}

// ---------------------------------------------------------------------------
// Step 4: Progressive TSDF fusion
// ---------------------------------------------------------------------------

async function fuseProgressive(vol, points) {
    const numPoints = points.length / 7;
    const chunkSize = 500;
    const reportEvery = 2;
    const startTime = performance.now();
    let chunkIndex = 0;

    for (let i = 0; i < numPoints; i += chunkSize) {
        const end = Math.min(i + chunkSize, numPoints);
        vol.fuse(points.subarray(i * 7, end * 7));
        chunkIndex++;

        const isDone = end === numPoints;
        if (isDone || chunkIndex % reportEvery === 0) {
            self.postMessage({
                type: 'fusing_progress',
                data: {
                    distances: vol.getGridData(),
                    progress: end / numPoints,
                    isDone,
                    time: performance.now() - startTime,
                },
            });
        }

        await new Promise(r => setTimeout(r, 80));
    }
}
