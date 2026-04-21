import { TSDFVolume } from './tsdf.js';

let frameVolume = null;
let globalVolume = null;
let activeCameraIndex = 0;
let latestVolumeKind = null;

let sceneType = 'sphere';
let noiseStd = 0.02;
let outlierRate = 0.05;
let observationWeight = 1.0;
let gsplatNoiseScale = 1.0;
let gaussianPoints = null;

let cameraPoints = null;
let cameraDepthMaps = null;
let framePoints = null;
let frameDepthMap = null;

/** Must match the constants in main.js */
const CAM_COUNT = 8;
const CAM_RADIUS = 2.5;
const CAM_HEIGHT = 1.0;
const CAM_FOV_DEG = 45;
export const DEPTH_RES = 48;

self.onmessage = (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            initState(data);
            self.postMessage({ type: 'ready' });
            break;

        case 'acquire_depth':
            ensureObservations();
            setActiveCamera(data.cameraIndex ?? 0);
            self.postMessage({
                type: 'depth_acquired',
                data: {
                    cameraIndex: activeCameraIndex,
                    depthMap: frameDepthMap,
                    points: framePoints,
                },
            });
            break;

        case 'generate_tsdf':
            ensureObservations();
            updateConfig(data);
            setActiveCamera(data.cameraIndex ?? activeCameraIndex);
            buildFrameTSDF();
            const frameDebug = summarizeVolume(frameVolume);
            self.postMessage({
                type: 'frame_tsdf_generated',
                data: {
                    cameraIndex: activeCameraIndex,
                    voxelCenters: getAffectedVoxels(frameVolume, framePoints),
                    distances: frameVolume.distances,
                    weights: frameVolume.weights,
                    debug: frameDebug,
                },
            });
            break;

        case 'fuse_tsdf':
            updateConfig(data);
            if (!frameVolume) buildFrameTSDF();
            const hadGlobal = !!globalVolume;
            const previousGlobal = globalVolume ? globalVolume.clone() : null;
            if (!globalVolume) {
                globalVolume = frameVolume.clone();
            } else {
                globalVolume.fuseVolume(frameVolume);
            }
            latestVolumeKind = 'global';
            const globalDebug = summarizeVolume(globalVolume);
            const fuseDebug = {
                frame: summarizeVolume(frameVolume),
                global: globalDebug,
                changedVoxels: previousGlobal ? countChangedVoxels(previousGlobal, globalVolume) : globalDebug.observedVoxels,
            };
            self.postMessage({
                type: 'tsdf_fused',
                data: {
                    cameraIndex: activeCameraIndex,
                    skipped: !hadGlobal,
                    distances: globalVolume.distances,
                    weights: globalVolume.weights,
                    debug: fuseDebug,
                },
            });
            break;

        case 'extract_mesh':
            self.postMessage({
                type: 'mesh_ready',
                data: {
                    cameraIndex: activeCameraIndex,
                    source: latestVolumeKind,
                    distances: getLatestVolume()?.distances ?? null,
                },
            });
            break;

        case 'reset':
            resetRuntimeState();
            self.postMessage({ type: 'reset_done' });
            break;
    }
};

function initState(data = {}) {
    sceneType = data.type || 'sphere';
    noiseStd = data.noise || 0.02;
    outlierRate = data.outliers || 0.05;
    observationWeight = data.observationWeight ?? 1.0;
    gsplatNoiseScale = data.gsplatNoise ?? 1.0;
    gaussianPoints = data.gaussianPoints ?? null;
    activeCameraIndex = 0;
    latestVolumeKind = null;
    cameraPoints = null;
    cameraDepthMaps = null;
    framePoints = null;
    frameDepthMap = null;
    frameVolume = createVolume(data);
    globalVolume = null;
}

function resetRuntimeState() {
    frameVolume?.reset();
    frameVolume = frameVolume ? frameVolume.clone() : null;
    globalVolume = null;
    latestVolumeKind = null;
    cameraPoints = null;
    cameraDepthMaps = null;
    framePoints = null;
    frameDepthMap = null;
    activeCameraIndex = 0;
}

function createVolume(config = {}) {
    return new TSDFVolume({
        size: config.size ?? frameVolume?.size ?? globalVolume?.size ?? 64,
        extent: config.extent ?? frameVolume?.extent ?? globalVolume?.extent ?? 2.0,
        mu: config.mu ?? frameVolume?.mu ?? globalVolume?.mu ?? 0.1,
        maxTSDFWeight: config.maxTSDFWeight ?? frameVolume?.maxTSDFWeight ?? globalVolume?.maxTSDFWeight ?? 32,
    });
}

function updateConfig(config = {}) {
    const nextSize = config.size ?? frameVolume?.size ?? globalVolume?.size ?? 64;
    const nextExtent = config.extent ?? frameVolume?.extent ?? globalVolume?.extent ?? 2.0;
    const nextMu = config.mu ?? frameVolume?.mu ?? globalVolume?.mu ?? 0.1;
    const nextMaxWeight = config.maxTSDFWeight ?? frameVolume?.maxTSDFWeight ?? globalVolume?.maxTSDFWeight ?? 32;

    const shouldRebuild = !frameVolume
        || frameVolume.size !== nextSize
        || frameVolume.extent !== nextExtent;

    if (shouldRebuild) {
        frameVolume = createVolume({
            size: nextSize,
            extent: nextExtent,
            mu: nextMu,
            maxTSDFWeight: nextMaxWeight,
        });
        globalVolume = null;
        latestVolumeKind = null;
        return;
    }

    frameVolume.mu = nextMu;
    frameVolume.range = Math.ceil(nextMu / frameVolume.voxelSize);
    frameVolume.maxTSDFWeight = nextMaxWeight;

    if (globalVolume) {
        globalVolume.mu = nextMu;
        globalVolume.range = Math.ceil(nextMu / globalVolume.voxelSize);
        globalVolume.maxTSDFWeight = nextMaxWeight;
    }

    if (config.observationWeight !== undefined) observationWeight = config.observationWeight;
}

function ensureObservations() {
    if (cameraPoints && cameraDepthMaps) return;

    const observations = gaussianPoints
        ? buildObservationsFromGaussians()
        : buildObservationsFromAnalyticSurface();

    cameraDepthMaps = observations.depthMaps;
    cameraPoints = observations.pointClouds;
    setActiveCamera(activeCameraIndex);
}

function setActiveCamera(cameraIndex) {
    activeCameraIndex = Math.max(0, Math.min(CAM_COUNT - 1, cameraIndex));
    frameDepthMap = cameraDepthMaps?.[activeCameraIndex] ?? null;
    framePoints = cameraPoints?.[activeCameraIndex] ?? null;
}

function buildFrameTSDF() {
    frameVolume = createVolume({
        size: frameVolume?.size,
        extent: frameVolume?.extent,
        mu: frameVolume?.mu,
        maxTSDFWeight: frameVolume?.maxTSDFWeight,
    });
    if (framePoints?.length) frameVolume.fuse(framePoints);
    latestVolumeKind = 'frame';
}

function getLatestVolume() {
    if (latestVolumeKind === 'global') return globalVolume;
    if (latestVolumeKind === 'frame') return frameVolume;
    return globalVolume || frameVolume;
}

function summarizeVolume(volume) {
    if (!volume) {
        return {
            observedVoxels: 0,
            weightSum: 0,
            maxWeight: 0,
            meanAbsDistance: 0,
        };
    }

    let observedVoxels = 0;
    let weightSum = 0;
    let maxWeight = 0;
    let absDistanceSum = 0;

    for (let i = 0; i < volume.numVoxels; i++) {
        const weight = volume.weights[i];
        if (weight <= 0) continue;
        observedVoxels++;
        weightSum += weight;
        maxWeight = Math.max(maxWeight, weight);
        absDistanceSum += Math.abs(volume.distances[i]);
    }

    return {
        observedVoxels,
        weightSum,
        maxWeight,
        meanAbsDistance: observedVoxels > 0 ? absDistanceSum / observedVoxels : 0,
    };
}

function countChangedVoxels(beforeVolume, afterVolume, epsilon = 1e-6) {
    let changed = 0;
    for (let i = 0; i < afterVolume.numVoxels; i++) {
        if (Math.abs(beforeVolume.weights[i] - afterVolume.weights[i]) > epsilon
            || Math.abs(beforeVolume.distances[i] - afterVolume.distances[i]) > epsilon) {
            changed++;
        }
    }
    return changed;
}

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
    const fwd = [-pos[0] / L, -pos[1] / L, -pos[2] / L];

    const rx = -fwd[2];
    const rz = fwd[0];
    const rLen = Math.sqrt(rx * rx + rz * rz);
    const right = [rx / rLen, 0, rz / rLen];

    const ux = right[1] * fwd[2] - right[2] * fwd[1];
    const uy = right[2] * fwd[0] - right[0] * fwd[2];
    const uz = right[0] * fwd[1] - right[1] * fwd[0];
    const uLen = Math.sqrt(ux * ux + uy * uy + uz * uz);

    return { pos, fwd, right, up: [ux / uLen, uy / uLen, uz / uLen] };
}

function buildObservationsFromGaussians() {
    const halfFOV = (CAM_FOV_DEG * Math.PI / 180) / 2;
    const tanFOV = Math.tan(halfFOV);
    const depthMaps = [];
    const normalMaps = [];

    for (let camIdx = 0; camIdx < CAM_COUNT; camIdx++) {
        depthMaps.push(new Float32Array(DEPTH_RES * DEPTH_RES).fill(-1));
        normalMaps.push(new Float32Array(DEPTH_RES * DEPTH_RES * 3));
    }

    const numGaussians = gaussianPoints.length / 6;
    for (let i = 0; i < numGaussians; i++) {
        const px = gaussianPoints[i * 6];
        const py = gaussianPoints[i * 6 + 1];
        const pz = gaussianPoints[i * 6 + 2];
        const baseNx = gaussianPoints[i * 6 + 3];
        const baseNy = gaussianPoints[i * 6 + 4];
        const baseNz = gaussianPoints[i * 6 + 5];

        for (let camIdx = 0; camIdx < CAM_COUNT; camIdx++) {
            const { pos, fwd, right, up } = getCameraAxes(camIdx);
            const rx = px - pos[0];
            const ry = py - pos[1];
            const rz = pz - pos[2];
            const camZ = rx * fwd[0] + ry * fwd[1] + rz * fwd[2];
            if (camZ <= 0) continue;

            const camX = rx * right[0] + ry * right[1] + rz * right[2];
            const camY = rx * up[0] + ry * up[1] + rz * up[2];
            const ndcX = camX / (camZ * tanFOV);
            const ndcY = camY / (camZ * tanFOV);
            if (Math.abs(ndcX) > 1 || Math.abs(ndcY) > 1) continue;

            const pixX = Math.floor(((ndcX + 1) * 0.5) * DEPTH_RES);
            const pixY = Math.floor(((1 - ndcY) * 0.5) * DEPTH_RES);
            if (pixX < 0 || pixX >= DEPTH_RES || pixY < 0 || pixY >= DEPTH_RES) continue;

            const range = Math.sqrt(rx * rx + ry * ry + rz * rz);
            const pixelIndex = pixY * DEPTH_RES + pixX;
            const depthMap = depthMaps[camIdx];
            if (depthMap[pixelIndex] >= 0 && depthMap[pixelIndex] <= range) continue;

            let nx = baseNx;
            let ny = baseNy;
            let nz = baseNz;
            if (nx * (-rx) + ny * (-ry) + nz * (-rz) < 0) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            depthMap[pixelIndex] = range;
            const normalMap = normalMaps[camIdx];
            normalMap[pixelIndex * 3] = nx;
            normalMap[pixelIndex * 3 + 1] = ny;
            normalMap[pixelIndex * 3 + 2] = nz;
        }
    }

    return {
        depthMaps,
        pointClouds: depthMaps.map((depthMap, camIdx) => backProjectDepthMap(camIdx, depthMap, normalMaps[camIdx])),
    };
}

function buildObservationsFromAnalyticSurface() {
    const halfFOV = (CAM_FOV_DEG * Math.PI / 180) / 2;
    const tanFOV = Math.tan(halfFOV);
    const depthMaps = [];
    const normalMaps = [];

    for (let camIdx = 0; camIdx < CAM_COUNT; camIdx++) {
        const { pos, fwd, right, up } = getCameraAxes(camIdx);
        const depthMap = new Float32Array(DEPTH_RES * DEPTH_RES).fill(-1);
        const normalMap = new Float32Array(DEPTH_RES * DEPTH_RES * 3);

        for (let py = 0; py < DEPTH_RES; py++) {
            for (let px = 0; px < DEPTH_RES; px++) {
                const ndcX = (px + 0.5) / DEPTH_RES * 2 - 1;
                const ndcY = 1 - (py + 0.5) / DEPTH_RES * 2;
                const dir = getPixelRayDirection(fwd, right, up, tanFOV, ndcX, ndcY);
                const t = sceneType === 'sphere' ? raySphere(pos, dir) : rayTorus(pos, dir);
                if (t < 0) continue;

                const hx = pos[0] + t * dir[0];
                const hy = pos[1] + t * dir[1];
                const hz = pos[2] + t * dir[2];
                const idx = py * DEPTH_RES + px;
                depthMap[idx] = t;

                let nx;
                let ny;
                let nz;
                if (sceneType === 'sphere') {
                    const nLen = Math.sqrt(hx ** 2 + hy ** 2 + hz ** 2);
                    nx = hx / nLen;
                    ny = hy / nLen;
                    nz = hz / nLen;
                } else {
                    const eps = 0.001;
                    const gx = sdTorus([hx + eps, hy, hz]) - sdTorus([hx - eps, hy, hz]);
                    const gy = sdTorus([hx, hy + eps, hz]) - sdTorus([hx, hy - eps, hz]);
                    const gz = sdTorus([hx, hy, hz + eps]) - sdTorus([hx, hy, hz - eps]);
                    const gLen = Math.sqrt(gx ** 2 + gy ** 2 + gz ** 2);
                    nx = gx / gLen;
                    ny = gy / gLen;
                    nz = gz / gLen;
                }
                if (nx * (-dir[0]) + ny * (-dir[1]) + nz * (-dir[2]) < 0) {
                    nx = -nx;
                    ny = -ny;
                    nz = -nz;
                }
                normalMap[idx * 3] = nx;
                normalMap[idx * 3 + 1] = ny;
                normalMap[idx * 3 + 2] = nz;
            }
        }

        depthMaps.push(depthMap);
        normalMaps.push(normalMap);
    }

    return {
        depthMaps,
        pointClouds: depthMaps.map((depthMap, camIdx) => backProjectDepthMap(camIdx, depthMap, normalMaps[camIdx])),
    };
}

function backProjectDepthMap(camIdx, depthMap, normalMap) {
    const halfFOV = (CAM_FOV_DEG * Math.PI / 180) / 2;
    const tanFOV = Math.tan(halfFOV);
    const { pos, fwd, right, up } = getCameraAxes(camIdx);
    const points = [];
    const measurementNoiseStd = gaussianPoints ? noiseStd * gsplatNoiseScale : noiseStd;

    for (let py = 0; py < DEPTH_RES; py++) {
        for (let px = 0; px < DEPTH_RES; px++) {
            const idx = py * DEPTH_RES + px;
            const depth = depthMap[idx];
            if (depth < 0) continue;

            const ndcX = (px + 0.5) / DEPTH_RES * 2 - 1;
            const ndcY = 1 - (py + 0.5) / DEPTH_RES * 2;
            const dir = getPixelRayDirection(fwd, right, up, tanFOV, ndcX, ndcY);

            const pxWorld = pos[0] + depth * dir[0] + gaussianRandom() * measurementNoiseStd;
            const pyWorld = pos[1] + depth * dir[1] + gaussianRandom() * measurementNoiseStd;
            const pzWorld = pos[2] + depth * dir[2] + gaussianRandom() * measurementNoiseStd;

            let nx = normalMap[idx * 3];
            let ny = normalMap[idx * 3 + 1];
            let nz = normalMap[idx * 3 + 2];
            const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
            if (nLen === 0) continue;
            nx /= nLen;
            ny /= nLen;
            nz /= nLen;

            points.push(pxWorld, pyWorld, pzWorld, nx, ny, nz, observationWeight);
        }
    }

    const totalPoints = points.length / 7;
    const numOutliers = Math.floor(totalPoints * outlierRate);
    for (let i = 0; i < numOutliers; i++) {
        const ox = (Math.random() - 0.5) * 4;
        const oy = (Math.random() - 0.5) * 4;
        const oz = (Math.random() - 0.5) * 4;
        let onx = Math.random() - 0.5;
        let ony = Math.random() - 0.5;
        let onz = Math.random() - 0.5;
        const onLen = Math.sqrt(onx ** 2 + ony ** 2 + onz ** 2);
        points.push(ox, oy, oz, onx / onLen, ony / onLen, onz / onLen, observationWeight);
    }

    return new Float32Array(points);
}

function getPixelRayDirection(fwd, right, up, tanFOV, ndcX, ndcY) {
    const rdx = fwd[0] + ndcX * tanFOV * right[0] + ndcY * tanFOV * up[0];
    const rdy = fwd[1] + ndcX * tanFOV * right[1] + ndcY * tanFOV * up[1];
    const rdz = fwd[2] + ndcX * tanFOV * right[2] + ndcY * tanFOV * up[2];
    const rLen = Math.sqrt(rdx ** 2 + rdy ** 2 + rdz ** 2);
    return [rdx / rLen, rdy / rLen, rdz / rLen];
}

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

function sdTorus(p) {
    const q = Math.sqrt(p[0] ** 2 + p[2] ** 2) - 1.0;
    return Math.sqrt(q ** 2 + p[1] ** 2) - 0.4;
}

function rayTorus(ro, rd) {
    let t = 0;
    for (let i = 0; i < 150; i++) {
        const p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
        const d = sdTorus(p);
        if (Math.abs(d) < 0.001) return t;
        t += d * 0.9;
        if (t > 12) return -1;
    }
    return -1;
}

function gaussianRandom() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function getAffectedVoxels(vol, points) {
    if (!vol || !points) return new Float32Array(0);

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
        centers[count * 3] = (ix + 0.5) * voxelSize - extent;
        centers[count * 3 + 1] = (iy + 0.5) * voxelSize - extent;
        centers[count * 3 + 2] = (iz + 0.5) * voxelSize - extent;
        count++;
    }
    return centers;
}
