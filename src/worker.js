import { TSDFVolume } from './tsdf.js';
import { SurfaceSampler } from './sampling.js';

let volume = null;
let sampler = null;
let currentPoints = null;

self.onmessage = (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            volume = new TSDFVolume(data);
            sampler = new SurfaceSampler(data);
            self.postMessage({ type: 'ready' });
            break;

        case 'step1_generate':
            // State: Point Cloud Generation
            currentPoints = sampler.generate();
            // We send a copy so we can keep currentPoints in worker memory
            self.postMessage({ type: 'points', data: currentPoints });
            break;

        case 'step2_voxelize':
            // State: Voxelization (Finding affected voxels)
            if (volume && currentPoints) {
                const affectedVoxels = getAffectedVoxels(volume, currentPoints);
                self.postMessage({ type: 'voxels', data: affectedVoxels });
            }
            break;

        case 'step3_fuse':
            // State: Progressive Fusion Logic
            if (volume && currentPoints) {
                fuseProgressive(volume, currentPoints);
            }
            break;

        case 'step4_extract':
            // State: Marching Cubes
            if (volume) {
                const distances = volume.getGridData();
                self.postMessage({ type: 'extracted', data: distances });
            }
            break;

        case 'reset':
            if (volume) {
                volume.distances.fill(1.0);
                volume.weights.fill(0.0);
                currentPoints = null;
                self.postMessage({ type: 'reset_done' });
            }
            break;
    }
};

async function fuseProgressive(vol, points) {
    const numPoints = points.length / 7;
    const chunkSize = 500;
    // Send a visual update every N chunks to halve postMessage overhead
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

/**
 * Returns the centers of voxels that are within the truncation distance of any point
 */
function getAffectedVoxels(vol, points) {
    const numPoints = points.length / 7;
    const affectedIndices = new Set();
    const voxelSize = vol.voxelSize;
    const extent = vol.extent;
    const size = vol.size;
    const mu = vol.mu;

    for (let i = 0; i < numPoints; i++) {
        const px = points[i * 7];
        const py = points[i * 7 + 1];
        const pz = points[i * 7 + 2];
        
        const range = Math.ceil(mu / voxelSize);
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

                    const idx = ix + iy * size + iz * size * size;
                    affectedIndices.add(idx);
                }
            }
        }
    }

    // Convert indices to centers for visualization
    const centers = new Float32Array(affectedIndices.size * 3);
    let count = 0;
    for (const idx of affectedIndices) {
        const iz = Math.floor(idx / (size * size));
        const iy = Math.floor((idx % (size * size)) / size);
        const ix = idx % size;

        centers[count * 3] = (ix + 0.5) * voxelSize - extent;
        centers[count * 3 + 1] = (iy + 0.5) * voxelSize - extent;
        centers[count * 3 + 2] = (iz + 0.5) * voxelSize - extent;
        count++;
    }

    return centers;
}
