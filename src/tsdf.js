/**
 * TSDF (Truncated Signed Distance Function) implementation
 */

export class TSDFVolume {
    /**
     * @param {Object} options
     * @param {number} options.size - Grid resolution (N x N x N)
     * @param {number} options.extent - World coordinate range (e.g. 2.0 means [-2, 2])
     * @param {number} options.mu - Truncation distance
     */
    constructor(options = {}) {
        this.size = options.size || 64;
        this.extent = options.extent || 2.0;
        this.mu = options.mu || 0.1;
        
        this.numVoxels = this.size * this.size * this.size;
        this.distances = new Float32Array(this.numVoxels).fill(1.0); // 1.0 is "far" (max distance)
        this.weights = new Float32Array(this.numVoxels).fill(0.0);
        
        this.voxelSize = (this.extent * 2) / this.size;
        this.sizeSq = this.size * this.size;
        this.range = Math.ceil(this.mu / this.voxelSize);
    }

    /**
     * Map world coordinates [x, y, z] to voxel index
     */
    getIndex(x, y, z) {
        const ix = Math.floor((x + this.extent) / this.voxelSize);
        const iy = Math.floor((y + this.extent) / this.voxelSize);
        const iz = Math.floor((z + this.extent) / this.voxelSize);

        if (ix < 0 || ix >= this.size || iy < 0 || iy >= this.size || iz < 0 || iz >= this.size) {
            return -1;
        }

        return ix + iy * this.size + iz * this.sizeSq;
    }

    /**
     * Fuse a point cloud into the volume
     * @param {Float32Array} points - [x, y, z, nx, ny, nz, weight, ...]
     */
    fuse(points) {
        const numPoints = points.length / 7;
        
        for (let i = 0; i < numPoints; i++) {
            const px = points[i * 7];
            const py = points[i * 7 + 1];
            const pz = points[i * 7 + 2];
            const nx = points[i * 7 + 3];
            const ny = points[i * 7 + 4];
            const nz = points[i * 7 + 5];
            const pw = points[i * 7 + 6];

            const nix = Math.floor((px + this.extent) / this.voxelSize);
            const niy = Math.floor((py + this.extent) / this.voxelSize);
            const niz = Math.floor((pz + this.extent) / this.voxelSize);

            for (let dx = -this.range; dx <= this.range; dx++) {
                for (let dy = -this.range; dy <= this.range; dy++) {
                    for (let dz = -this.range; dz <= this.range; dz++) {
                        const ix = nix + dx;
                        const iy = niy + dy;
                        const iz = niz + dz;

                        if (ix < 0 || ix >= this.size || iy < 0 || iy >= this.size || iz < 0 || iz >= this.size) continue;

                        const idx = ix + iy * this.size + iz * this.sizeSq;
                        
                        const vx = (ix + 0.5) * this.voxelSize - this.extent;
                        const vy = (iy + 0.5) * this.voxelSize - this.extent;
                        const vz = (iz + 0.5) * this.voxelSize - this.extent;

                        // Signed distance along normal: (V - P) · N
                        const sdf = (vx - px) * nx + (vy - py) * ny + (vz - pz) * nz;
                        
                        if (Math.abs(sdf) <= this.mu) {
                            const currentD = this.distances[idx];
                            const currentW = this.weights[idx];
                            
                            const newW = currentW + pw;
                            this.distances[idx] = (currentW * currentD + pw * sdf) / newW;
                            this.weights[idx] = newW;
                        }
                    }
                }
            }
        }
    }

    getGridData() {
        return this.distances;
    }
}
