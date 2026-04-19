/**
 * Synthetic Data Generation for TSDF Fusion Demo
 */

export class SurfaceSampler {
    /**
     * @param {Object} options
     * @param {string} options.type - 'sphere' | 'torus'
     * @param {number} options.numPoints - Number of points to sample
     * @param {number} options.noise - Gaussian noise standard deviation
     * @param {number} options.outliers - Percentage of outliers (0-1)
     */
    constructor(options = {}) {
        this.type = options.type || 'sphere';
        this.numPoints = options.numPoints || 5000;
        this.noise = options.noise || 0.02;
        this.outliers = options.outliers || 0.05;
    }

    generate() {
        const points = new Float32Array(this.numPoints * 4); // [x, y, z, weight]
        
        for (let i = 0; i < this.numPoints; i++) {
            let x, y, z;
            
            if (Math.random() < this.outliers) {
                // Generate outlier
                x = (Math.random() - 0.5) * 4;
                y = (Math.random() - 0.5) * 4;
                z = (Math.random() - 0.5) * 4;
            } else {
                if (this.type === 'sphere') {
                    const phi = Math.random() * Math.PI * 2;
                    const theta = Math.acos(2 * Math.random() - 1);
                    const r = 1.0;
                    x = r * Math.sin(theta) * Math.cos(phi);
                    y = r * Math.sin(theta) * Math.sin(phi);
                    z = r * Math.cos(theta);
                } else if (this.type === 'torus') {
                    const phi = Math.random() * Math.PI * 2;
                    const theta = Math.random() * Math.PI * 2;
                    const R = 1.0;
                    const r = 0.4;
                    x = (R + r * Math.cos(theta)) * Math.cos(phi);
                    y = (R + r * Math.cos(theta)) * Math.sin(phi);
                    z = r * Math.sin(theta);
                }

                // Add Gaussian noise
                x += this.gaussianRandom() * this.noise;
                y += this.gaussianRandom() * this.noise;
                z += this.gaussianRandom() * this.noise;
            }

            points[i * 4] = x;
            points[i * 4 + 1] = y;
            points[i * 4 + 2] = z;
            points[i * 4 + 3] = 1.0; // Initial weight
        }

        return points;
    }

    /**
     * Box-Muller transform for Gaussian noise
     */
    gaussianRandom() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}
