import { TSDFVolume } from './tsdf.js';
import { SurfaceSampler } from './sampling.js';

let volume = null;
let sampler = null;

self.onmessage = (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            volume = new TSDFVolume(data);
            sampler = new SurfaceSampler(data);
            self.postMessage({ type: 'ready' });
            break;

        case 'generate':
            const points = sampler.generate();
            self.postMessage({ type: 'points', data: points }, [points.buffer]);
            break;

        case 'fuse':
            if (volume && data.points) {
                const startTime = performance.now();
                volume.fuse(new Float32Array(data.points));
                const endTime = performance.now();
                
                // Return the grid data
                const distances = volume.getGridData();
                const weights = volume.weights;
                
                // We send a copy because it's safer for now, 
                // but we could use transferable objects if we want more speed.
                self.postMessage({ 
                    type: 'fused', 
                    data: {
                        distances: distances,
                        weights: weights,
                        time: endTime - startTime
                    }
                });
            }
            break;

        case 'reset':
            if (volume) {
                volume.distances.fill(1.0);
                volume.weights.fill(0.0);
                self.postMessage({ type: 'reset_done' });
            }
            break;
    }
};
