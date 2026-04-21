# TSDF Fusion Demo

An interactive browser demo for learning how TSDF fusion, frame/global volumes, and Marching Cubes mesh extraction work.

This project visualizes a simplified KinectFusion-style pipeline:

1. Acquire depth from a virtual camera
2. Generate a frame TSDF
3. Fuse the frame TSDF into a global TSDF
4. Extract a mesh with Marching Cubes

The demo is implemented as a zero-build web app with `Three.js`, `lil-gui`, and a `Web Worker` for TSDF computation.

## Links

- Repository: https://github.com/jason9075/TSDF-Fusion-Demo
- Live Demo: https://jason9075.github.io/TSDF-Fusion-Demo/

## Features

- Step-by-step TSDF reconstruction pipeline
- Separate frame voxels and global voxels
- Auto Play with pause, next-step, stop, and fast-forward controls
- Interactive voxel hover info for distance and weight
- Adjustable TSDF and Marching Cubes parameters
- Synthetic Gaussian splat scene generation

## Project Structure

```text
.
├── index.html        # App entry point
├── style.css         # Shared styling
├── src/
│   ├── main.js       # Three.js scene, UI, playback flow
│   ├── worker.js     # TSDF pipeline in a Web Worker
│   ├── tsdf.js       # TSDF volume implementation
│   └── sampling.js   # Sampling helpers
├── flake.nix         # Nix dev shell
└── Justfile          # Dev shortcuts
```

## Local Development

If you use Nix:

```bash
nix develop
just dev
```

Then open:

```text
http://localhost:8080
```

Without Nix, you can use a simple static server:

```bash
python3 -m http.server 8080
```

## Notes

- There is no bundler or build step.
- ES modules are loaded directly in the browser.
- Heavy TSDF work runs in `src/worker.js`.

## License

MIT. See [LICENSE](LICENSE).
