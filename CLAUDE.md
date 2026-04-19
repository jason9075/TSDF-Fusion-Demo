# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Enter Nix dev shell
nix develop

# Start dev server with auto-reload (port 8080)
just dev
```

`just dev` uses `entr` to watch `*.html`, `*.js`, `*.css` and restarts `python3 -m http.server 8080` on changes.

No build step — all source files are loaded directly in the browser via ES module import maps.

## Architecture

This is a browser-only, zero-build educational visualization of **TSDF (Truncated Signed Distance Function) fusion** and mesh extraction.

### Thread Model

- **Main thread** (`src/main.js`): Three.js scene, lil-gui controls, OrbitControls, UI state
- **Web Worker** (`src/worker.js`): All CPU-bound TSDF computation — never block the main thread with fusion logic

Communication is via `postMessage`. Worker messages: `init`, `step1_generate`, `step2_voxelize`, `step3_fuse`, `step4_extract`, `reset`.

### 4-Step Pipeline

Each step maps to a GUI button and a worker message:

| Step | Button | Worker Op | Visualization |
|------|--------|-----------|---------------|
| 1 | Generate Points | `step1_generate` | Red point cloud |
| 2 | Voxelize | `step2_voxelize` | Yellow voxels |
| 3 | Fuse | `step3_fuse` | Progressive fusion + slicing plane heatmap |
| 4 | Extract Mesh | `step4_extract` | Marching Cubes mesh (frost blue) |

Step 3 uses 500-point chunks with 80ms async yields to keep UI responsive during fusion.

### Core Data Structures

- **`TSDFVolume`** (`src/tsdf.js`): 1D `Float32Array` for both `distances` and `weights`, indexed as `ix + iy*size + iz*size²`. Algorithm follows Curless & Levoy (1996) weighted averaging.
- **`SurfaceSampler`** (`src/sampling.js`): Generates synthetic point clouds (sphere/torus) as `Float32Array([x,y,z, nx,ny,nz, weight, ...])`.

### Module Loading

No bundler. Dependencies loaded via import map in `index.html`:
- Three.js 0.160.0 (CDN)
- lil-gui 0.19.1 (CDN)
- KaTeX 0.16.9 (CDN, for math rendering in knowledge modal)

Local modules under `src/` use relative ESM imports.

### UI/UX Notes

- **Knowledge modal**: Bilingual EN/ZH toggle. KaTeX re-renders on language switch.
- **Nord color theme**: `style.css` uses the Nord palette throughout — keep new UI consistent with it.
- **Slicing plane**: Renders a 2D TSDF cross-section with a heatmap colormap; driven by a GUI slider for Z position.
