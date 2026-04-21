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

- **Main thread** (`src/main.js`): Three.js scene, lil-gui controls, OrbitControls, UI state, auto-play orchestration
- **Web Worker** (`src/worker.js`): All CPU-bound TSDF computation — never block the main thread with fusion logic

Communication is via `postMessage`. Main → Worker messages:

| Message | Purpose |
|---------|---------|
| `init` | Initialize volume and scene config |
| `acquire_depth` | Render depth map from camera at `cameraIndex` |
| `update_gaussians` | Update Gaussian splat scene data |
| `generate_tsdf` | Build per-frame TSDF from current depth map |
| `fuse_tsdf` | Fuse frame TSDF into global TSDF |
| `extract_mesh` | Run Marching Cubes on latest volume |
| `reset` | Reset all runtime state |

Worker → Main replies use mirrored type names: `ready`, `depth_acquired`, `gaussians_updated`, `frame_tsdf_generated`, `tsdf_fused`, `mesh_ready`, `reset_done`.

### Observation Pipeline

The demo simulates **8 cameras** arranged in a ring (radius 2.5, height 1.0, FOV 45°). Each observation:
1. Acquires a 48×48 depth map (`DEPTH_RES = 48`) via ray-casting against the scene
2. Back-projects to a point cloud with Gaussian measurement noise
3. Builds a per-frame TSDF, then fuses it into the global TSDF

The UI exposes auto-play (sequential observation loop) and a manual "Next Step" button. `fastForward` mode skips per-step rendering delays.

### Scene Types

Two built-in analytic surfaces: **sphere** and **torus**. The worker supports an alternative **Gaussian splat** input path (`gaussianPoints`): if provided, depth maps are rendered by splatting Gaussian centres instead of ray-casting the analytic surface.

### Core Data Structures

- **`TSDFVolume`** (`src/tsdf.js`): 1D `Float32Array` for both `distances` (init 1.0) and `weights` (init 0.0), indexed as `ix + iy*size + iz*size²`. Algorithm follows Curless & Levoy (1996) weighted averaging. Has `.clone()`, `.reset()`, `.copyFrom()`, `.buildFrame()`, `.fuseVolume()`.
- **`SurfaceSampler`** (`src/sampling.js`): Legacy helper — generates random point clouds for sphere/torus. The worker now primarily uses ray-cast depth maps rather than `SurfaceSampler`.

### Key Constants (must stay in sync between main.js and worker.js)

```js
CAM_COUNT = 8
CAM_RADIUS = 2.5
CAM_HEIGHT = 1.0
CAM_FOV_DEG = 45
DEPTH_RES = 48   // exported from worker.js
```

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
- **Voxel tooltip**: Hover interaction on voxels shows TSDF value/weight.
- **Observation counter**: `observationProgressText` tracks current / total observations in the GUI.
- **Indentation**: 4 spaces (JS, CSS, HTML) — matches AGENTS.md and existing source files.
