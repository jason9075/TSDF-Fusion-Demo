# Repository Guidelines

## Project Structure & Module Organization
This repository is a zero-build browser demo for TSDF fusion and mesh extraction. The app entry point is [`index.html`](/home/jason9075/data/tsdf_demo/index.html), with shared styling in [`style.css`](/home/jason9075/data/tsdf_demo/style.css). Core logic lives under [`src/`](/home/jason9075/data/tsdf_demo/src): `main.js` manages the Three.js scene and GUI, `worker.js` handles CPU-heavy TSDF work in a Web Worker, and `tsdf.js` plus `sampling.js` contain the volume and sampling utilities. Nix shell setup is in `flake.nix`; developer shortcuts live in `Justfile`.

## Build, Test, and Development Commands
Use `nix develop` to enter the pinned dev shell with `just`, `entr`, and `python3`.

- `just dev` - watches `*.html`, `*.js`, and `*.css`, then serves the repo at `http://localhost:8080`.
- `python3 -m http.server 8080` - simple fallback if you do not need file watching.
- `just` - lists available recipes.

There is no bundler or production build step; modules are loaded directly in the browser through the import map in `index.html`.

## Coding Style & Naming Conventions
Follow the existing style: 4-space indentation in JS, CSS, and HTML; semicolons enabled; ES modules with named imports where practical. Use `camelCase` for variables and functions, `UPPER_SNAKE_CASE` for shared constants such as `DEPTH_RES`, and descriptive file names under `src/`. Keep rendering/UI work on the main thread and move heavy numerical work into `worker.js`. Match the current Nord-inspired UI palette when touching CSS.

## Testing Guidelines
There is no automated test suite yet. Validate changes by running `just dev` and exercising the full demo flow in the browser: point generation, voxelization, fusion, mesh extraction, slice controls, and modal toggles. When changing worker message types or shared constants, verify both ends stay in sync. If you add tests, place them in a top-level `tests/` directory and use names like `feature-name.test.js`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative messages, often Conventional Commit-style, for example `feat: add fake gsplat for demo` and `ui: fix camera position`. Prefer that format: `feat: ...`, `fix: ...`, `ui: ...`, `hotfix: ...`. PRs should explain the user-visible change, list manual verification steps, and include screenshots or a short screen recording for UI changes.
