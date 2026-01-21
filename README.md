# FlexSlicer / QuickCurve (Python Reference)

This repository contains a Python reference implementation of:

- *QuickCurve: revisiting slightly non-planar 3D printing* (arXiv:2406.03966)
- a FlexSlicer-style `K=2` extension with depth-varying field warping.

## What Is Implemented

- Vertical ray sampling of a watertight mesh onto a 2D grid.
- Construction of top-surface and secondary terrace targets from ray intersections.
- Optional morphological filtering on `Theta`.
- Least-squares solve for non-planar slice fields with:
  - free grid heights,
  - per-connected-component floating offsets `z_c`,
  - gradient steepening objective,
  - boundary matching objective.
- Post-process enforcing conical slope validity (Lipschitz/conical projection).
- `K=2` FlexField:
  - two anchors `S_low(x,y)` and `S_high(x,y)`,
  - smooth depth blending `D(x,y,z)` between anchors.
- Inverse-warp G-code with extrusion compensation based on warped 3D path length.
- Exported anisotropy field maps (`angle`, `strength`) and optional G-code heading steering
  for perimeter/infill paths.
- Curved slicing on rasterized solid intervals and contour extraction per layer.

This is a practical reference implementation, not a production slicer.

## Requirements

- Python 3.10+
- `uv` (recommended)

## Install

```bash
uv sync
```

## Run

```bash
uv run quickcurve \
  --mesh /path/to/model.stl \
  --out /path/to/output \
  --flex-k 2 \
  --terrace-gap-mm 0.6 \
  --flex-blend-start-frac 0.2 \
  --flex-blend-end-frac 0.85 \
  --grid-step 0.5 \
  --layer-height 0.2 \
  --theta-target 27 \
  --theta-max 40 \
  --filter-radius 0.5
```

### PrusaSlicer Hook (Deformed STL + Final G-code)

This runs the full pipeline you asked for:
1) export intermediate deformed STL,
2) call PrusaSlicer CLI,
3) warp planar G-code back to non-planar Z.

```bash
uv run quickcurve \
  --mesh /path/to/model.stl \
  --out /path/to/output \
  --deformed-stl /path/to/output/deformed_for_prusaslicer.stl \
  --gcode-out /path/to/output/final_nonplanar.gcode \
  --prusaslicer-cli /applications/prusaslicer \
  --prusaslicer-profile /path/to/prusaslicer_config.ini
```

Notes:
- `--flex-k 2` enables dual-anchor FlexField warping; use `--flex-k 1` for single-field behavior.
- `--terrace-gap-mm` controls how deep secondary terrace targets must be below the top surface.
- `--terrace-max-gap-mm` optionally rejects very deep secondary hits (useful to ignore
  opposite-shell bottom surfaces in thin coupons).
- `--flex-blend-start-frac` / `--flex-blend-end-frac` set depth blending range for `S_low`→`S_high`.
- If `/applications/prusaslicer` does not exist, the tool automatically falls back to:
  `/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer`
- `--prusaslicer-extra` can be repeated to pass extra CLI flags.
- By default the pipeline sets PrusaSlicer first layer to `0.2` mm. You can change it with
  `--prusaslicer-first-layer-height` or disable override by setting it to `<= 0`.
  If a model has no printable area at that first-layer height, the tool automatically retries with
  a small bottom flatten on the deformed STL to preserve the requested first-layer height.
  If that still fails, it retries without the override as a final fallback.
- Intermediate planar G-code from the deformed STL is stored at
  `<out>/planar_from_deformed.gcode` unless `--keep-prusaslicer-gcode` is specified.
- XY auto-alignment is enabled by default during warping, so PrusaSlicer recentering is compensated.
  You can disable it with `--no-warp-auto-align` or override manually using
  `--warp-shift-x` and `--warp-shift-y`.
- Z bed anchoring is enabled by default, so the warped G-code does not float above the bed.
  Disable with `--no-z-bed-anchor` only if you explicitly want raw absolute reconstruction.
- Bottom layer preservation and smooth transition are enabled by default:
  - `--preserve-planar-layers 1` keeps the first sliced layer planar.
  - `--warp-transition-layers 4` smoothly ramps into full non-planar warping.
  - The deformed STL keeps bottom geometry unchanged through preserved + transition layers,
    then blends into inverse deformation smoothly.
  - The warped first layer keeps the same Z as the planar G-code first layer.
- Optional anisotropy-aware heading steering can be enabled in warp:
  - `--anisotropy-steer`
  - `--steer-perimeter-strength` / `--steer-infill-strength`
  - `--steer-max-angle-deg` / `--steer-max-shift-mm`
  - `--steer-strength-floor` can force steering on flat models where geometric anisotropy
    strength would otherwise be near zero.

## Outputs

The output directory contains:

- `x_coords.npy`, `y_coords.npy`
- `top_z.npy`, `terrace_z.npy`
- `valid_mask.npy`, `terrace_mask.npy`
- `theta_mask.npy`, `labels.npy`
- `theta_mask_low.npy`, `theta_mask_high.npy`, `labels_low.npy`, `labels_high.npy`
- `raw_surface.npy`, `final_surface.npy`
- `raw_surface_low.npy`, `raw_surface_high.npy`
- `final_surface_low.npy`, `final_surface_high.npy`
- `anisotropy_angle.npy`, `anisotropy_strength.npy`
- `layers.json` (per-layer contour polylines in 3D)
- `metadata.json`
- `deformed_for_prusaslicer.stl` (if requested)
- `planar_from_deformed.gcode` (if `--gcode-out` is used)
- final warped non-planar G-code at `--gcode-out`

## Notes

- Input mesh should be watertight for reliable interior interval extraction.
- Runtime and memory scale with XY grid resolution and layer count.
- This implementation focuses on slicing-surface optimization and curved contour extraction; it does not include full production G-code planning.
