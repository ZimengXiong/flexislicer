# QuickCurve (Python Reference)

This repository now contains a Python reference implementation of the core QuickCurve pipeline from:

- *QuickCurve: revisiting slightly non-planar 3D printing* (arXiv:2406.03966)

## What Is Implemented

- Vertical ray sampling of a watertight mesh onto a 2D grid.
- Construction of the target map `Theta` from top-surface slope threshold.
- Optional morphological filtering on `Theta`.
- Least-squares solve for a non-planar slice field with:
  - free grid heights,
  - per-connected-component floating offsets `z_c`,
  - gradient steepening objective,
  - boundary matching objective.
- Post-process enforcing conical slope validity (top-down style propagation).
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

## Outputs

The output directory contains:

- `x_coords.npy`, `y_coords.npy`
- `top_z.npy`
- `valid_mask.npy`, `theta_mask.npy`, `labels.npy`
- `raw_surface.npy`, `final_surface.npy`
- `layers.json` (per-layer contour polylines in 3D)
- `metadata.json`
- `deformed_for_prusaslicer.stl` (if requested)
- `planar_from_deformed.gcode` (if `--gcode-out` is used)
- final warped non-planar G-code at `--gcode-out`

## Notes

- Input mesh should be watertight for reliable interior interval extraction.
- Runtime and memory scale with XY grid resolution and layer count.
- This implementation focuses on slicing-surface optimization and curved contour extraction; it does not include full production G-code planning.
