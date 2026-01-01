# Extrusion Compensation

After inverse warping, a planar move and its real-space counterpart may have different 3D lengths.

Use a local correction:

`E_real = E_slice * ||Δp_real|| / ||Δp_slice||`.

## Why this default

- cheap to compute per move
- compatible with existing G-code postprocessing
- improves flow consistency without requiring a full bead-volume model

## Deferred extensions

- include layer-thickness scaling from `1 + ∂D/∂z_s`
- account for bead cross-section changes under steep slope
- couple correction with anisotropy steering when XY endpoints are adjusted
