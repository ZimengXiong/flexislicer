# Target Extraction

## Input

- Watertight mesh.
- XY raster step `Δx = Δy`.
- Vertical ray cast per grid cell.

## Surface maps

### Top-hit map

- Sort all ray intersections by `z`.
- Use the highest hit as `T_high(x,y)`.
- This behaves like QuickCurve and drives the outer anchor.

### Second-hit map

- Take the next lower valid hit as `T_low(x,y)`.
- Keep only hits separated from the top surface by at least a configurable terrace gap.
- Interpret surviving samples as candidate internal ceilings / shelves.

## Cleanup

Second-hit data is noisy in exactly the cases we care about, so extraction must be filtered:

- Near-horizontal filtering: reject local slopes above the target threshold.
- Component filtering: remove tiny isolated fragments.
- Morphological close/open: stabilize cavity roofs and broken shelves.
- Optional maximum gap threshold: reject very deep hits likely belonging to opposite shells.

## Masks

- `V`: valid top-hit footprint.
- `T`: valid second-hit / terrace footprint.

Outside `T`, the low anchor should default to the high anchor to avoid unrelated global lifting.
