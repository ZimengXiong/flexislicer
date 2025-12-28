# Warp / Slice / Unwarp Plan

## Pipeline

1. Solve anchor fields on the original mesh sampling grid.
2. Define `D(x,y,z_s)` and verify monotonicity.
3. Invert `z = z_s + D(x,y,z_s)` per mesh vertex to create a deformed STL in slicing space.
4. Slice that STL with a mainstream slicer.
5. Map planar G-code moves back to real-space Z with the forward field.

## Non-negotiable condition

The mapping must remain invertible:

`1 + ∂D/∂z_s > 0`.

If this fails, the warp folds over and the pipeline is unusable.

## Practical notes

- Keep bottom layers planar to protect bed adhesion.
- Blend into non-planarity smoothly over a short transition interval.
- Auto-align XY when the slicer recenters the deformed STL.
- Preserve existing segmentation from the slicer rather than oversampling during unwarp.
