# Related Work Matrix

## Comparison axes

| System | Representation | Preprocessing | Solver | Main friction points | Practical upside |
| --- | --- | --- | --- | --- | --- |
| Planar adaptive layer height | Planar Z levels | None beyond standard slicing | Heuristic | Cannot curve interfaces within a layer | Mature slicer support |
| QuickCurve | Single heightfield `S(x,y)` | Top-hit ray map on XY grid | One least-squares solve + post clamp | Only one surface shape per `(x,y)` column | Fast, simple, reproducible |
| CurviSlicer | Volumetric deformation field | Tetrahedralization + field setup | Constrained volumetric optimization / QP | Tet robustness, solver tuning, collision assumptions | Rich volumetric control |
| FlexSlicer | Rank-2 depth-varying heightfield | Top-hit + second-hit target extraction | Two LS solves + depth blend | Terrace extraction quality, blend schedule | Multi-depth control without tetrahedralization |

## Working conclusion

- QuickCurve is the baseline for practicality.
- CurviSlicer is the baseline for volumetric expressiveness.
- FlexSlicer should sit between them: recover a second controllable surface family without inheriting full volumetric preprocessing cost.
