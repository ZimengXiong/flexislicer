# Multi-Depth Failure Cases

## Why `K=1` breaks

A single heightfield can only target the topmost near-horizontal surface visible along each vertical ray. Any quality-critical ceiling or shelf beneath that top hit becomes invisible to the solver.

## Canonical examples

### Covered channel

- Top roof is captured by the top-hit map.
- Internal ceiling is missing from the target set.
- Result: good outer finish, poor internal ceiling alignment.

### Recessed shelf under a cap

- Shelf exists only below a higher overhang.
- `K=1` field follows the cap and cannot independently tune the shelf.

### Double-deck geometry

- Two shallow roofs occupy different heights in the same XY footprint.
- One heightfield can only align one of them.

### Cavity roof

- Internal roof strongly affects support-free quality.
- Top-hit extraction discards it completely.

## Working definition

`Multi-depth control` means being able to target more than one quality-critical, near-horizontal surface family along a single `(x,y)` column.
