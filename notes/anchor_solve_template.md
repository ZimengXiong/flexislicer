# Anchor Solve Template

Each anchor field is solved with a QuickCurve-style least-squares system.

## Unknowns

- Free grid heights in unconstrained regions.
- Per-component floating offsets for connected target regions.

## Objectives

### Boundary match

Tie free cells near target components to the component offset plus observed target height.

### Gradient shaping

In free regions, encourage directional height differences that achieve the desired target slope.

### Gauge / null-space control

Add a weak regularizer on component offsets or a reference variable to prevent drift.

### Post solve

Project the raw solution through a Lipschitz / conical validity pass to enforce nozzle reachability.

## Coupling rule

Outside the multi-depth mask:

- hard mode: overwrite `S_low = S_high`
- soft mode: penalize deviation `||S_low - S_high||^2`

Recommended default is hard coupling.
