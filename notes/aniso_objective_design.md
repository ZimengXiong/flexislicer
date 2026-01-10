# Anisotropy Objective Design

## Goal

Bias layer-interface orientation toward a preferred in-plane stress direction while keeping the optimizer in least-squares form.

## Simplified proxy

Instead of solving directly on a full stress tensor field, use a directional penalty on XY gradients aligned with a prescribed stress direction map.

## LS-friendly form

For grid edges aligned with local stress direction `d(x,y)`, weight the gradient shaping term more strongly. This steers the anchor fields toward interface normals that better resist delamination under the target loading direction.

## Why this matters

- keeps the solver quadratic / linear least squares
- fits the platform story: geometry alignment first, optional objective modules second
- avoids introducing a separate nonlinear optimization layer
