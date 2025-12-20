# Height-Only Blend Rationale

Use `w = w(z_s)` only.

## Why

If the blend also depends on `x` or `y`, then

`∇D = (1 - w)∇S_low + w∇S_high + (S_high - S_low)∇w`

and the extra `∇w` term breaks the clean convex-combination argument used for slope bounds.

## Consequence

Height-only blending is not just a convenience. It is the key reason the rank-2 representation still admits simple manufacturability reasoning.
