# Low-Rank FlexField

We model real-space height as

`z = z_s + D(x, y, z_s)`.

To stay lighter than a full volumetric parameterization, we use a separable vertical displacement:

`D(x, y, z_s) ≈ Σ_k b_k(z_s) S_k(x, y)`.

## Rank choices

- `K=1`: recovers QuickCurve-style behavior.
- `K=2`: minimum extension that allows the deformation shape to vary with height.
- `K>2`: possible, but expands target extraction, coupling, and manufacturability bookkeeping.

## Proposed rank-2 form

Use two anchor fields:

- `S_low(x, y)` for lower terrace / internal ceiling targets.
- `S_high(x, y)` for the outer top surface.

Then define

`D(x, y, z_s) = (1 - w(z_s)) S_low(x, y) + w(z_s) S_high(x, y)`.

This is the smallest model that can encode two distinct surface families per column while preserving closed-form derivative checks.
