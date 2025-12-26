# Thickness-Aware Blend Schedule

For the rank-2 field,

`D(x, y, z_s) = (1 - w(z_s)) S_low + w(z_s) S_high`.

Thickness in slicing space scales approximately as

`τ = τ_s (1 + ∂D/∂z_s)`.

## Key derivative

`∂D/∂z_s = w'(z_s) (S_high - S_low)`.

For smoothstep blending, the peak derivative is approximately

`max |w'(z_s)| ≈ 1.5 / Δz`.

Let `B_max = max |S_high - S_low|`. Then a practical upper bound is

`max |∂D/∂z_s| ≈ 1.5 B_max / Δz`.

## Engineering bound

To keep thickness below `τ_max`,

`Δz ≥ (1.5 B_max) / (τ_max / τ_s - 1)`.

## Mitigations

- widen the blend interval `Δz`
- reduce base layer height `τ_s`
- reduce anchor separation
- clamp or smooth terrace targets
