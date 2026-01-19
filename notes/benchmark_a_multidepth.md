# Benchmark A: Synthetic Multi-Depth Geometry

## Purpose

Construct a clean capability test that separates `K=1` from `K=2` without printer noise.

## Geometry sketch

- smooth dome-like exterior roof
- internal ceiling under the dome
- cavity region sized so that second-hit rays exist only where the internal ceiling matters

## Expected outcome

- `K=1`: aligns the top roof and misses the internal ceiling
- `K=2`: aligns both with bounded thickness distortion

## Measurement

Use component-wise layer-alignment RMS error for the top surface and internal ceiling separately.
