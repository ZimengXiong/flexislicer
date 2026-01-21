from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
import trimesh
from trimesh.creation import box, cylinder

from quickcurve.solver import QuickCurveConfig, run_quickcurve


def _smoothstep(t: float) -> float:
    tc = float(np.clip(t, 0.0, 1.0))
    return tc * tc * (3.0 - 2.0 * tc)


def _boolean_difference(base: trimesh.Trimesh, subtractors: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    mesh = trimesh.boolean.difference([base, *subtractors])
    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Boolean difference failed to create a mesh.")
    # Keep cleanup compatible across trimesh versions.
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def make_coupon_mesh(
    length_mm: float,
    width_mm: float,
    thickness_mm: float,
    hole_diameter_mm: float,
    recesses: list[dict[str, float]],
) -> trimesh.Trimesh:
    plate = box(extents=(length_mm, width_mm, thickness_mm))
    plate.apply_translation([0.0, 0.0, thickness_mm / 2.0])

    subtractors: list[trimesh.Trimesh] = []
    hole = cylinder(radius=hole_diameter_mm * 0.5, height=thickness_mm + 1.0, sections=128)
    hole.apply_translation([0.0, 0.0, thickness_mm * 0.5])
    subtractors.append(hole)

    for rec in recesses:
        radius = float(rec["radius"])
        depth = float(rec["depth"])
        x = float(rec.get("x", 0.0))
        y = float(rec.get("y", 0.0))
        if depth <= 0.0:
            continue
        h = depth + 0.4
        zc = thickness_mm - depth * 0.5 + 0.2
        pocket = cylinder(radius=radius, height=h, sections=128)
        pocket.apply_translation([x, y, zc])
        subtractors.append(pocket)

    mesh = _boolean_difference(plate, subtractors)
    zmin = float(np.min(mesh.vertices[:, 2]))
    mesh.apply_translation([0.0, 0.0, -zmin])
    return mesh


def build_specimens(spec_dir: Path) -> dict[str, Path]:
    spec_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    specs: dict[str, dict[str, object]] = {
        "flat_open_hole": {
            "length": 120.0,
            "width": 30.0,
            "thickness": 4.0,
            "hole_diameter": 6.0,
            "recesses": [],
        },
        "terraced_ring_light": {
            "length": 120.0,
            "width": 30.0,
            "thickness": 4.0,
            "hole_diameter": 6.0,
            "recesses": [
                {"radius": 14.0, "depth": 0.6},
                {"radius": 8.0, "depth": 1.4},
            ],
        },
        "terraced_ring_strong": {
            "length": 120.0,
            "width": 30.0,
            "thickness": 4.0,
            "hole_diameter": 6.0,
            "recesses": [
                {"radius": 16.0, "depth": 0.9},
                {"radius": 10.0, "depth": 2.0},
            ],
        },
        "dual_terrace_offset": {
            "length": 120.0,
            "width": 30.0,
            "thickness": 4.0,
            "hole_diameter": 6.0,
            "recesses": [
                {"radius": 13.0, "depth": 0.8},
                {"radius": 7.5, "depth": 1.8},
                {"radius": 5.0, "depth": 1.2, "x": 22.0, "y": 0.0},
            ],
        },
    }

    for name, cfg in specs.items():
        mesh = make_coupon_mesh(
            length_mm=float(cfg["length"]),
            width_mm=float(cfg["width"]),
            thickness_mm=float(cfg["thickness"]),
            hole_diameter_mm=float(cfg["hole_diameter"]),
            recesses=list(cfg["recesses"]),  # type: ignore[arg-type]
        )
        p = spec_dir / f"{name}.stl"
        mesh.export(p)
        out[name] = p

    # Multi-roof canopy specimen creates true shallow multi-hit rays,
    # which is where K=2 should help most.
    parts: list[trimesh.Trimesh] = []
    base = box(extents=(100.0, 30.0, 4.0))
    base.apply_translation([0.0, 0.0, 2.0])
    parts.append(base)
    for x in (-35.0, 35.0):
        for y in (-10.0, 10.0):
            col = box(extents=(6.0, 6.0, 8.0))
            col.apply_translation([x, y, 8.0])
            parts.append(col)
    roof_hi = box(extents=(80.0, 24.0, 0.8))
    roof_hi.apply_translation([0.0, 0.0, 12.4])
    parts.append(roof_hi)
    roof_lo = box(extents=(42.0, 16.0, 0.8))
    roof_lo.apply_translation([0.0, 0.0, 9.8])
    parts.append(roof_lo)

    canopy = trimesh.boolean.union(parts)
    if isinstance(canopy, list):
        canopy = trimesh.util.concatenate(canopy)
    if not isinstance(canopy, trimesh.Trimesh):
        raise ValueError("Failed to build canopy specimen mesh.")
    canopy = trimesh.Trimesh(vertices=canopy.vertices.copy(), faces=canopy.faces.copy(), process=True)
    canopy.remove_unreferenced_vertices()
    zmin = float(np.min(canopy.vertices[:, 2]))
    canopy.apply_translation([0.0, 0.0, -zmin])
    p_canopy = spec_dir / "canopy_terrace.stl"
    canopy.export(p_canopy)
    out["canopy_terrace"] = p_canopy

    return out


def _build_displacement_volume(result, cfg: QuickCurveConfig, z_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    valid = result.valid_mask
    low = result.final_surface_low
    high = result.final_surface_high

    stats = result.stats
    z_min = float(stats["mesh_z_min"])
    z_max = float(stats["mesh_z_max"])
    blend_start = float(stats["blend_z_start"])
    blend_end = float(stats["blend_z_end"])
    flex_k = int(stats["flex_k"])

    finite_low = np.abs(low[np.isfinite(low)])
    finite_high = np.abs(high[np.isfinite(high)])
    if finite_low.size == 0 and finite_high.size == 0:
        field_p95 = 0.0
    else:
        pooled = np.concatenate([finite_low, finite_high])
        field_p95 = float(np.percentile(pooled, 95))

    preserve_h = cfg.layer_height
    transition_h = 4.0 * cfg.layer_height
    transition_h_eff = max(transition_h, 2.0 * field_p95) if transition_h > 0.0 else 0.0
    z_guard = z_min + preserve_h

    zs = np.linspace(z_min, z_max, z_samples, dtype=float)
    vol = np.full((z_samples, *valid.shape), np.nan, dtype=float)

    for i, z in enumerate(zs):
        if flex_k < 2:
            b = 1.0
        elif blend_end <= blend_start:
            b = 1.0 if z >= blend_start else 0.0
        else:
            b = _smoothstep((z - blend_start) / (blend_end - blend_start))

        base = (1.0 - b) * low + b * high
        if z <= z_guard:
            gw = 0.0
        elif transition_h_eff <= 0.0:
            gw = 1.0
        else:
            gw = _smoothstep((z - z_guard) / transition_h_eff)

        d = gw * base
        vol[i] = np.where(valid, d, np.nan)

    return zs, vol


def _topology_metrics(volume: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    v = volume[np.isfinite(volume)]
    if v.size == 0:
        return {
            "disp_mean_abs": 0.0,
            "disp_p95_abs": 0.0,
            "disp_max_abs": 0.0,
            "unique_signatures": 0.0,
            "change_events": 0.0,
            "total_components": 0.0,
            "nonzero_frac_abs_gt_0p05": 0.0,
        }

    max_abs = float(np.max(np.abs(v)))
    thresholds = [0.2 * max_abs, 0.4 * max_abs, 0.6 * max_abs] if max_abs > 1e-9 else [0.0]

    structure = np.ones((3, 3), dtype=bool)
    signatures: list[tuple[int, ...]] = []
    total_components = 0
    for i in range(volume.shape[0]):
        d = volume[i]
        sig: list[int] = []
        for t in thresholds:
            m = valid_mask & np.isfinite(d) & (d >= t)
            n = int(ndi.label(m, structure=structure)[1]) if np.any(m) else 0
            sig.append(n)
            total_components += n
        signatures.append(tuple(sig))

    change_events = sum(1 for i in range(1, len(signatures)) if signatures[i] != signatures[i - 1])
    nonzero = np.abs(v) > 0.05

    return {
        "disp_mean_abs": float(np.mean(np.abs(v))),
        "disp_p95_abs": float(np.percentile(np.abs(v), 95)),
        "disp_max_abs": max_abs,
        "unique_signatures": float(len(set(signatures))),
        "change_events": float(change_events),
        "total_components": float(total_components),
        "nonzero_frac_abs_gt_0p05": float(np.mean(nonzero)),
    }


def _depth_pattern_metrics(volume: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    if volume.shape[0] < 4:
        return {"depth_corr": 1.0, "depth_affine_residual": 0.0}

    i0 = int(round(0.25 * (volume.shape[0] - 1)))
    i1 = int(round(0.75 * (volume.shape[0] - 1)))
    a = volume[i0]
    b = volume[i1]
    m = valid_mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return {"depth_corr": 1.0, "depth_affine_residual": 0.0}

    va = a[m].astype(float)
    vb = b[m].astype(float)
    va0 = va - np.mean(va)
    vb0 = vb - np.mean(vb)
    na = float(np.linalg.norm(va0))
    nb = float(np.linalg.norm(vb0))
    if na < 1e-12 or nb < 1e-12:
        corr = 1.0
    else:
        corr = float(np.clip(np.dot(va0, vb0) / (na * nb), -1.0, 1.0))

    denom = float(np.dot(va, va))
    alpha = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
    resid = vb - alpha * va
    resid_rel = float(np.linalg.norm(resid) / (np.linalg.norm(vb) + 1e-12))
    return {"depth_corr": corr, "depth_affine_residual": resid_rel}


def _run_case(mesh_path: Path, cfg: QuickCurveConfig) -> dict[str, float]:
    t0 = perf_counter()
    result = run_quickcurve(mesh_path, cfg)
    dt = perf_counter() - t0

    _, vol = _build_displacement_volume(result, cfg)
    topo = _topology_metrics(vol, result.valid_mask)
    depth = _depth_pattern_metrics(vol, result.valid_mask)

    delta = np.abs(result.final_surface_high - result.final_surface_low)
    delta = delta[np.isfinite(delta) & result.valid_mask]

    return {
        "runtime_s": float(dt),
        "flex_k_used": float(result.stats.get("flex_k", 1)),
        "num_valid": float(result.stats["num_valid"]),
        "num_terrace": float(result.stats.get("num_terrace", 0)),
        "terrace_frac": float(result.stats.get("num_terrace", 0) / max(1, result.stats["num_valid"])),
        "low_high_delta_mean": float(np.mean(delta)) if delta.size else 0.0,
        "low_high_delta_p95": float(np.percentile(delta, 95)) if delta.size else 0.0,
        "low_high_delta_gt_0p05_frac": float(np.mean(delta > 0.05)) if delta.size else 0.0,
        **topo,
        **depth,
    }


def run_campaign(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = out_dir / "specimens"
    specimens = build_specimens(spec_dir)

    base = QuickCurveConfig(
        grid_step=0.5,
        layer_height=0.2,
        theta_target_deg=25.0,
        theta_max_deg=40.0,
        filter_radius_mm=0.0,
        max_post_iters=200,
        w_gradient=1.0,
        w_boundary=3.0,
        w_smooth=0.05,
        w_component_reg=1e-3,
        terrace_min_gap_mm=0.6,
        terrace_max_gap_mm=2.5,
        flex_blend_start_frac=0.2,
        flex_blend_end_frac=0.85,
    )

    models: dict[str, dict[str, dict[str, float]]] = {}
    ratios: dict[str, dict[str, float | None]] = {}

    for name, mesh in specimens.items():
        k1 = _run_case(mesh, replace(base, flex_k=1))
        k2 = _run_case(mesh, replace(base, flex_k=2))
        models[name] = {"k1": k1, "k2": k2}

        ratio: dict[str, float | None] = {}
        for key in (
            "unique_signatures",
            "change_events",
            "total_components",
            "low_high_delta_gt_0p05_frac",
            "disp_mean_abs",
            "depth_affine_residual",
            "runtime_s",
        ):
            a = k1[key]
            b = k2[key]
            ratio[f"{key}_k2_over_k1"] = float(b / a) if abs(a) > 1e-12 else None
        ratios[name] = ratio

    payload: dict[str, object] = {
        "config": {
            "grid_step": base.grid_step,
            "layer_height": base.layer_height,
            "theta_target_deg": base.theta_target_deg,
            "theta_max_deg": base.theta_max_deg,
            "terrace_min_gap_mm": base.terrace_min_gap_mm,
            "terrace_max_gap_mm": base.terrace_max_gap_mm,
            "blend_start_frac": base.flex_blend_start_frac,
            "blend_end_frac": base.flex_blend_end_frac,
        },
        "models": models,
        "ratios": ratios,
    }

    (out_dir / "k2_campaign.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_markdown_report(payload: dict[str, object], out_path: Path) -> None:
    ratios = payload["ratios"]  # type: ignore[assignment]
    models = payload["models"]  # type: ignore[assignment]

    lines = [
        "# K2 Campaign Report",
        "",
        "Metrics compare `flex_k=1` and `flex_k=2` on terrace-focused coupon specimens.",
        "",
        "## K2/K1 Ratios",
        "",
        "| Specimen | Unique Signatures | Change Events | Total Components | Depth Residual | Runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    sorted_names = sorted(
        models.keys(),  # type: ignore[attr-defined]
        key=lambda n: (models[n]["k2"]["depth_affine_residual"] - models[n]["k1"]["depth_affine_residual"]),  # type: ignore[index]
        reverse=True,
    )
    for name in sorted_names:
        r = ratios[name]  # type: ignore[index]
        us = r["unique_signatures_k2_over_k1"]  # type: ignore[index]
        ce = r["change_events_k2_over_k1"]  # type: ignore[index]
        tc = r["total_components_k2_over_k1"]  # type: ignore[index]
        dr = r["depth_affine_residual_k2_over_k1"]  # type: ignore[index]
        rt = r["runtime_s_k2_over_k1"]  # type: ignore[index]
        lines.append(
            f"| {name} | {us if us is not None else 'n/a'} | {ce if ce is not None else 'n/a'} | "
            f"{tc if tc is not None else 'n/a'} | {dr if dr is not None else 'n/a'} | "
            f"{rt if rt is not None else 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Depth Pattern Separation",
            "",
            "| Specimen | K1 Residual | K2 Residual | K2-K1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in sorted_names:
        k1 = models[name]["k1"]  # type: ignore[index]
        k2 = models[name]["k2"]  # type: ignore[index]
        r1 = k1["depth_affine_residual"]  # type: ignore[index]
        r2 = k2["depth_affine_residual"]  # type: ignore[index]
        lines.append(f"| {name} | {r1} | {r2} | {r2 - r1} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `flat_open_hole` is a control where K2 should not help much.",
            "- `Depth Residual` is the key K2 signal here: it measures how different",
            "  the displacement pattern becomes between low and high slicing depths.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run K1 vs K2 campaign on terrace-focused coupons.")
    p.add_argument("--out", default="out_k2_campaign", help="Output directory.")
    args = p.parse_args()

    out_dir = Path(args.out)
    payload = run_campaign(out_dir)
    _write_markdown_report(payload, out_dir / "k2_campaign.md")

    top = sorted(
        payload["models"].items(),  # type: ignore[union-attr]
        key=lambda kv: (
            kv[1]["k2"]["depth_affine_residual"] - kv[1]["k1"]["depth_affine_residual"]
        ),
        reverse=True,
    )
    print("K2 campaign done.")
    for name, model in top:
        delta_res = (
            model["k2"]["depth_affine_residual"] - model["k1"]["depth_affine_residual"]
        )
        print(
            f"{name}: depth_residual_delta={delta_res}, "
            f"k2_used={model['k2']['flex_k_used']}, "
            f"runtime_k2={model['k2']['runtime_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
