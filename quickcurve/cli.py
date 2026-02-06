from __future__ import annotations

import argparse
from pathlib import Path

from quickcurve.prusa import (
    export_deformed_stl,
    flatten_stl_bottom,
    resolve_prusaslicer_cli,
    run_prusaslicer_export_gcode,
    warp_gcode_with_surface,
)
from quickcurve.solver import QuickCurveConfig, run_quickcurve, save_result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FlexSlicer/QuickCurve reference implementation (Python)."
    )
    p.add_argument("--mesh", required=True, help="Input watertight mesh (STL/OBJ/PLY/...).")
    p.add_argument("--out", required=True, help="Output directory.")

    p.add_argument("--grid-step", type=float, default=0.5, help="XY raster step in mm.")
    p.add_argument("--layer-height", type=float, default=0.2, help="Layer height in mm.")
    p.add_argument(
        "--theta-target",
        type=float,
        default=27.0,
        help="Target slope used for gradient steepening (degrees).",
    )
    p.add_argument(
        "--theta-max",
        type=float,
        default=40.0,
        help="Maximum conical slope for post-process validity (degrees).",
    )
    p.add_argument(
        "--filter-radius",
        type=float,
        default=0.0,
        help="Optional morphological closure radius for Theta mask (mm).",
    )
    p.add_argument(
        "--max-post-iters",
        type=int,
        default=200,
        help="Maximum post-process iterations.",
    )
    p.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional cap on number of generated layers.",
    )
    p.add_argument(
        "--flex-k",
        type=int,
        default=2,
        choices=[1, 2],
        help="FlexField anchor count: 1=QuickCurve-style single field, 2=FlexSlicer dual-anchor field.",
    )
    p.add_argument(
        "--terrace-gap-mm",
        type=float,
        default=0.6,
        help="Minimum vertical gap from top surface to accept a secondary terrace target (mm).",
    )
    p.add_argument(
        "--flex-blend-start-frac",
        type=float,
        default=0.2,
        help="Start of depth blending as a fraction of model height [0..1].",
    )
    p.add_argument(
        "--flex-blend-end-frac",
        type=float,
        default=0.85,
        help="End of depth blending as a fraction of model height [0..1].",
    )

    p.add_argument("--w-gradient", type=float, default=1.0, help="Gradient objective weight.")
    p.add_argument("--w-boundary", type=float, default=3.0, help="Boundary objective weight.")
    p.add_argument("--w-smooth", type=float, default=0.05, help="Smoothness objective weight.")
    p.add_argument(
        "--w-component-reg",
        type=float,
        default=1e-3,
        help="Weak component offset regularization weight.",
    )
    p.add_argument(
        "--deformed-stl",
        default=None,
        help="Path to write intermediate deformed STL (default: <out>/deformed_for_prusaslicer.stl).",
    )
    p.add_argument(
        "--gcode-out",
        default=None,
        help="Path for final non-planar warped G-code. Enables PrusaSlicer pipeline.",
    )
    p.add_argument(
        "--prusaslicer-cli",
        default="/applications/prusaslicer",
        help="PrusaSlicer CLI path (default: /applications/prusaslicer; app bundle fallback is automatic).",
    )
    p.add_argument(
        "--prusaslicer-profile",
        default=None,
        help="Optional PrusaSlicer config/profile file passed via --load.",
    )
    p.add_argument(
        "--prusaslicer-first-layer-height",
        type=float,
        default=0.2,
        help="PrusaSlicer first layer height in mm (default: 0.2). Set <=0 to leave unchanged.",
    )
    p.add_argument(
        "--prusaslicer-extra",
        action="append",
        default=[],
        help="Extra argument to pass to PrusaSlicer (repeatable).",
    )
    p.add_argument(
        "--keep-prusaslicer-gcode",
        default=None,
        help="Optional path to keep the intermediate planar G-code from the deformed STL.",
    )
    p.add_argument(
        "--warp-shift-x",
        type=float,
        default=None,
        help="Optional manual X shift applied to G-code XY before sampling the surface.",
    )
    p.add_argument(
        "--warp-shift-y",
        type=float,
        default=None,
        help="Optional manual Y shift applied to G-code XY before sampling the surface.",
    )
    p.add_argument(
        "--no-warp-auto-align",
        action="store_true",
        help="Disable automatic XY alignment between gcode coordinates and surface grid.",
    )
    p.add_argument(
        "--no-z-bed-anchor",
        action="store_true",
        help="Disable automatic Z anchor removal (may cause floating first layer).",
    )
    p.add_argument(
        "--preserve-planar-layers",
        type=int,
        default=1,
        help="Keep the first N sliced layers fully planar before non-planar warping.",
    )
    p.add_argument(
        "--warp-transition-layers",
        type=int,
        default=4,
        help="Blend from planar to full non-planar over this many layers after preserved layers.",
    )
    p.add_argument(
        "--anisotropy-steer",
        action="store_true",
        help="Steer perimeter/infill segment headings toward the anisotropy field.",
    )
    p.add_argument(
        "--steer-perimeter-strength",
        type=float,
        default=0.35,
        help="Steering gain for perimeter-like paths (>=0).",
    )
    p.add_argument(
        "--steer-infill-strength",
        type=float,
        default=0.65,
        help="Steering gain for infill-like paths (>=0).",
    )
    p.add_argument(
        "--steer-max-angle-deg",
        type=float,
        default=18.0,
        help="Maximum heading correction per move in degrees.",
    )
    p.add_argument(
        "--steer-max-shift-mm",
        type=float,
        default=0.12,
        help="Maximum XY endpoint shift per move from steering.",
    )
    p.add_argument(
        "--steer-strength-floor",
        type=float,
        default=0.0,
        help="Minimum normalized anisotropy strength [0..1] used for steering.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    if (args.warp_shift_x is None) != (args.warp_shift_y is None):
        raise ValueError("Provide both --warp-shift-x and --warp-shift-y, or neither.")
    if args.preserve_planar_layers < 0:
        raise ValueError("--preserve-planar-layers must be >= 0")
    if args.warp_transition_layers < 0:
        raise ValueError("--warp-transition-layers must be >= 0")
    if args.terrace_gap_mm < 0:
        raise ValueError("--terrace-gap-mm must be >= 0")
    if args.steer_perimeter_strength < 0:
        raise ValueError("--steer-perimeter-strength must be >= 0")
    if args.steer_infill_strength < 0:
        raise ValueError("--steer-infill-strength must be >= 0")
    if args.steer_max_angle_deg < 0:
        raise ValueError("--steer-max-angle-deg must be >= 0")
    if args.steer_max_shift_mm < 0:
        raise ValueError("--steer-max-shift-mm must be >= 0")
    if args.steer_strength_floor < 0 or args.steer_strength_floor > 1:
        raise ValueError("--steer-strength-floor must be within [0, 1]")

    cfg = QuickCurveConfig(
        grid_step=args.grid_step,
        layer_height=args.layer_height,
        theta_target_deg=args.theta_target,
        theta_max_deg=args.theta_max,
        filter_radius_mm=args.filter_radius,
        max_post_iters=args.max_post_iters,
        max_layers=args.max_layers,
        w_gradient=args.w_gradient,
        w_boundary=args.w_boundary,
        w_smooth=args.w_smooth,
        w_component_reg=args.w_component_reg,
        flex_k=args.flex_k,
        terrace_min_gap_mm=args.terrace_gap_mm,
        flex_blend_start_frac=args.flex_blend_start_frac,
        flex_blend_end_frac=args.flex_blend_end_frac,
    )

    result = run_quickcurve(Path(args.mesh), cfg)
    out_dir = Path(args.out)
    save_result(result, cfg, out_dir)

    deformed_stl_path: Path | None = None
    if args.deformed_stl is not None or args.gcode_out is not None:
        deformed_stl_path = Path(args.deformed_stl) if args.deformed_stl else out_dir / "deformed_for_prusaslicer.stl"
        export_deformed_stl(
            input_mesh_path=Path(args.mesh),
            result=result,
            out_stl_path=deformed_stl_path,
            layer_height=cfg.layer_height,
            preserve_planar_layers=args.preserve_planar_layers,
            transition_layers=args.warp_transition_layers,
        )

    warped_gcode_path: Path | None = None
    if args.gcode_out is not None:
        cli_path = resolve_prusaslicer_cli(args.prusaslicer_cli)
        planar_gcode = (
            Path(args.keep_prusaslicer_gcode)
            if args.keep_prusaslicer_gcode
            else out_dir / "planar_from_deformed.gcode"
        )
        base_extra_args = list(args.prusaslicer_extra)
        inject_first_layer = (
            args.prusaslicer_first_layer_height is not None
            and args.prusaslicer_first_layer_height > 0.0
            and not any(str(a).startswith("--first-layer-height") for a in args.prusaslicer_extra)
        )
        full_extra_args = (
            base_extra_args + [f"--first-layer-height={args.prusaslicer_first_layer_height:.5f}"]
            if inject_first_layer
            else base_extra_args
        )
        try:
            run_prusaslicer_export_gcode(
                prusaslicer_cli=cli_path,
                input_stl=deformed_stl_path,
                output_gcode=planar_gcode,
                profile_path=args.prusaslicer_profile,
                extra_args=full_extra_args,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if inject_first_layer and "no extrusions in the first layer" in msg:
                flh = float(args.prusaslicer_first_layer_height)
                flatten_depths = [flh, 1.25 * flh, 1.5 * flh, 2.0 * flh, 3.0 * flh]
                sliced = False
                for depth in flatten_depths:
                    flat_stl = out_dir / f"deformed_for_prusaslicer_flat_{depth:.3f}mm.stl"
                    flatten_stl_bottom(
                        input_stl=deformed_stl_path,
                        output_stl=flat_stl,
                        flatten_depth_mm=depth,
                    )
                    try:
                        run_prusaslicer_export_gcode(
                            prusaslicer_cli=cli_path,
                            input_stl=flat_stl,
                            output_gcode=planar_gcode,
                            profile_path=args.prusaslicer_profile,
                            extra_args=full_extra_args,
                        )
                        print(
                            "Warning: auto-flattened deformed STL bottom by "
                            f"{depth:.3f} mm to ensure first-layer-height {flh:.3f} has printable area."
                        )
                        sliced = True
                        break
                    except RuntimeError as e_flat:
                        msg_flat = str(e_flat).lower()
                        if "no extrusions in the first layer" not in msg_flat:
                            raise

                if not sliced:
                    print(
                        "Warning: Could not enforce first-layer-height "
                        f"{args.prusaslicer_first_layer_height:.3f} on this geometry; retrying without override."
                    )
                    run_prusaslicer_export_gcode(
                        prusaslicer_cli=cli_path,
                        input_stl=deformed_stl_path,
                        output_gcode=planar_gcode,
                        profile_path=args.prusaslicer_profile,
                        extra_args=base_extra_args,
                    )
            else:
                raise
        warped_gcode_path = warp_gcode_with_surface(
            input_gcode=planar_gcode,
            output_gcode=Path(args.gcode_out),
            result=result,
            xy_shift=(
                (args.warp_shift_x, args.warp_shift_y)
                if args.warp_shift_x is not None and args.warp_shift_y is not None
                else None
            ),
            auto_align_xy=not args.no_warp_auto_align,
            z_anchor_to_bed=not args.no_z_bed_anchor,
            preserve_planar_layers=args.preserve_planar_layers,
            transition_layers=args.warp_transition_layers,
            layer_height=cfg.layer_height,
            anisotropy_steer=args.anisotropy_steer,
            steer_perimeter_strength=args.steer_perimeter_strength,
            steer_infill_strength=args.steer_infill_strength,
            steer_max_angle_deg=args.steer_max_angle_deg,
            steer_max_shift_mm=args.steer_max_shift_mm,
            steer_strength_floor=args.steer_strength_floor,
        )

    print("QuickCurve run completed.")
    print(f"Output directory: {args.out}")
    print(f"Grid: {result.stats['grid_shape']} @ {cfg.grid_step} mm")
    print(f"Valid samples: {result.stats['num_valid']}")
    print(f"Theta samples: {result.stats['num_theta']}")
    print(f"Connected components: {result.stats['num_components']}")
    print(f"FlexField K: {result.stats.get('flex_k', 1)}")
    print(f"Terrace samples: {result.stats.get('num_terrace', 0)}")
    print(f"Layers with contours: {result.stats['num_layers_with_paths']}")
    if deformed_stl_path is not None:
        print(f"Deformed STL: {deformed_stl_path}")
    if warped_gcode_path is not None:
        print(f"Warped G-code: {warped_gcode_path}")


if __name__ == "__main__":
    main()
