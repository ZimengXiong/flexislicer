from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Iterable

import numpy as np
import trimesh

from quickcurve.solver import QuickCurveResult


DEFAULT_PRUSASLICER_PATHS = [
    "/applications/prusaslicer",
    "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
]


@dataclass
class SurfaceSampler:
    surface: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray

    def __post_init__(self) -> None:
        self._filled = self._fill_nans_nearest(self.surface)
        self._x0 = float(self.x_coords[0])
        self._y0 = float(self.y_coords[0])
        self._sx = float(self.x_coords[1] - self.x_coords[0]) if self.x_coords.size > 1 else 1.0
        self._sy = float(self.y_coords[1] - self.y_coords[0]) if self.y_coords.size > 1 else 1.0

    @staticmethod
    def _fill_nans_nearest(arr: np.ndarray) -> np.ndarray:
        valid = np.isfinite(arr)
        if np.all(valid):
            return arr.copy()
        if not np.any(valid):
            return np.zeros_like(arr)
        from scipy import ndimage as ndi

        _, indices = ndi.distance_transform_edt(~valid, return_indices=True)
        return arr[tuple(indices)]

    @staticmethod
    def _bilinear(arr: np.ndarray, row: float, col: float) -> float:
        h, w = arr.shape
        r0 = int(np.clip(np.floor(row), 0, h - 1))
        r1 = int(np.clip(r0 + 1, 0, h - 1))
        c0 = int(np.clip(np.floor(col), 0, w - 1))
        c1 = int(np.clip(c0 + 1, 0, w - 1))

        fr = float(np.clip(row - r0, 0.0, 1.0))
        fc = float(np.clip(col - c0, 0.0, 1.0))

        v00 = arr[r0, c0]
        v01 = arr[r0, c1]
        v10 = arr[r1, c0]
        v11 = arr[r1, c1]

        top = v00 * (1.0 - fc) + v01 * fc
        bot = v10 * (1.0 - fc) + v11 * fc
        return float(top * (1.0 - fr) + bot * fr)

    def sample(self, x: float, y: float) -> float:
        col = (x - self._x0) / self._sx
        row = (y - self._y0) / self._sy
        return self._bilinear(self._filled, row, col)


def resolve_prusaslicer_cli(preferred: str | None = None) -> Path:
    candidates = [preferred] if preferred else []
    candidates.extend(DEFAULT_PRUSASLICER_PATHS)

    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.exists() and p.is_file():
            return p

    joined = "\n  - ".join(str(c) for c in candidates if c)
    raise FileNotFoundError(
        "Unable to find PrusaSlicer CLI. Tried:\n  - " + joined
    )


def export_deformed_stl(
    input_mesh_path: str | Path,
    result: QuickCurveResult,
    out_stl_path: str | Path,
    layer_height: float = 0.2,
    preserve_planar_layers: int = 1,
    transition_layers: int = 4,
) -> Path:
    if layer_height <= 0.0:
        raise ValueError("layer_height must be > 0")
    if preserve_planar_layers < 0:
        raise ValueError("preserve_planar_layers must be >= 0")
    if transition_layers < 0:
        raise ValueError("transition_layers must be >= 0")

    loaded = trimesh.load(input_mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("Input mesh has no geometry")
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError("Input mesh is not a valid Trimesh")

    sampler = SurfaceSampler(
        surface=result.final_surface,
        x_coords=result.x_coords,
        y_coords=result.y_coords,
    )

    verts = mesh.vertices.copy()
    offsets = np.array([sampler.sample(v[0], v[1]) for v in verts], dtype=float)
    z_min = float(np.min(verts[:, 2]))
    rel_z = verts[:, 2] - z_min
    preserve_h = float(preserve_planar_layers) * float(layer_height)
    transition_h = float(transition_layers) * float(layer_height)
    lock_h = preserve_h + transition_h
    z_guard = z_min + lock_h

    if transition_h <= 0.0:
        deform_w = np.where(verts[:, 2] <= z_guard, 0.0, 1.0)
    else:
        t = np.clip((rel_z - lock_h) / transition_h, 0.0, 1.0)
        deform_w = t * t * (3.0 - 2.0 * t)
        deform_w = np.where(verts[:, 2] <= z_guard, 0.0, deform_w)

    z_new = verts[:, 2] - offsets * deform_w
    upper_mask = verts[:, 2] > z_guard
    z_new[upper_mask] = np.maximum(z_new[upper_mask], z_guard)
    verts[:, 2] = z_new

    deformed = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    out = Path(out_stl_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    deformed.export(out)
    return out


def flatten_stl_bottom(
    input_stl: str | Path,
    output_stl: str | Path,
    flatten_depth_mm: float,
) -> Path:
    if flatten_depth_mm <= 0.0:
        raise ValueError("flatten_depth_mm must be > 0")

    loaded = trimesh.load(input_stl, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("Input mesh has no geometry")
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError("Input mesh is not a valid Trimesh")

    verts = mesh.vertices.copy()
    z_min = float(np.min(verts[:, 2]))
    z_cut = z_min + float(flatten_depth_mm)
    verts[:, 2] = np.maximum(verts[:, 2], z_cut)

    flat = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    out = Path(output_stl)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat.export(out)
    return out


def run_prusaslicer_export_gcode(
    prusaslicer_cli: str | Path,
    input_stl: str | Path,
    output_gcode: str | Path,
    profile_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    exe = resolve_prusaslicer_cli(str(prusaslicer_cli))
    cmd: list[str] = [str(exe), "--export-gcode", str(input_stl), "--output", str(output_gcode)]

    if profile_path:
        cmd.extend(["--load", str(profile_path)])

    if extra_args:
        cmd.extend(list(extra_args))

    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "PrusaSlicer CLI failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
    return proc


_TOKEN = re.compile(r"^([A-Za-z])([+-]?\d+(?:\.\d*)?|[+-]?\.\d+)$")


def _format_float(value: float) -> str:
    text = f"{value:.5f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _layer_factor(idx: int, preserve_planar_layers: int, transition_layers: int) -> float:
    if idx < preserve_planar_layers:
        return 0.0
    if transition_layers <= 0:
        return 1.0
    rel = idx - preserve_planar_layers
    if rel < transition_layers:
        t = float(rel + 1) / float(transition_layers + 1)
        # Smoothstep blend to avoid abrupt slope changes at transition boundaries.
        return t * t * (3.0 - 2.0 * t)
    return 1.0


def _collect_xy_bounds(gcode_path: str | Path) -> tuple[float, float, float, float]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    cur_x = 0.0
    cur_y = 0.0
    abs_xy = True

    with Path(gcode_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            code = line.split(";", 1)[0].strip()
            if not code:
                continue
            cmd = code.split()[0].upper()
            if cmd == "G90":
                abs_xy = True
                continue
            if cmd == "G91":
                abs_xy = False
                continue
            if cmd not in {"G0", "G1"}:
                continue

            vals: dict[str, float] = {}
            for w in code.split()[1:]:
                m = _TOKEN.match(w)
                if not m:
                    continue
                vals[m.group(1).upper()] = float(m.group(2))
            if abs_xy:
                if "X" in vals:
                    cur_x = vals["X"]
                if "Y" in vals:
                    cur_y = vals["Y"]
            else:
                cur_x += vals.get("X", 0.0)
                cur_y += vals.get("Y", 0.0)

            x_vals.append(cur_x)
            y_vals.append(cur_y)

    if not x_vals or not y_vals:
        raise ValueError(f"Could not extract XY bounds from gcode: {gcode_path}")

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if x_arr.size >= 200:
        qlo, qhi = 0.02, 0.98
        x_lo, x_hi = np.quantile(x_arr, [qlo, qhi])
        y_lo, y_hi = np.quantile(y_arr, [qlo, qhi])
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)
    return float(np.min(x_arr)), float(np.max(x_arr)), float(np.min(y_arr)), float(np.max(y_arr))


def _collect_xy_positions(gcode_path: str | Path) -> np.ndarray:
    points: list[tuple[float, float]] = []
    cur_x = 0.0
    cur_y = 0.0
    abs_xy = True

    with Path(gcode_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            code = line.split(";", 1)[0].strip()
            if not code:
                continue
            cmd = code.split()[0].upper()
            if cmd == "G90":
                abs_xy = True
                continue
            if cmd == "G91":
                abs_xy = False
                continue
            if cmd not in {"G0", "G1"}:
                continue

            vals: dict[str, float] = {}
            for w in code.split()[1:]:
                m = _TOKEN.match(w)
                if not m:
                    continue
                vals[m.group(1).upper()] = float(m.group(2))

            if abs_xy:
                if "X" in vals:
                    cur_x = vals["X"]
                if "Y" in vals:
                    cur_y = vals["Y"]
            else:
                cur_x += vals.get("X", 0.0)
                cur_y += vals.get("Y", 0.0)

            points.append((cur_x, cur_y))

    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _collect_extrusion_samples(gcode_path: str | Path) -> np.ndarray:
    samples: list[tuple[float, float, float, float]] = []

    cur_x = 0.0
    cur_y = 0.0
    cur_z = 0.0
    cur_e = 0.0
    abs_xyz = True
    abs_e = True
    layer_idx = -1

    with Path(gcode_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            code_raw, sep, comment = raw.partition(";")
            comment_text = comment.strip() if sep else ""
            if comment_text:
                up = comment_text.upper()
                if up.startswith("LAYER_CHANGE"):
                    layer_idx += 1
                m_layer = re.match(r"^LAYER\s*:\s*(-?\d+)\s*$", comment_text, flags=re.IGNORECASE)
                if m_layer:
                    layer_idx = int(m_layer.group(1))

            code = code_raw.strip()
            if not code:
                continue
            cmd = code.split()[0].upper()
            if cmd == "G90":
                abs_xyz = True
                continue
            if cmd == "G91":
                abs_xyz = False
                continue
            if cmd == "M82":
                abs_e = True
                continue
            if cmd == "M83":
                abs_e = False
                continue
            if cmd not in {"G0", "G1"}:
                continue

            vals: dict[str, float] = {}
            for w in code.split()[1:]:
                m = _TOKEN.match(w)
                if not m:
                    continue
                vals[m.group(1).upper()] = float(m.group(2))

            if abs_xyz:
                nx = vals.get("X", cur_x)
                ny = vals.get("Y", cur_y)
                nz = vals.get("Z", cur_z)
            else:
                nx = cur_x + vals.get("X", 0.0)
                ny = cur_y + vals.get("Y", 0.0)
                nz = cur_z + vals.get("Z", 0.0)

            if abs_e:
                ne = vals.get("E", cur_e)
                delta_e = ne - cur_e
            else:
                de = vals.get("E", 0.0)
                ne = cur_e + de
                delta_e = de

            if delta_e > 1e-8:
                samples.append((nx, ny, nz, float(layer_idx)))

            cur_x, cur_y, cur_z, cur_e = nx, ny, nz, ne

    if not samples:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(samples, dtype=float)


def _estimate_transition_lift(
    gcode_path: str | Path,
    sampler: SurfaceSampler,
    shift_x: float,
    shift_y: float,
    z_baseline: float,
    preserve_planar_layers: int,
    transition_layers: int,
) -> float:
    cur_x = 0.0
    cur_y = 0.0
    cur_z = 0.0
    cur_e = 0.0
    abs_xyz = True
    abs_e = True
    layer_idx = -1

    min_def = float("inf")
    min_warp = float("inf")

    with Path(gcode_path).open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            code_raw, sep, comment = line.partition(";")
            comment_text = comment.strip() if sep else ""
            if comment_text:
                up = comment_text.upper()
                if up.startswith("LAYER_CHANGE"):
                    layer_idx += 1
                m_layer = re.match(r"^LAYER\s*:\s*(-?\d+)\s*$", comment_text, flags=re.IGNORECASE)
                if m_layer:
                    layer_idx = int(m_layer.group(1))

            code = code_raw.strip()
            if not code:
                continue
            cmd = code.split()[0].upper()
            if cmd == "G90":
                abs_xyz = True
                continue
            if cmd == "G91":
                abs_xyz = False
                continue
            if cmd == "M82":
                abs_e = True
                continue
            if cmd == "M83":
                abs_e = False
                continue
            if cmd not in {"G0", "G1"}:
                continue

            vals: dict[str, float] = {}
            for w in code.split()[1:]:
                m = _TOKEN.match(w)
                if not m:
                    continue
                vals[m.group(1).upper()] = float(m.group(2))

            if abs_xyz:
                nx = vals.get("X", cur_x)
                ny = vals.get("Y", cur_y)
                nz = vals.get("Z", cur_z)
            else:
                nx = cur_x + vals.get("X", 0.0)
                ny = cur_y + vals.get("Y", 0.0)
                nz = cur_z + vals.get("Z", 0.0)

            if abs_e:
                ne = vals.get("E", cur_e)
                delta_e = ne - cur_e
            else:
                de = vals.get("E", 0.0)
                ne = cur_e + de
                delta_e = de

            f_layer = _layer_factor(layer_idx, preserve_planar_layers, transition_layers)
            z_off = sampler.sample(nx - shift_x, ny - shift_y)
            z_warp = nz + f_layer * (z_off - z_baseline)

            if delta_e > 1e-8 and layer_idx == preserve_planar_layers:
                min_def = min(min_def, nz)
                min_warp = min(min_warp, z_warp)

            cur_x, cur_y, cur_z, cur_e = nx, ny, nz, ne

    if not np.isfinite(min_def) or not np.isfinite(min_warp):
        return 0.0
    return max(0.0, float(min_warp - min_def))


def warp_gcode_with_surface(
    input_gcode: str | Path,
    output_gcode: str | Path,
    result: QuickCurveResult,
    xy_shift: tuple[float, float] | None = None,
    auto_align_xy: bool = True,
    z_anchor_to_bed: bool = True,
    preserve_planar_layers: int = 1,
    transition_layers: int = 4,
) -> Path:
    if preserve_planar_layers < 0:
        raise ValueError("preserve_planar_layers must be >= 0")
    if transition_layers < 0:
        raise ValueError("transition_layers must be >= 0")

    sampler = SurfaceSampler(
        surface=result.final_surface,
        x_coords=result.x_coords,
        y_coords=result.y_coords,
    )

    in_path = Path(input_gcode)
    out_path = Path(output_gcode)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sx_min, sx_max = float(result.x_coords[0]), float(result.x_coords[-1])
    sy_min, sy_max = float(result.y_coords[0]), float(result.y_coords[-1])
    src_cx = 0.5 * (sx_min + sx_max)
    src_cy = 0.5 * (sy_min + sy_max)

    if xy_shift is not None:
        shift_x, shift_y = float(xy_shift[0]), float(xy_shift[1])
    elif auto_align_xy:
        gx_min, gx_max, gy_min, gy_max = _collect_xy_bounds(in_path)
        gcode_cx = 0.5 * (gx_min + gx_max)
        gcode_cy = 0.5 * (gy_min + gy_max)
        shift_x = gcode_cx - src_cx
        shift_y = gcode_cy - src_cy
    else:
        shift_x = 0.0
        shift_y = 0.0

    z_baseline = 0.0
    if z_anchor_to_bed:
        ext = _collect_extrusion_samples(in_path)
        if ext.shape[0] > 0:
            layer_col = ext[:, 3]
            basis = ext[layer_col == float(preserve_planar_layers)]
            if basis.shape[0] == 0 and preserve_planar_layers > 0:
                basis = ext[layer_col == float(preserve_planar_layers - 1)]
            if basis.shape[0] == 0:
                z_first = float(np.min(ext[:, 2]))
                basis = ext[ext[:, 2] <= (z_first + 0.05)]
            if basis.shape[0] == 0:
                basis = ext
            offsets = np.array(
                [sampler.sample(float(x - shift_x), float(y - shift_y)) for x, y, _, _ in basis],
                dtype=float,
            )
        else:
            xy_points = _collect_xy_positions(in_path)
            offsets = np.array(
                [sampler.sample(float(x - shift_x), float(y - shift_y)) for x, y in xy_points],
                dtype=float,
            ) if xy_points.shape[0] > 0 else np.array([], dtype=float)

        finite = offsets[np.isfinite(offsets)]
        if finite.size > 0:
            if preserve_planar_layers > 0:
                z_baseline = float(np.median(finite))
            else:
                z_baseline = float(np.min(finite))

    transition_lift = _estimate_transition_lift(
        gcode_path=in_path,
        sampler=sampler,
        shift_x=shift_x,
        shift_y=shift_y,
        z_baseline=z_baseline,
        preserve_planar_layers=preserve_planar_layers,
        transition_layers=transition_layers,
    )

    abs_xyz = True
    abs_e = True
    cur_x = 0.0
    cur_y = 0.0
    cur_z_def = 0.0
    cur_z_warp = 0.0
    cur_e = 0.0
    layer_idx = -1

    with in_path.open("r", encoding="utf-8", errors="replace") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        dst.write("; QuickCurve non-planar warp applied\n")
        dst.write(f"; Source gcode: {in_path}\n")
        dst.write(f"; XY shift applied before sampling: dx={_format_float(shift_x)} dy={_format_float(shift_y)}\n")
        dst.write(f"; Z baseline removed from warp: {_format_float(z_baseline)}\n")
        dst.write(f"; Transition lift correction: {_format_float(transition_lift)}\n")
        dst.write(f"; Planar layers preserved: {preserve_planar_layers}\n")
        dst.write(f"; Warp transition layers: {transition_layers}\n")

        for line in src:
            raw = line.rstrip("\n")
            if not raw:
                dst.write("\n")
                continue

            code, sep, comment = raw.partition(";")
            comment_text = comment.strip() if sep else ""
            if comment_text:
                upper = comment_text.upper()
                if upper.startswith("LAYER_CHANGE"):
                    layer_idx += 1
                m_layer = re.match(r"^LAYER\s*:\s*(-?\d+)\s*$", comment_text, flags=re.IGNORECASE)
                if m_layer:
                    layer_idx = int(m_layer.group(1))

            stripped = code.strip()
            if not stripped:
                dst.write(line)
                continue

            words = stripped.split()
            cmd = words[0].upper()

            if cmd == "G90":
                abs_xyz = True
                dst.write(line)
                continue
            if cmd == "G91":
                abs_xyz = False
                dst.write(line)
                continue
            if cmd == "M82":
                abs_e = True
                dst.write(line)
                continue
            if cmd == "M83":
                abs_e = False
                dst.write(line)
                continue

            if cmd not in {"G0", "G1"}:
                dst.write(line)
                continue

            parsed: dict[str, float] = {}
            for w in words[1:]:
                m = _TOKEN.match(w)
                if not m:
                    continue
                letter = m.group(1).upper()
                val = float(m.group(2))
                if letter in {"X", "Y", "Z", "E"}:
                    parsed[letter] = val

            if abs_xyz:
                next_x = parsed.get("X", cur_x)
                next_y = parsed.get("Y", cur_y)
                next_z_def = parsed.get("Z", cur_z_def)
            else:
                next_x = cur_x + parsed.get("X", 0.0)
                next_y = cur_y + parsed.get("Y", 0.0)
                next_z_def = cur_z_def + parsed.get("Z", 0.0)

            if abs_e:
                next_e = parsed.get("E", cur_e)
            else:
                next_e = cur_e + parsed.get("E", 0.0)

            z_off = sampler.sample(next_x - shift_x, next_y - shift_y)
            f = _layer_factor(layer_idx, preserve_planar_layers, transition_layers)
            next_z_warp = next_z_def + f * (z_off - z_baseline)
            if f > 0.0 and layer_idx >= preserve_planar_layers:
                next_z_warp -= transition_lift

            if abs_xyz:
                z_token = f"Z{_format_float(next_z_warp)}"
            else:
                z_token = f"Z{_format_float(next_z_warp - cur_z_warp)}"

            rebuilt = [words[0]]
            for w in words[1:]:
                if _TOKEN.match(w) and w[0].upper() == "Z":
                    continue
                rebuilt.append(w)

            rebuilt.append(z_token)

            out_line = " ".join(rebuilt)
            if sep:
                out_line += ";" + comment
            dst.write(out_line + "\n")

            cur_x = next_x
            cur_y = next_y
            cur_z_def = next_z_def
            cur_z_warp = next_z_warp
            cur_e = next_e

    return out_path
