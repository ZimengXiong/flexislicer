from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class ToolpathConfig:
    layer_height: float = 0.2
    first_layer_height: float = 0.2
    line_width: float = 0.42
    filament_diameter: float = 1.75
    nozzle_temp_c: int = 200
    bed_temp_c: int = 60
    print_feed_mm_min: float = 1500.0
    travel_feed_mm_min: float = 7200.0
    perimeter_speed_mm_min: float = 1200.0
    infill_spacing: float = 0.40
    extrusion_multiplier: float = 1.0
    bed_offset_x: float = 70.0
    bed_offset_y: float = 80.0
    z_hop: float = 0.6
    infill_angle_deg: float = 0.0
    top_skin_angle_deg: float = 0.0
    top_skin_passes: int = 1
    prime_line_x: float = 15.0
    prime_line_y0: float = 10.0
    prime_line_y1: float = 90.0


def _fmt(v: float) -> str:
    s = f"{v:.5f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _close_poly(poly: np.ndarray) -> np.ndarray:
    if len(poly) == 0:
        return poly
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[0]])


def _polygon_area(poly: np.ndarray) -> float:
    pts = _close_poly(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _dedupe_points(poly: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in poly:
        if not out or np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out, dtype=float)


def _section_polygon(mesh: trimesh.Trimesh, z_query: float) -> np.ndarray | None:
    sec = mesh.section(plane_origin=[0.0, 0.0, float(z_query)], plane_normal=[0.0, 0.0, 1.0])
    if sec is None or not sec.discrete:
        return None
    loops = [np.asarray(loop[:, :2], dtype=float) for loop in sec.discrete if len(loop) >= 4]
    if not loops:
        return None
    poly = max(loops, key=lambda p: abs(_polygon_area(p)))
    poly = _dedupe_points(poly)
    if len(poly) < 3:
        return None
    if _polygon_area(poly) < 0.0:
        poly = poly[::-1]
    return poly


def _rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T


def _scanline_segments(poly: np.ndarray, spacing: float, angle_deg: float) -> list[np.ndarray]:
    ang = math.radians(angle_deg)
    work = _rotate(poly, -ang)
    closed = _close_poly(work)
    y_min = float(np.min(work[:, 1])) + spacing * 0.5
    y_max = float(np.max(work[:, 1])) - spacing * 0.5
    if y_max < y_min:
        return []

    lines: list[np.ndarray] = []
    y = y_min
    flip = False
    while y <= y_max + 1e-9:
        xs: list[float] = []
        for p0, p1 in zip(closed[:-1], closed[1:]):
            y0 = float(p0[1])
            y1 = float(p1[1])
            if abs(y1 - y0) < 1e-9:
                continue
            ymin = min(y0, y1)
            ymax = max(y0, y1)
            if not (ymin <= y < ymax):
                continue
            t = (y - y0) / (y1 - y0)
            x = float(p0[0] + t * (p1[0] - p0[0]))
            xs.append(x)
        xs.sort()
        segs_this_row: list[np.ndarray] = []
        for i in range(0, len(xs) - 1, 2):
            x0 = xs[i]
            x1 = xs[i + 1]
            if x1 - x0 <= 0.15:
                continue
            if flip:
                seg = np.array([[x1, y], [x0, y]], dtype=float)
            else:
                seg = np.array([[x0, y], [x1, y]], dtype=float)
            segs_this_row.append(seg)
            flip = not flip
        lines.extend(segs_this_row)
        y += spacing

    if not lines:
        return []
    return [_rotate(seg, ang) for seg in lines]


def _top_plane(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    normals = np.asarray(mesh.face_normals, dtype=float)
    best = int(np.argmax(normals[:, 2]))
    n = normals[best]
    v0 = mesh.vertices[mesh.faces[best][0]]
    d = -float(np.dot(n, v0))
    return n, d


def _plane_z(plane_normal: np.ndarray, plane_d: float, x: float, y: float) -> float:
    nx, ny, nz = [float(v) for v in plane_normal]
    return -(plane_d + nx * x + ny * y) / nz


def _extrusion_amount(length_mm: float, cfg: ToolpathConfig, layer_scale: float = 1.0) -> float:
    area = cfg.line_width * cfg.layer_height * layer_scale
    filament_area = math.pi * (0.5 * cfg.filament_diameter) ** 2
    return cfg.extrusion_multiplier * area * length_mm / filament_area


class GCodeWriter:
    def __init__(self, cfg: ToolpathConfig) -> None:
        self.cfg = cfg
        self.lines: list[str] = []
        self.e = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def line(self, text: str) -> None:
        self.lines.append(text)

    def travel(self, x: float, y: float, z: float | None = None, feed: float | None = None) -> None:
        parts = ["G0"]
        parts.append(f"X{_fmt(x)}")
        parts.append(f"Y{_fmt(y)}")
        if z is not None:
            parts.append(f"Z{_fmt(z)}")
            self.z = z
        if feed is not None:
            parts.append(f"F{_fmt(feed)}")
        self.x = x
        self.y = y
        self.line(" ".join(parts))

    def extrude_to(self, x: float, y: float, z: float, feed: float, layer_scale: float = 1.0) -> None:
        dist = math.dist((self.x, self.y, self.z), (x, y, z))
        self.e += _extrusion_amount(dist, self.cfg, layer_scale=layer_scale)
        self.x = x
        self.y = y
        self.z = z
        self.line(
            f"G1 X{_fmt(x)} Y{_fmt(y)} Z{_fmt(z)} E{_fmt(self.e)} F{_fmt(feed)}"
        )

    def retract(self, amount: float = 0.8, feed: float = 2400.0) -> None:
        self.e -= amount
        self.line(f"G1 E{_fmt(self.e)} F{_fmt(feed)}")

    def unretract(self, amount: float = 0.8, feed: float = 2400.0) -> None:
        self.e += amount
        self.line(f"G1 E{_fmt(self.e)} F{_fmt(feed)}")

    def reset_extruder(self) -> None:
        self.e = 0.0
        self.line("G92 E0")

    def write_text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _add_start_gcode(w: GCodeWriter, cfg: ToolpathConfig) -> None:
    w.line("; Hand-coded slope toolpath")
    w.line("; First layer is fully planar at Z=0.2")
    w.line("; Only local topmost paths are lifted onto the true ramp")
    w.line(f"M140 S{cfg.bed_temp_c}")
    w.line(f"M104 S{cfg.nozzle_temp_c}")
    w.line("G28")
    w.line(f"M190 S{cfg.bed_temp_c}")
    w.line(f"M109 S{cfg.nozzle_temp_c}")
    w.line("G21")
    w.line("G90")
    w.line("M82")
    w.reset_extruder()
    w.travel(cfg.prime_line_x, cfg.prime_line_y0, z=0.3, feed=cfg.travel_feed_mm_min)
    w.extrude_to(cfg.prime_line_x, cfg.prime_line_y1, 0.3, feed=900.0)
    w.travel(cfg.prime_line_x + 0.5, cfg.prime_line_y1, z=0.3, feed=cfg.travel_feed_mm_min)
    w.extrude_to(cfg.prime_line_x + 0.5, cfg.prime_line_y0, 0.3, feed=900.0)
    w.reset_extruder()


def _add_end_gcode(w: GCodeWriter) -> None:
    w.line("M104 S0")
    w.line("M140 S0")
    w.line("G1 E-1 F1800")
    w.line("G1 Z10 F3000")
    w.line("G0 X0 Y180 F6000")
    w.line("M84")


def _emit_polyline(
    w: GCodeWriter,
    pts_xy: np.ndarray,
    z_func,
    start_z: float,
    feed: float,
    close_loop: bool,
) -> None:
    pts = pts_xy.copy()
    if close_loop:
        pts = _close_poly(pts)[:-1]
    start = pts[0]
    z0 = max(start_z, float(z_func(start[0], start[1])))
    w.travel(start[0], start[1], z=min(z0 + w.cfg.z_hop, max(z0, w.z + w.cfg.z_hop)), feed=w.cfg.travel_feed_mm_min)
    w.travel(start[0], start[1], z=z0, feed=w.cfg.travel_feed_mm_min)
    w.unretract()
    for p in pts[1:]:
        z = max(start_z, float(z_func(p[0], p[1])))
        w.extrude_to(float(p[0]), float(p[1]), z, feed=feed)
    if close_loop:
        p = pts[0]
        z = max(start_z, float(z_func(p[0], p[1])))
        w.extrude_to(float(p[0]), float(p[1]), z, feed=feed)
    w.retract()


def generate_slope_gcode(mesh_path: Path, out_path: Path, cfg: ToolpathConfig) -> None:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a single watertight mesh")

    plane_n, plane_d = _top_plane(mesh)
    max_z = float(mesh.bounds[1, 2])
    total_layers = int(math.ceil((max_z - cfg.first_layer_height) / cfg.layer_height)) + 1
    eps = 1e-4
    top_skin_poly = _section_polygon(mesh, cfg.first_layer_height)
    if top_skin_poly is None:
        raise ValueError("Could not extract printable first-layer/top-skin footprint")
    top_skin_poly = top_skin_poly + np.array([cfg.bed_offset_x, cfg.bed_offset_y], dtype=float)

    w = GCodeWriter(cfg)
    _add_start_gcode(w, cfg)

    def top_z_func(xb: float, yb: float) -> float:
        x = xb - cfg.bed_offset_x
        y = yb - cfg.bed_offset_y
        z_top = _plane_z(plane_n, plane_d, x, y)
        return min(max_z, max(cfg.first_layer_height, z_top))

    w.line(";LAYER:0")
    w.line(f";Z:{_fmt(cfg.first_layer_height)}")
    _emit_polyline(
        w=w,
        pts_xy=top_skin_poly,
        z_func=lambda _x, _y, z=cfg.first_layer_height: z,
        start_z=cfg.first_layer_height,
        feed=cfg.perimeter_speed_mm_min,
        close_loop=True,
    )
    first_infill = _scanline_segments(top_skin_poly, spacing=cfg.infill_spacing, angle_deg=cfg.infill_angle_deg)
    for seg in first_infill:
        seg = np.asarray(seg, dtype=float)
        start = seg[0]
        end = seg[1]
        z_emit = cfg.first_layer_height
        w.travel(float(start[0]), float(start[1]), z=min(z_emit + cfg.z_hop, max(z_emit, w.z + cfg.z_hop)), feed=cfg.travel_feed_mm_min)
        w.travel(float(start[0]), float(start[1]), z=z_emit, feed=cfg.travel_feed_mm_min)
        w.unretract()
        w.extrude_to(float(end[0]), float(end[1]), z_emit, feed=cfg.print_feed_mm_min, layer_scale=1.0)
        w.retract()

    max_offset_layers = int(math.floor((max_z - cfg.first_layer_height) / cfg.layer_height + 1e-9))
    for step in range(max_offset_layers, -1, -1):
        offset = step * cfg.layer_height
        if offset <= 1e-9:
            query_z = cfg.first_layer_height
        else:
            query_z = cfg.first_layer_height + offset
        poly = _section_polygon(mesh, query_z)
        if poly is None:
            continue
        poly = poly + np.array([cfg.bed_offset_x, cfg.bed_offset_y], dtype=float)

        def layer_z_func(xb: float, yb: float, off: float = offset) -> float:
            return max(cfg.first_layer_height, top_z_func(xb, yb) - off)

        layer_index = max_offset_layers - step + 1
        z_hint = max(cfg.first_layer_height, max_z - offset)
        w.line(f";LAYER:{layer_index}")
        w.line(f";Z:{_fmt(z_hint)}")
        w.line(f";NONPLANAR_OFFSET:{_fmt(offset)}")
        _emit_polyline(
            w=w,
            pts_xy=poly,
            z_func=layer_z_func,
            start_z=cfg.first_layer_height,
            feed=cfg.perimeter_speed_mm_min,
            close_loop=True,
        )
        infill_angle = cfg.top_skin_angle_deg
        infill = _scanline_segments(poly, spacing=cfg.infill_spacing, angle_deg=infill_angle)
        for seg in infill:
            seg = np.asarray(seg, dtype=float)
            start = seg[0]
            end = seg[1]
            z_start = layer_z_func(float(start[0]), float(start[1]))
            z_end = layer_z_func(float(end[0]), float(end[1]))
            w.travel(float(start[0]), float(start[1]), z=min(z_start + cfg.z_hop, max(z_start, w.z + cfg.z_hop)), feed=cfg.travel_feed_mm_min)
            w.travel(float(start[0]), float(start[1]), z=z_start, feed=cfg.travel_feed_mm_min)
            w.unretract()
            w.extrude_to(float(end[0]), float(end[1]), z_end, feed=cfg.print_feed_mm_min, layer_scale=1.0)
            w.retract()

    _add_end_gcode(w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(w.write_text(), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate dedicated hand-coded G-code for a simple sloped wedge STL.")
    p.add_argument("--mesh", default="/Users/zimengx/Downloads/slope.stl")
    p.add_argument("--out", default="/Users/zimengx/Projects/flexislicer/out_slope_handcoded/slope_handcoded.gcode")
    p.add_argument("--bed-offset-x", type=float, default=70.0)
    p.add_argument("--bed-offset-y", type=float, default=80.0)
    args = p.parse_args()

    cfg = ToolpathConfig(
        bed_offset_x=args.bed_offset_x,
        bed_offset_y=args.bed_offset_y,
    )
    generate_slope_gcode(Path(args.mesh), Path(args.out), cfg)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
