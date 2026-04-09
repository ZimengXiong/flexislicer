from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage import measure
import trimesh


@dataclass
class flexiSlicerConfig:
    grid_step: float = 0.5
    layer_height: float = 0.2
    theta_target_deg: float = 27.0
    theta_max_deg: float = 40.0
    filter_radius_mm: float = 0.0
    max_post_iters: int = 200
    w_gradient: float = 1.0
    w_boundary: float = 3.0
    w_smooth: float = 0.05
    w_component_reg: float = 1e-3
    max_layers: int | None = None

    @property
    def h_target(self) -> float:
        return self.grid_step * math.tan(math.radians(self.theta_target_deg))

    @property
    def h_max(self) -> float:
        return self.grid_step * math.tan(math.radians(self.theta_max_deg))


@dataclass
class flexiSlicerResult:
    x_coords: np.ndarray
    y_coords: np.ndarray
    top_z: np.ndarray
    valid_mask: np.ndarray
    theta_mask: np.ndarray
    labels: np.ndarray
    raw_surface: np.ndarray
    final_surface: np.ndarray
    layers: list[dict[str, Any]]
    stats: dict[str, Any]


def _disk_structure(radius_px: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (xx * xx + yy * yy) <= (radius_px * radius_px)


def _fill_nans_nearest(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if np.all(valid):
        return arr.copy()
    if not np.any(valid):
        return np.zeros_like(arr)
    _, indices = ndi.distance_transform_edt(~valid, return_indices=True)
    return arr[tuple(indices)]


def _shift_with_nan(arr: np.ndarray, di: int, dj: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    src_i = slice(max(0, -di), min(arr.shape[0], arr.shape[0] - di))
    src_j = slice(max(0, -dj), min(arr.shape[1], arr.shape[1] - dj))
    dst_i = slice(max(0, di), min(arr.shape[0], arr.shape[0] + di))
    dst_j = slice(max(0, dj), min(arr.shape[1], arr.shape[1] + dj))
    out[dst_i, dst_j] = arr[src_i, src_j]
    return out


def _bilinear_sample(arr: np.ndarray, row: float, col: float) -> float:
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
    vals = np.array([v00, v01, v10, v11], dtype=float)
    if np.any(~np.isfinite(vals)):
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    top = v00 * (1.0 - fc) + v01 * fc
    bot = v10 * (1.0 - fc) + v11 * fc
    return float(top * (1.0 - fr) + bot * fr)


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("Input mesh has no geometry")
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError("Unsupported input type")

    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError("Failed to load a valid mesh")

    return mesh


def _sample_mesh_to_grid(mesh: trimesh.Trimesh, step: float) -> dict[str, Any]:
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    nx = int(np.floor((x_max - x_min) / step)) + 1
    ny = int(np.floor((y_max - y_min) / step)) + 1

    x_coords = x_min + np.arange(nx, dtype=float) * step
    y_coords = y_min + np.arange(ny, dtype=float) * step

    xv, yv = np.meshgrid(x_coords, y_coords)
    rays_origins = np.column_stack(
        [xv.ravel(), yv.ravel(), np.full(xv.size, z_max + 10.0 * step)]
    )
    rays_dirs = np.tile(np.array([0.0, 0.0, -1.0]), (xv.size, 1))

    loc, idx_ray, idx_tri = mesh.ray.intersects_location(
        ray_origins=rays_origins,
        ray_directions=rays_dirs,
        multiple_hits=True,
    )

    hits_by_ray: list[list[float]] = [[] for _ in range(xv.size)]
    top_z = np.full(xv.size, np.nan)
    top_tri = np.full(xv.size, -1, dtype=int)

    for p, rid, tid in zip(loc, idx_ray, idx_tri):
        z = float(p[2])
        hits_by_ray[int(rid)].append(z)
        if not np.isfinite(top_z[rid]) or z > top_z[rid]:
            top_z[rid] = z
            top_tri[rid] = int(tid)

    ray_hits_sorted: list[np.ndarray] = []
    for values in hits_by_ray:
        if values:
            values.sort()
            ray_hits_sorted.append(np.array(values, dtype=float))
        else:
            ray_hits_sorted.append(np.array([], dtype=float))

    top_z_2d = top_z.reshape(ny, nx)
    valid_mask = np.isfinite(top_z_2d)

    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "top_z": top_z_2d,
        "valid_mask": valid_mask,
        "ray_hits": ray_hits_sorted,
        "grid_shape": (ny, nx),
        "bounds": bounds,
        "step": step,
    }


def _build_theta_map(
    top_z: np.ndarray,
    valid_mask: np.ndarray,
    step: float,
    theta_target_deg: float,
    filter_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    filled = _fill_nans_nearest(top_z)
    gy, gx = np.gradient(filled, step, step)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    theta_mask = valid_mask & (slope_deg <= theta_target_deg)

    if filter_radius_mm > 0.0:
        radius_px = max(1, int(round(filter_radius_mm / step)))
        theta_mask = ndi.binary_closing(theta_mask, structure=_disk_structure(radius_px))
        theta_mask &= valid_mask

    labels, _ = ndi.label(theta_mask, structure=np.ones((3, 3), dtype=bool))
    return theta_mask, labels


def _solve_slice_surface(
    top_z: np.ndarray,
    valid_mask: np.ndarray,
    theta_mask: np.ndarray,
    labels: np.ndarray,
    cfg: flexiSlicerConfig,
) -> np.ndarray:
    ny, nx = top_z.shape
    free_mask = valid_mask & ~theta_mask

    free_idx = np.full((ny, nx), -1, dtype=int)
    free_idx[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=int)
    n_free = int(np.count_nonzero(free_mask))

    comp_ids = [int(v) for v in np.unique(labels) if v > 0]
    comp_var = {cid: n_free + k for k, cid in enumerate(comp_ids)}
    n_unknown = n_free + len(comp_ids)

    if n_unknown == 0:
        return np.full_like(top_z, np.nan)

    comp_ref: dict[int, float] = {}
    for cid in comp_ids:
        pts = np.argwhere(labels == cid)
        i0, j0 = pts[0]
        comp_ref[cid] = float(top_z[i0, j0])

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_id = 0

    def add_eq(coeffs: list[tuple[int, float]], b: float, w: float) -> None:
        nonlocal row_id
        for var, val in coeffs:
            rows.append(row_id)
            cols.append(var)
            data.append(float(val) * w)
        rhs.append(float(b) * w)
        row_id += 1

    # Gradient-steepening objective in free regions.
    for i in range(ny):
        for j in range(nx):
            if not free_mask[i, j]:
                continue
            vi = int(free_idx[i, j])

            for di, dj in ((0, 1), (1, 0)):
                ni, nj = i + di, j + dj
                if ni >= ny or nj >= nx:
                    continue
                if not free_mask[ni, nj]:
                    continue

                vj = int(free_idx[ni, nj])
                sign = 1.0 if top_z[i, j] > top_z[ni, nj] else -1.0
                add_eq([(vi, 1.0), (vj, -1.0)], sign * cfg.h_target, cfg.w_gradient)

                if cfg.w_smooth > 0.0:
                    add_eq([(vi, 1.0), (vj, -1.0)], 0.0, cfg.w_smooth)

    # Boundary consistency between free variables and fixed target components.
    for i in range(ny):
        for j in range(nx):
            if not theta_mask[i, j]:
                continue
            cid = int(labels[i, j])
            zvar = comp_var[cid]
            target_rhs = float(top_z[i, j] - comp_ref[cid])

            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or nj < 0 or ni >= ny or nj >= nx:
                        continue
                    if not free_mask[ni, nj]:
                        continue
                    vf = int(free_idx[ni, nj])
                    add_eq([(vf, 1.0), (zvar, -1.0)], target_rhs, cfg.w_boundary)

    # Weak regularization to avoid null-space drift.
    if comp_ids:
        for cid in comp_ids:
            add_eq([(comp_var[cid], 1.0)], 0.0, cfg.w_component_reg)
    elif n_free > 0:
        add_eq([(0, 1.0)], 0.0, cfg.w_component_reg)

    if row_id == 0:
        return np.full_like(top_z, np.nan)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(row_id, n_unknown)).tocsr()
    b = np.array(rhs, dtype=float)

    sol = lsqr(A, b, atol=1e-9, btol=1e-9, iter_lim=20000)[0]

    h = np.full_like(top_z, np.nan)
    h[free_mask] = sol[free_idx[free_mask]]

    for cid in comp_ids:
        mask = labels == cid
        h[mask] = sol[comp_var[cid]] + top_z[mask] - comp_ref[cid]

    if np.any(np.isfinite(h[valid_mask])):
        h_shift = np.nanmin(h[valid_mask])
        h[valid_mask] -= h_shift

    return h


def _enforce_postprocess(
    h: np.ndarray,
    valid_mask: np.ndarray,
    h_max: float,
    max_iters: int,
) -> np.ndarray:
    out = h.copy()
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for _ in range(max_iters):
        prev = out.copy()
        neigh_term = np.full_like(out, -np.inf)

        for di, dj in neighbors:
            shifted = _shift_with_nan(out, di, dj)
            neigh_term = np.maximum(neigh_term, shifted - h_max)

        out = np.where(valid_mask, np.maximum(out, neigh_term), np.nan)

        delta = np.nanmax(np.abs(out - prev))
        if not np.isfinite(delta) or delta < 1e-6:
            break

    return out


def _extract_layers(
    h: np.ndarray,
    valid_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ray_hits: list[np.ndarray],
    bounds: np.ndarray,
    cfg: flexiSlicerConfig,
) -> list[dict[str, Any]]:
    ny, nx = h.shape
    z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])

    span = max(0.0, z_max - z_min)
    num_layers = int(math.ceil(span / cfg.layer_height)) + 1
    if cfg.max_layers is not None:
        num_layers = min(num_layers, cfg.max_layers)

    layers: list[dict[str, Any]] = []

    for layer_idx in range(num_layers):
        base = z_min + layer_idx * cfg.layer_height
        occ = np.zeros((ny, nx), dtype=bool)

        for rid, hits in enumerate(ray_hits):
            if hits.size == 0:
                continue
            iy = rid // nx
            ix = rid % nx
            if not valid_mask[iy, ix]:
                continue

            zq = base + h[iy, ix]
            n = int(np.searchsorted(hits, zq, side="right"))
            occ[iy, ix] = (n % 2) == 1

        if not np.any(occ):
            continue

        contours = measure.find_contours(occ.astype(float), 0.5)
        layer_paths: list[list[list[float]]] = []

        for contour in contours:
            if contour.shape[0] < 3:
                continue
            pts: list[list[float]] = []
            for row, col in contour:
                x = float(x_coords[0] + col * cfg.grid_step)
                y = float(y_coords[0] + row * cfg.grid_step)
                hz = _bilinear_sample(h, float(row), float(col))
                if not np.isfinite(hz):
                    continue
                z = float(base + hz)
                pts.append([x, y, z])

            if len(pts) >= 3:
                layer_paths.append(pts)

        if layer_paths:
            layers.append(
                {
                    "layer_index": layer_idx,
                    "base_z": float(base),
                    "contours": layer_paths,
                }
            )

    return layers


def run_flexiSlicer(mesh_path: str | Path, cfg: flexiSlicerConfig) -> flexiSlicerResult:
    mesh = _load_mesh(mesh_path)

    sampled = _sample_mesh_to_grid(mesh, cfg.grid_step)
    x_coords = sampled["x_coords"]
    y_coords = sampled["y_coords"]
    top_z = sampled["top_z"]
    valid_mask = sampled["valid_mask"]

    theta_mask, labels = _build_theta_map(
        top_z=top_z,
        valid_mask=valid_mask,
        step=cfg.grid_step,
        theta_target_deg=cfg.theta_target_deg,
        filter_radius_mm=cfg.filter_radius_mm,
    )

    raw_surface = _solve_slice_surface(
        top_z=top_z,
        valid_mask=valid_mask,
        theta_mask=theta_mask,
        labels=labels,
        cfg=cfg,
    )

    final_surface = _enforce_postprocess(
        h=raw_surface,
        valid_mask=valid_mask,
        h_max=cfg.h_max,
        max_iters=cfg.max_post_iters,
    )

    layers = _extract_layers(
        h=final_surface,
        valid_mask=valid_mask,
        x_coords=x_coords,
        y_coords=y_coords,
        ray_hits=sampled["ray_hits"],
        bounds=sampled["bounds"],
        cfg=cfg,
    )

    stats = {
        "grid_shape": list(top_z.shape),
        "grid_step": cfg.grid_step,
        "num_valid": int(np.count_nonzero(valid_mask)),
        "num_theta": int(np.count_nonzero(theta_mask)),
        "num_components": int(np.max(labels)),
        "num_layers_with_paths": len(layers),
        "mesh_faces": int(mesh.faces.shape[0]),
        "mesh_vertices": int(mesh.vertices.shape[0]),
    }

    return flexiSlicerResult(
        x_coords=x_coords,
        y_coords=y_coords,
        top_z=top_z,
        valid_mask=valid_mask,
        theta_mask=theta_mask,
        labels=labels,
        raw_surface=raw_surface,
        final_surface=final_surface,
        layers=layers,
        stats=stats,
    )


def save_result(result: flexiSlicerResult, cfg: flexiSlicerConfig, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "x_coords.npy", result.x_coords)
    np.save(out / "y_coords.npy", result.y_coords)
    np.save(out / "top_z.npy", result.top_z)
    np.save(out / "valid_mask.npy", result.valid_mask)
    np.save(out / "theta_mask.npy", result.theta_mask)
    np.save(out / "labels.npy", result.labels)
    np.save(out / "raw_surface.npy", result.raw_surface)
    np.save(out / "final_surface.npy", result.final_surface)

    with (out / "layers.json").open("w", encoding="utf-8") as f:
        json.dump(result.layers, f)

    metadata = {
        "config": asdict(cfg),
        "stats": result.stats,
        "notes": [
            "Reference implementation of the flexiSlicer paper method.",
            "Uses vertical ray sampling, least-squares surface solve, and top-down post-process.",
            "Contours are extracted from deformed raster slices.",
        ],
    }

    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
