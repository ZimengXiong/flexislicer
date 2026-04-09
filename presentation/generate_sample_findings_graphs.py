from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation" / "generated" / "sample_findings_graphs"

BG = "#ffffff"
GRID = "#d9dde3"
INK = "#1f2933"
PLANAR = "#d5d9df"
CURVI = "#c97b2a"
FLEX = "#0f766e"
QUICK = "#6b7280"
LIMIT = "#374151"
POINT = "#0b1526"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


SAMPLE_DATA = {
    "runtime_seconds": {
        "Planar": [0.60, 0.63, 0.61, 0.64, 0.62],
        "QuickCurve": [1.20, 1.23, 1.21, 1.25, 1.22],
        "FlexSlicer": [1.26, 1.30, 1.28, 1.31, 1.27],
        "CurviSlicer": [12.6, 13.0, 12.8, 13.2, 12.9],
    },
    "unique_layer_signatures": {
        "Planar": [1, 1, 1, 1, 1],
        "CurviSlicer": [5, 4, 6, 5, 4],
        "FlexSlicer": [18, 14, 20, 17, 16],
    },
    "nonplanar_layers": {
        "Planar": [0, 0, 0, 0, 0],
        "CurviSlicer": [10, 12, 11, 13, 12],
        "FlexSlicer": [21, 24, 22, 26, 23],
    },
    "z_change_events": {
        "Planar": [0, 0, 0, 0, 0],
        "CurviSlicer": [12, 14, 10, 16, 13],
        "FlexSlicer": [42, 48, 36, 55, 45],
    },
    "avg_delta_z_mm": {
        "Planar": [0.0, 0.0, 0.0, 0.0, 0.0],
        "CurviSlicer": [0.38, 0.42, 0.36, 0.45, 0.39],
        "FlexSlicer": [3.1, 3.4, 3.0, 3.6, 3.3],
    },
    "peak_delta_z_mm": {
        "Planar": [0.0, 0.0, 0.0, 0.0, 0.0],
        "CurviSlicer": [0.72, 0.85, 0.68, 0.92, 0.80],
        "FlexSlicer": [8.7, 9.9, 8.1, 10.4, 9.5],
    },
    "peak_nozzle_angle_deg": {
        "CurviSlicer": [27.5, 28.0, 26.8, 29.1, 27.2],
        "FlexSlicer": [31.0, 31.5, 30.2, 32.1, 30.7],
        "Safe limit": [35.0, 35.0, 35.0, 35.0, 35.0],
    },
}


Y_LABELS = {
    "runtime_seconds": "Runtime (s)",
    "unique_layer_signatures": "Unique layer signatures",
    "nonplanar_layers": "Non-planar layers",
    "z_change_events": "Z change events",
    "avg_delta_z_mm": "Average delta Z (mm)",
    "peak_delta_z_mm": "Peak delta Z (mm)",
    "peak_nozzle_angle_deg": "Peak nozzle angle (deg)",
}


COLOR_MAP = {
    "Planar": PLANAR,
    "CurviSlicer": CURVI,
    "QuickCurve": QUICK,
    "FlexSlicer": FLEX,
    "Safe limit": LIMIT,
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def configure_runtime_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(0.5, 20.0)
    ax.set_yticks([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8, alpha=0.7)


def save_metric_plot(metric_name: str, values_by_method: dict[str, list[float]]) -> list[Path]:
    methods = list(values_by_method)
    means = [float(np.mean(values_by_method[method])) for method in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(4.6, 3.35), constrained_layout=True)
    if metric_name == "runtime_seconds":
        runtime_floor = 0.5
        ax.bar(
            x,
            [max(mean - runtime_floor, 0.01) for mean in means],
            bottom=runtime_floor,
            width=0.62,
            color=[COLOR_MAP[method] for method in methods],
            edgecolor="#eef2f6",
            linewidth=1.0,
        )
    else:
        ax.bar(
            x,
            means,
            width=0.62,
            color=[COLOR_MAP[method] for method in methods],
            edgecolor="#eef2f6",
            linewidth=1.0,
        )

    for idx, method in enumerate(methods):
        samples = values_by_method[method]
        if len(samples) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.11, 0.11, len(samples))
        ax.scatter(
            np.full(len(samples), idx) + jitter,
            samples,
            s=28,
            color=POINT,
            edgecolors=BG,
            linewidths=0.5,
            zorder=3,
            alpha=0.9,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel(Y_LABELS[metric_name])
    ax.set_xticks(x, methods)
    style_axes(ax)

    if metric_name in {"avg_delta_z_mm", "peak_delta_z_mm", "nonplanar_layers", "z_change_events"}:
        ax.set_ylim(bottom=0.0)
    elif metric_name == "runtime_seconds":
        configure_runtime_axis(ax)
    elif metric_name == "peak_nozzle_angle_deg":
        ax.set_ylim(0.0, 38.0)

    out_png = OUT_DIR / f"{metric_name}.png"
    out_pdf = OUT_DIR / f"{metric_name}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def build_contact_sheet() -> list[Path]:
    order = [
        "runtime_seconds",
        "unique_layer_signatures",
        "nonplanar_layers",
        "z_change_events",
        "avg_delta_z_mm",
        "peak_delta_z_mm",
        "peak_nozzle_angle_deg",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.0), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax, metric_name in zip(flat_axes, order):
        values_by_method = SAMPLE_DATA[metric_name]
        methods = list(values_by_method)
        means = [float(np.mean(values_by_method[method])) for method in methods]
        x = np.arange(len(methods))
        if metric_name == "runtime_seconds":
            runtime_floor = 0.5
            ax.bar(
                x,
                [max(mean - runtime_floor, 0.01) for mean in means],
                bottom=runtime_floor,
                width=0.62,
                color=[COLOR_MAP[method] for method in methods],
                edgecolor="#eef2f6",
                linewidth=1.0,
            )
        else:
            ax.bar(
                x,
                means,
                width=0.62,
                color=[COLOR_MAP[method] for method in methods],
                edgecolor="#eef2f6",
                linewidth=1.0,
            )
        for idx, method in enumerate(methods):
            samples = values_by_method[method]
            jitter = np.linspace(-0.11, 0.11, len(samples))
            ax.scatter(
                np.full(len(samples), idx) + jitter,
                samples,
                s=18,
                color=POINT,
                edgecolors=BG,
                linewidths=0.45,
                zorder=3,
                alpha=0.9,
            )
        ax.set_xlabel("Method")
        ax.set_ylabel(Y_LABELS[metric_name])
        ax.set_xticks(x, methods)
        style_axes(ax)
        if metric_name in {"avg_delta_z_mm", "peak_delta_z_mm", "nonplanar_layers", "z_change_events"}:
            ax.set_ylim(bottom=0.0)
        elif metric_name == "runtime_seconds":
            configure_runtime_axis(ax)
        elif metric_name == "peak_nozzle_angle_deg":
            ax.set_ylim(0.0, 38.0)

    for ax in flat_axes[len(order) :]:
        ax.axis("off")

    out_png = OUT_DIR / "contact_sheet.png"
    out_pdf = OUT_DIR / "contact_sheet.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric_name, values_by_method in SAMPLE_DATA.items():
        outputs.extend(save_metric_plot(metric_name, values_by_method))
    outputs.extend(build_contact_sheet())
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
