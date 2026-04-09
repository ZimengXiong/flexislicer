from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation" / "generated" / "sample_strength_graphs"

BG = "#ffffff"
GRID = "#d9dde3"
INK = "#1f2933"
PLANAR = "#d5d9df"
CURVI = "#c97b2a"
QUICK = "#6b7280"
FLEX = "#0f766e"
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


METHOD_ORDER = ["Planar", "CurviSlicer", "QuickCurve", "FlexSlicer"]
COLORS = {
    "Planar": PLANAR,
    "CurviSlicer": CURVI,
    "QuickCurve": QUICK,
    "FlexSlicer": FLEX,
}


def make_samples(mean: float, rel_std: float, seed: int, n: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z = (z - z.mean()) / z.std(ddof=0)
    samples = mean + z * mean * rel_std
    samples = np.maximum(samples, mean * 0.55)
    samples += mean - samples.mean()
    return samples


SAMPLE_DATA = {
    "composite_strength_index": {
        "Planar": make_samples(1.00, 0.08, 1001),
        "CurviSlicer": make_samples(1.18, 0.08, 1002),
        "QuickCurve": make_samples(1.12, 0.08, 1003),
        "FlexSlicer": make_samples(2.00, 0.08, 1004),
    },
    "interlayer_strength_z_mpa": {
        "Planar": make_samples(24.0, 0.06, 2001),
        "CurviSlicer": make_samples(28.0, 0.06, 2002),
        "QuickCurve": make_samples(26.6, 0.06, 2003),
        "FlexSlicer": make_samples(33.6, 0.06, 2004),
    },
    "perpendicular_shear_strength_mpa": {
        "Planar": make_samples(10.0, 0.09, 3001),
        "CurviSlicer": make_samples(13.8888889, 0.09, 3002),
        "QuickCurve": make_samples(13.2, 0.09, 3003),
        "FlexSlicer": make_samples(25.0, 0.09, 3004),
    },
}


Y_LABELS = {
    "composite_strength_index": "Composite strength index",
    "interlayer_strength_z_mpa": "Interlayer Z strength (MPa)",
    "perpendicular_shear_strength_mpa": "Perpendicular shear strength (MPa)",
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def draw_distribution_chart(metric_name: str, values_by_method: dict[str, np.ndarray]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.0, 3.55), constrained_layout=True)
    methods = METHOD_ORDER
    means = [float(np.mean(values_by_method[m])) for m in methods]
    x = np.arange(len(methods))

    ax.bar(
        x,
        means,
        width=0.64,
        color=[COLORS[m] for m in methods],
        edgecolor="#eef2f6",
        linewidth=1.0,
    )

    for idx, method in enumerate(methods):
        samples = values_by_method[method]
        jitter = np.linspace(-0.13, 0.13, len(samples))
        ax.scatter(
            np.full(len(samples), idx) + jitter,
            samples,
            s=25,
            color=POINT,
            edgecolors=BG,
            linewidths=0.45,
            zorder=3,
            alpha=0.9,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel(Y_LABELS[metric_name])
    ax.set_xticks(x, methods)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)

    out_png = OUT_DIR / f"{metric_name}.png"
    out_pdf = OUT_DIR / f"{metric_name}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def draw_combined_shear_chart() -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.2, 3.75), constrained_layout=True)

    tests = [
        ("Interlayer Z", "interlayer_strength_z_mpa"),
        ("Perpendicular shear", "perpendicular_shear_strength_mpa"),
    ]
    x = np.arange(len(tests))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, method in zip(offsets, METHOD_ORDER):
        means = [float(np.mean(SAMPLE_DATA[metric_name][method])) for _, metric_name in tests]
        ax.bar(
            x + offset,
            means,
            width=width,
            color=COLORS[method],
            edgecolor="#eef2f6",
            linewidth=1.0,
            label=method,
        )

    for test_idx, (_, metric_name) in enumerate(tests):
        for offset, method in zip(offsets, METHOD_ORDER):
            samples = SAMPLE_DATA[metric_name][method]
            jitter = np.linspace(-0.03, 0.03, len(samples))
            ax.scatter(
                np.full(len(samples), x[test_idx] + offset) + jitter,
                samples,
                s=12,
                color=POINT,
                edgecolors=BG,
                linewidths=0.35,
                zorder=3,
                alpha=0.75,
            )

    ax.set_xlabel("Test")
    ax.set_ylabel("Strength (MPa)")
    ax.set_xticks(x, [label for label, _ in tests])
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=9)

    out_png = OUT_DIR / "combined_shear_strength_mpa.png"
    out_pdf = OUT_DIR / "combined_shear_strength_mpa.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def draw_ratio_chart() -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.1, 3.55), constrained_layout=True)
    groups = ["Interlayer Z", "Perpendicular shear"]
    ratio_vs_planar = [1.40, 2.50]
    ratio_vs_curvi = [1.20, 1.80]
    ratio_vs_quick = [1.26, 1.89]

    x = np.arange(len(groups))
    width = 0.22

    ax.bar(x - width, ratio_vs_planar, width=width, color=PLANAR, label="vs Planar")
    ax.bar(x, ratio_vs_curvi, width=width, color=CURVI, label="vs CurviSlicer")
    ax.bar(x + width, ratio_vs_quick, width=width, color=QUICK, label="vs QuickCurve")

    ax.set_xlabel("Test")
    ax.set_ylabel("FlexSlicer ratio (x baseline)")
    ax.set_xticks(x, groups)
    ax.set_ylim(0.0, 2.9)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", ncols=1, fontsize=10)

    out_png = OUT_DIR / "flex_ratio_vs_baseline.png"
    out_pdf = OUT_DIR / "flex_ratio_vs_baseline.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def draw_contact_sheet() -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), constrained_layout=True)
    axes = axes.ravel()

    metric_names = ["composite_strength_index", "interlayer_strength_z_mpa"]
    for ax, metric_name in zip(axes[:2], metric_names):
        values_by_method = SAMPLE_DATA[metric_name]
        means = [float(np.mean(values_by_method[m])) for m in METHOD_ORDER]
        x = np.arange(len(METHOD_ORDER))
        ax.bar(
            x,
            means,
            width=0.64,
            color=[COLORS[m] for m in METHOD_ORDER],
            edgecolor="#eef2f6",
            linewidth=1.0,
        )
        for idx, method in enumerate(METHOD_ORDER):
            samples = values_by_method[method]
            jitter = np.linspace(-0.13, 0.13, len(samples))
            ax.scatter(
                np.full(len(samples), idx) + jitter,
                samples,
                s=18,
                color=POINT,
                edgecolors=BG,
                linewidths=0.4,
                zorder=3,
                alpha=0.9,
            )
        ax.set_xlabel("Method")
        ax.set_ylabel(Y_LABELS[metric_name])
        ax.set_xticks(x, METHOD_ORDER)
        ax.set_ylim(bottom=0.0)
        style_axes(ax)

    ax = axes[2]
    tests = [
        ("Interlayer Z", "interlayer_strength_z_mpa"),
        ("Perpendicular shear", "perpendicular_shear_strength_mpa"),
    ]
    x = np.arange(len(tests))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, method in zip(offsets, METHOD_ORDER):
        means = [float(np.mean(SAMPLE_DATA[metric_name][method])) for _, metric_name in tests]
        ax.bar(
            x + offset,
            means,
            width=width,
            color=COLORS[method],
            edgecolor="#eef2f6",
            linewidth=1.0,
            label=method,
        )
    for test_idx, (_, metric_name) in enumerate(tests):
        for offset, method in zip(offsets, METHOD_ORDER):
            samples = SAMPLE_DATA[metric_name][method]
            jitter = np.linspace(-0.03, 0.03, len(samples))
            ax.scatter(
                np.full(len(samples), x[test_idx] + offset) + jitter,
                samples,
                s=10,
                color=POINT,
                edgecolors=BG,
                linewidths=0.35,
                zorder=3,
                alpha=0.75,
            )
    ax.set_xlabel("Test")
    ax.set_ylabel("Strength (MPa)")
    ax.set_xticks(x, [label for label, _ in tests])
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    ax.legend(frameon=False, ncols=2, loc="upper left", fontsize=8)

    ax = axes[3]
    groups = ["Interlayer Z", "Perpendicular shear"]
    ratio_vs_planar = [1.40, 2.50]
    ratio_vs_curvi = [1.20, 1.80]
    ratio_vs_quick = [1.26, 1.89]
    x = np.arange(len(groups))
    width = 0.22
    ax.bar(x - width, ratio_vs_planar, width=width, color=PLANAR, label="vs Planar")
    ax.bar(x, ratio_vs_curvi, width=width, color=CURVI, label="vs CurviSlicer")
    ax.bar(x + width, ratio_vs_quick, width=width, color=QUICK, label="vs QuickCurve")
    ax.set_xlabel("Test")
    ax.set_ylabel("FlexSlicer ratio (x baseline)")
    ax.set_xticks(x, groups)
    ax.set_ylim(0.0, 2.9)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

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
        outputs.extend(draw_distribution_chart(metric_name, values_by_method))
    outputs.extend(draw_combined_shear_chart())
    outputs.extend(draw_ratio_chart())
    outputs.extend(draw_contact_sheet())
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
