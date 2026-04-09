from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation" / "generated" / "findings_page2"

BG = "#ffffff"
GRID = "#d9dde3"
INK = "#1f2933"
PLANAR = "#d5d9df"
FLEX = "#0f766e"
FLEX_COOL = "#67b99a"

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
        "legend.fontsize": 10,
    }
)


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


def build_surface_scan_chart() -> list[Path]:
    regions = ["Average terrace", "Average overhang"]
    planar = [5.5, 3.8]
    flex = [2.1, 3.5]
    flex_cool = [2.0, 3.2]

    x = np.arange(len(regions))
    width = 0.23

    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    ax.bar(x - width, planar, width=width, color=PLANAR, label="Planar")
    ax.bar(x, flex, width=width, color=FLEX, label="FlexiSlicer")
    ax.bar(x + width, flex_cool, width=width, color=FLEX_COOL, label="FlexiSlicer + high fan")
    ax.set_xlabel("Scanned feature class")
    ax.set_ylabel("Areal roughness Sa (um)")
    ax.set_xticks(x, regions)
    ax.set_ylim(0, 6.2)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    out_png = OUT_DIR / "surface_scan_sa.png"
    out_pdf = OUT_DIR / "surface_scan_sa.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def build_print_time_chart() -> list[Path]:
    models = ["Shallow ramp", "Benchy roof", "Curved bracket", "Rabbit-ear arch"]
    planar = [118, 132, 74, 96]
    flex = [84, 101, 57, 73]

    x = np.arange(len(models))
    width = 0.33

    fig, ax = plt.subplots(figsize=(5.3, 3.2), constrained_layout=True)
    ax.bar(x - width / 2, planar, width=width, color=PLANAR, label="Planar")
    ax.bar(x + width / 2, flex, width=width, color=FLEX, label="FlexiSlicer")
    ax.set_xlabel("Model")
    ax.set_ylabel("Estimated print time (min)")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 145)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    out_png = OUT_DIR / "print_time_minutes.png"
    out_pdf = OUT_DIR / "print_time_minutes.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def build_combo_panel() -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), constrained_layout=True)

    regions = ["Average terrace", "Average overhang"]
    planar_scan = [5.5, 3.8]
    flex_scan = [2.1, 3.5]
    flex_cool = [2.0, 3.2]
    x = np.arange(len(regions))
    width = 0.23
    ax = axes[0]
    ax.bar(x - width, planar_scan, width=width, color=PLANAR, label="Planar")
    ax.bar(x, flex_scan, width=width, color=FLEX, label="FlexiSlicer")
    ax.bar(x + width, flex_cool, width=width, color=FLEX_COOL, label="FlexiSlicer + high fan")
    ax.set_xlabel("Scanned feature class")
    ax.set_ylabel("Areal roughness Sa (um)")
    ax.set_xticks(x, regions)
    ax.set_ylim(0, 6.2)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    models = ["Shallow ramp", "Benchy roof", "Curved bracket", "Rabbit-ear arch"]
    planar_time = [118, 132, 74, 96]
    flex_time = [84, 101, 57, 73]
    x = np.arange(len(models))
    width = 0.33
    ax = axes[1]
    ax.bar(x - width / 2, planar_time, width=width, color=PLANAR, label="Planar")
    ax.bar(x + width / 2, flex_time, width=width, color=FLEX, label="FlexiSlicer")
    ax.set_xlabel("Model")
    ax.set_ylabel("Estimated print time (min)")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 145)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")

    out_png = OUT_DIR / "findings_page2_combo.png"
    out_pdf = OUT_DIR / "findings_page2_combo.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [out_png, out_pdf]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(build_surface_scan_chart())
    outputs.extend(build_print_time_chart())
    outputs.extend(build_combo_panel())
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
