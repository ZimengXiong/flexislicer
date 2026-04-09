from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = ROOT / "presentation"
GENERATED_DIR = PRESENTATION_DIR / "generated"
CONFIG_PATH = PRESENTATION_DIR / "presentation_config.json"

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 36
CONTENT_W = PAGE_W - 2 * MARGIN
TOP_Y = PAGE_H - 34
BOTTOM_Y = 30

BG = colors.HexColor("#f6f4ef")
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#54606c")
ACCENT = colors.HexColor("#0f766e")
ACCENT_2 = colors.HexColor("#d97706")
PANEL = colors.HexColor("#ebe7dd")
LINE = colors.HexColor("#d1c8b8")
GOOD = colors.HexColor("#3f8f5c")
WARN = colors.HexColor("#b45309")
BAD = colors.HexColor("#b91c1c")
CARD = colors.HexColor("#fffdf8")
PLACEHOLDER = colors.HexColor("#7a3e12")
BG_HEX = "#f6f4ef"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


def load_config() -> dict:
    with CONFIG_PATH.open() as fp:
        cfg = json.load(fp)
    summary_words = len(cfg["project_summary"].split())
    if summary_words > 150:
        raise ValueError(f"Project summary exceeds 150 words ({summary_words}).")
    return cfg


def load_json(path: str) -> dict:
    with (ROOT / path).open() as fp:
        return json.load(fp)


def load_metrics() -> dict:
    flex = load_json("out_flex_benchmark_k1_vs_k2.json")
    canopy = load_json("out_canopy_planar_compare.json")
    coupon = load_json("out_openhole_fea_proxy_floor.json")

    return {
        "flex": flex,
        "canopy": canopy,
        "coupon": coupon,
        "runtime_models": [
            ("3DBenchy.stl", "Benchy"),
            ("test_sphere.stl", "Sphere"),
            ("test_cube.stl", "Cube"),
        ],
    }


def fmt_mm(value: float) -> str:
    return f"{value:.3f} mm"


def fmt_pct_ratio(value: float) -> str:
    return f"{(value - 1.0) * 100.0:+.1f}%"


def build_runtime_chart(metrics: dict) -> Path:
    out_path = GENERATED_DIR / "runtime_chart.png"
    flex = metrics["flex"]["models"]
    labels = [label for _, label in metrics["runtime_models"]]
    k1 = [flex[key]["k1"]["runtime_s"] for key, _ in metrics["runtime_models"]]
    k2 = [flex[key]["k2"]["runtime_s"] for key, _ in metrics["runtime_models"]]

    fig, ax = plt.subplots(figsize=(6.0, 3.1), constrained_layout=True)
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], k1, width=width, color="#0f766e", label="k=1")
    ax.bar([i + width / 2 for i in x], k2, width=width, color="#d97706", label="k=2")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Benchmark Runtime")
    ax.set_xticks(list(x), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    for i, value in enumerate(k1):
        ax.text(i - width / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    for i, value in enumerate(k2):
        ax.text(i + width / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    fig.savefig(out_path, dpi=200, facecolor=BG_HEX)
    plt.close(fig)
    return out_path


def build_canopy_chart(metrics: dict) -> Path:
    out_path = GENERATED_DIR / "canopy_span_chart.png"
    canopy = metrics["canopy"]
    series_names = ["Mean", "P95", "Max"]
    planar = [
        canopy["planar"]["layer_span_mean"],
        canopy["planar"]["layer_span_p95"],
        canopy["planar"]["layer_span_max"],
    ]
    k1 = [
        canopy["k1"]["layer_span_mean"],
        canopy["k1"]["layer_span_p95"],
        canopy["k1"]["layer_span_max"],
    ]
    k2 = [
        canopy["k2"]["layer_span_mean"],
        canopy["k2"]["layer_span_p95"],
        canopy["k2"]["layer_span_max"],
    ]

    fig, ax = plt.subplots(figsize=(6.1, 3.1), constrained_layout=True)
    x = range(len(series_names))
    width = 0.23
    ax.bar([i - width for i in x], planar, width=width, color="#c7c2b6", label="Planar")
    ax.bar(list(x), k1, width=width, color="#0f766e", label="k=1")
    ax.bar([i + width for i in x], k2, width=width, color="#d97706", label="k=2")
    ax.set_title("Canopy Within-Layer Z Span")
    ax.set_ylabel("Span (mm)")
    ax.set_xticks(list(x), series_names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    for offsets, values in [(-width, planar), (0.0, k1), (width, k2)]:
        for i, value in enumerate(values):
            ax.text(i + offsets, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.savefig(out_path, dpi=200, facecolor=BG_HEX)
    plt.close(fig)
    return out_path


def build_coupon_chart(metrics: dict) -> Path:
    out_path = GENERATED_DIR / "coupon_ratio_chart.png"
    ratios = metrics["coupon"]["ratios"]
    labels = [
        "Orientation",
        "Stiffness",
        "Stress conc.",
        "von Mises",
    ]
    values = [
        ratios["orient_mean_k2_over_k1"] * 100.0,
        ratios["k_eff_k2_over_k1"] * 100.0,
        ratios["kt_k2_over_k1"] * 100.0,
        ratios["vm_max_k2_over_k1"] * 100.0,
    ]
    colors_list = ["#0f766e", "#3f8f5c", "#d97706", "#b91c1c"]

    fig, ax = plt.subplots(figsize=(6.0, 3.1), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors_list)
    ax.axhline(100.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_title("Open-Hole Coupon Ratios (k=2 steered floor vs. k=1)")
    ax.set_ylabel("Percent of k=1 baseline")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.savefig(out_path, dpi=200, facecolor=BG_HEX)
    plt.close(fig)
    return out_path


def build_charts(metrics: dict) -> dict[str, Path]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "runtime": build_runtime_chart(metrics),
        "canopy": build_canopy_chart(metrics),
        "coupon": build_coupon_chart(metrics),
    }


def styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=sample["BodyText"],
        fontName="Helvetica",
        fontSize=14,
        leading=17,
        textColor=INK,
        spaceAfter=6,
    )
    body_small = ParagraphStyle(
        "BodySmall",
        parent=body,
        fontSize=12,
        leading=14,
        textColor=MUTED,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=ACCENT,
        spaceAfter=8,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=30,
        leading=34,
        textColor=INK,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=MUTED,
    )
    section = ParagraphStyle(
        "Section",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=30,
        textColor=INK,
        spaceAfter=12,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=16,
        bulletIndent=0,
        spaceAfter=8,
    )
    bullet_small = ParagraphStyle(
        "BulletSmall",
        parent=body_small,
        leftIndent=14,
        bulletIndent=0,
        spaceAfter=5,
    )
    placeholder = ParagraphStyle(
        "Placeholder",
        parent=body,
        textColor=PLACEHOLDER,
        fontName="Helvetica-Oblique",
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body_small,
        fontName="Helvetica",
        fontSize=11,
        leading=13,
        textColor=MUTED,
    )
    return {
        "body": body,
        "body_small": body_small,
        "heading": heading,
        "title": title,
        "subtitle": subtitle,
        "section": section,
        "bullet": bullet,
        "bullet_small": bullet_small,
        "placeholder": placeholder,
        "caption": caption,
    }


def is_placeholder(text: str) -> bool:
    return "Replace" in text or "[Replace" in text


def draw_background(c: canvas.Canvas, page_num: int) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    c.setFillColor(PANEL)
    c.rect(0, PAGE_H - 16, PAGE_W, 16, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(MARGIN, PAGE_H - 16, 210, 16, stroke=0, fill=1)
    c.setFillColor(PANEL)
    c.rect(0, 0, PAGE_W, 12, stroke=0, fill=1)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.line(MARGIN, BOTTOM_Y + 8, PAGE_W - MARGIN, BOTTOM_Y + 8)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawRightString(PAGE_W - MARGIN, 14, f"Page {page_num}")


def draw_section_title(c: canvas.Canvas, text: str, y: float, st: dict[str, ParagraphStyle]) -> float:
    para = Paragraph(text, st["section"])
    width, height = para.wrap(CONTENT_W, 120)
    para.drawOn(c, MARGIN, y - height)
    return y - height - 8


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    top_y: float,
    width: float,
    style: ParagraphStyle,
    bullet_text: str | None = None,
) -> float:
    para = Paragraph(text, style, bulletText=bullet_text)
    _, height = para.wrap(width, PAGE_H)
    para.drawOn(c, x, top_y - height)
    return top_y - height


def draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill_color= CARD) -> None:
    c.setFillColor(fill_color)
    c.setStrokeColor(LINE)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)


def draw_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    reader = ImageReader(str(path))
    img_w, img_h = reader.getSize()
    scale = min(w / img_w, h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2
    c.drawImage(reader, draw_x, draw_y, draw_w, draw_h, preserveAspectRatio=True, mask="auto")


def metric_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, value: str, note: str) -> None:
    draw_box(c, x, y, w, h)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 14, y + h - 18, title)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x + 14, y + h - 44, value)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 14)
    c.drawString(x + 14, y + 14, note)


def render_title_page(c: canvas.Canvas, cfg: dict, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 1)
    title_y = PAGE_H - 66
    title = Paragraph(cfg["project_title"], st["title"])
    title.wrapOn(c, CONTENT_W - 240, 120)
    title.drawOn(c, MARGIN, title_y - 88)

    names = ", ".join(cfg["student_names"])
    subtitle = Paragraph(names, st["subtitle"])
    subtitle.wrapOn(c, CONTENT_W - 240, 40)
    subtitle.drawOn(c, MARGIN, title_y - 122)

    draw_box(c, PAGE_W - 250, PAGE_H - 204, 206, 132, fill_color=colors.HexColor("#eef6f5"))
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.roundRect(PAGE_W - 216, PAGE_H - 168, 64, 44, 8, stroke=1, fill=0)
    c.roundRect(PAGE_W - 132, PAGE_H - 168, 64, 44, 8, stroke=1, fill=0)
    c.line(PAGE_W - 152, PAGE_H - 146, PAGE_W - 132, PAGE_H - 146)
    c.line(PAGE_W - 132, PAGE_H - 146, PAGE_W - 124, PAGE_H - 146)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(ACCENT)
    c.drawCentredString(PAGE_W - 184, PAGE_H - 178, "Mesh")
    c.drawCentredString(PAGE_W - 100, PAGE_H - 178, "Curved")
    c.drawCentredString(PAGE_W - 100, PAGE_H - 191, "G-code")
    c.setFont("Helvetica", 10)
    c.setFillColor(MUTED)
    c.drawCentredString(PAGE_W - 147, PAGE_H - 112, "single LS solve + warp")

    draw_box(c, MARGIN, 76, CONTENT_W, 300)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN + 18, 350, "Project Summary")
    text_style = st["body"]
    draw_paragraph(c, cfg["project_summary"], MARGIN + 18, 334, CONTENT_W - 36, text_style)
    c.showPage()


def render_intro_page(c: canvas.Canvas, cfg: dict, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 2)
    y = draw_section_title(c, "Introduction", TOP_Y, st)
    col_gap = 22
    col_w = (CONTENT_W - col_gap) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + col_gap

    left_y = y
    right_y = y

    left_y = draw_paragraph(c, "Research Question or Engineering Goal", left_x, left_y, col_w, st["heading"]) - 4
    left_y = draw_paragraph(c, cfg["research_question"], left_x, left_y, col_w, st["body"]) - 12
    left_y = draw_paragraph(c, "Continuation", left_x, left_y, col_w, st["heading"]) - 4
    cont_style = st["placeholder"] if is_placeholder(cfg["continuation"]) else st["body"]
    left_y = draw_paragraph(c, cfg["continuation"], left_x, left_y, col_w, cont_style)

    right_y = draw_paragraph(c, "Project Origin", right_x, right_y, col_w, st["heading"]) - 4
    origin_style = st["placeholder"] if is_placeholder(cfg["project_origin"]) else st["body"]
    right_y = draw_paragraph(c, cfg["project_origin"], right_x, right_y, col_w, origin_style) - 12
    right_y = draw_paragraph(c, "Work by Others", right_x, right_y, col_w, st["heading"]) - 4
    right_y = draw_paragraph(c, cfg["work_by_others"], right_x, right_y, col_w, st["body"])
    c.showPage()


def render_framework_page(c: canvas.Canvas, metrics: dict, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 3)
    y = draw_section_title(c, "Framework", TOP_Y, st)
    col_gap = 24
    left_w = 342
    right_w = CONTENT_W - left_w - col_gap
    left_x = MARGIN
    right_x = MARGIN + left_w + col_gap

    left_y = y
    left_y = draw_paragraph(c, "Key concepts and notation", left_x, left_y, left_w, st["heading"]) - 4
    left_y = draw_paragraph(
        c,
        "The implementation models the slicing surface as a height field S(x, y) above the print bed. "
        "A mesh is sampled by vertical rays on an XY grid. The top-surface slope map determines a target mask "
        "Theta: regions at or below theta_target are preferred for curved reproduction, while a stricter theta_max "
        "constraint is enforced during post-processing to avoid nozzle gouging.",
        left_x,
        left_y,
        left_w,
        st["body"],
    ) - 8
    left_y = draw_paragraph(c, "Pipeline used in this repository", left_x, left_y, left_w, st["heading"]) - 2
    steps = [
        "Sample the watertight mesh to a 2D grid and record top Z values.",
        "Build the slope mask and optionally clean it with morphological filtering.",
        "Solve one least-squares system balancing gradient steepening, boundary fidelity, smoothness, and component regularization.",
        "Apply a conical-slope post-process, slice contours, and save arrays and metadata.",
        "Optionally export a deformed STL, call PrusaSlicer, and warp planar G-code back onto the non-planar surface.",
    ]
    for item in steps:
        left_y = draw_paragraph(c, item, left_x, left_y, left_w, st["bullet"], bullet_text="-") - 2

    draw_box(c, right_x, 106, right_w, 392)
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(right_x + 16, 474, "Algorithm Flow")

    box_x = right_x + 18
    box_w = right_w - 36
    box_h = 48
    box_y = 404
    labels = [
        "Mesh input + grid sampling",
        "Slope map + Theta mask",
        "Single LS surface solve",
        "Post-process + layer extraction",
        "Deformed STL / warped G-code",
    ]
    for idx, label in enumerate(labels):
        current_y = box_y - idx * 66
        c.setFillColor(colors.HexColor("#f8faf9"))
        c.setStrokeColor(LINE)
        c.roundRect(box_x, current_y, box_w, box_h, 8, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(box_x + 14, current_y + 18, label)
        if idx < len(labels) - 1:
            c.setStrokeColor(ACCENT)
            c.setLineWidth(2)
            c.line(box_x + box_w / 2, current_y, box_x + box_w / 2, current_y - 18)
            c.line(box_x + box_w / 2, current_y - 18, box_x + box_w / 2 - 4, current_y - 14)
            c.line(box_x + box_w / 2, current_y - 18, box_x + box_w / 2 + 4, current_y - 14)

    c.setFont("Helvetica", 11)
    c.setFillColor(MUTED)
    cfg = metrics["flex"]["config"]
    c.drawString(
        right_x + 16,
        126,
        f"Example benchmark config: grid_step={cfg['grid_step']:.1f} mm, layer_height={cfg['layer_height']:.1f} mm, "
        f"theta_target={cfg['theta_target_deg']:.0f} deg, theta_max={cfg['theta_max_deg']:.0f} deg",
    )
    c.showPage()


def render_findings_page_one(c: canvas.Canvas, metrics: dict, charts: dict[str, Path], st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 4)
    y = draw_section_title(c, "Findings", TOP_Y, st)
    left_w = 300
    right_w = CONTENT_W - left_w - 20
    left_x = MARGIN
    right_x = MARGIN + left_w + 20

    flex = metrics["flex"]
    benchy_ratio = flex["ratios"]["3DBenchy.stl"]["runtime_ratio_k2_over_k1"]
    sphere_ratio = flex["ratios"]["test_sphere.stl"]["runtime_ratio_k2_over_k1"]
    benchy_sigs = flex["ratios"]["3DBenchy.stl"]["unique_signature_ratio_k2_over_k1"]
    sphere_sigs = flex["ratios"]["test_sphere.stl"]["unique_signature_ratio_k2_over_k1"]

    left_y = y
    left_y = draw_paragraph(c, "Observed behavior from repository benchmarks", left_x, left_y, left_w, st["heading"]) - 4
    bullets = [
        "Solve times remained low on the tested models: about 0.14 s for the sphere, 0.20 s for the cube, and 1.5 to 1.6 s for Benchy.",
        f"Moving from k=1 to k=2 changed runtime only modestly: Benchy {fmt_pct_ratio(benchy_ratio)}, Sphere {fmt_pct_ratio(sphere_ratio)}, Cube {-3.5:+.1f}%.",
        f"The k=2 variant produced more distinct path signatures than k=1: Benchy {benchy_sigs:.1f}x and Sphere {sphere_sigs:.1f}x.",
    ]
    for item in bullets:
        left_y = draw_paragraph(c, item, left_x, left_y, left_w, st["bullet"], bullet_text="-") - 2

    card_y = 118
    metric_card(c, left_x, card_y + 110, 138, 92, "Benchy", "1.53-1.59 s", "k=1 vs. k=2")
    metric_card(c, left_x + 154, card_y + 110, 138, 92, "Sphere", "0.135 s", "small curved case")
    metric_card(c, left_x, card_y, 138, 92, "Cube", "0.203 s", "flat control")
    metric_card(c, left_x + 154, card_y, 138, 92, "Layer 1", "0.200 mm", "fixed in outputs")

    draw_box(c, right_x, 126, right_w, 372)
    draw_image(c, charts["runtime"], right_x + 10, 166, right_w - 20, 300)
    draw_paragraph(
        c,
        "Runtime chart built from out_flex_benchmark_k1_vs_k2.json. The low spread between k=1 and k=2 suggests that the steering change is not dominated by additional solve cost on these small meshes.",
        right_x + 14,
        154,
        right_w - 28,
        st["caption"],
    )
    c.showPage()


def render_findings_page_two(c: canvas.Canvas, metrics: dict, charts: dict[str, Path], st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 5)
    y = draw_section_title(c, "Findings", TOP_Y, st)
    canopy = metrics["canopy"]
    coupon = metrics["coupon"]
    left_x = MARGIN
    right_x = MARGIN + CONTENT_W / 2 + 8
    panel_w = CONTENT_W / 2 - 12

    draw_box(c, left_x, 92, panel_w, 420)
    draw_paragraph(c, "Surface-following curvature on the canopy case", left_x + 14, y, panel_w - 28, st["heading"])
    draw_image(c, charts["canopy"], left_x + 10, 216, panel_w - 20, 236)
    canopy_text = (
        f"Compared with planar slicing, the canopy output introduces real within-layer curvature. "
        f"The mean span was {fmt_mm(canopy['k1']['layer_span_mean'])} for k=1 and {fmt_mm(canopy['k2']['layer_span_mean'])} for k=2. "
        f"The k=2 maximum span reached {fmt_mm(canopy['k2']['layer_span_max'])}, showing a slightly stronger non-planar effect while keeping the first layer fixed at 0.2 mm."
    )
    draw_paragraph(c, canopy_text, left_x + 14, 198, panel_w - 28, st["body"])

    draw_box(c, right_x, 92, panel_w, 420)
    draw_paragraph(c, "Tradeoff in the open-hole coupon proxy", right_x + 14, y, panel_w - 28, st["heading"])
    draw_image(c, charts["coupon"], right_x + 10, 216, panel_w - 20, 236)
    coupon_text = (
        f"In out_openhole_fea_proxy_floor.json, the steered k=2 floor variant raised orientation strength by {fmt_pct_ratio(coupon['ratios']['orient_mean_k2_over_k1'])} "
        f"and stiffness proxy by {fmt_pct_ratio(coupon['ratios']['k_eff_k2_over_k1'])}. The same change also increased stress concentration by {fmt_pct_ratio(coupon['ratios']['kt_k2_over_k1'])} "
        f"and max von Mises proxy by {fmt_pct_ratio(coupon['ratios']['vm_max_k2_over_k1'])}. Median endpoint shift stayed near 0.099 mm, so the structural tradeoff came from steering, not a large path rewrite."
    )
    draw_paragraph(c, coupon_text, right_x + 14, 198, panel_w - 28, st["body"])
    c.showPage()


def render_conclusions_page(c: canvas.Canvas, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 6)
    y = draw_section_title(c, "Conclusions", TOP_Y, st)
    col_gap = 24
    col_w = (CONTENT_W - col_gap) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + col_gap

    left_y = y
    left_y = draw_paragraph(c, "Assessment of findings", left_x, left_y, col_w, st["heading"]) - 4
    for item in [
        "The repository supports the main engineering goal: a compact Python workflow can produce collision-aware, slightly non-planar outputs for standard 3-axis FFF printers.",
        "The strongest evidence is computational efficiency and workflow completeness: the pipeline goes from mesh sampling to optional warped G-code without volumetric tetrahedral optimization.",
        "The canopy and coupon outputs show that parameter changes measurably affect curvature, path orientation, and proxy performance metrics.",
    ]:
        left_y = draw_paragraph(c, item, left_x, left_y, col_w, st["bullet"], bullet_text="-") - 2

    right_y = y
    right_y = draw_paragraph(c, "Limitations and next steps", right_x, right_y, col_w, st["heading"]) - 4
    for item in [
        "This draft relies on repository artifacts and proxy metrics; it does not yet include measured print-quality data such as roughness, dimensional error, or judge-visible photographs from this year's work.",
        "The current implementation assumes watertight meshes and uses PrusaSlicer as the planar slicing backend for the end-to-end path.",
        "Next steps: add physical print comparisons against planar baselines, quantify surface-finish improvement directly, and restrict expensive orientation optimization to visible cover layers only.",
    ]:
        right_y = draw_paragraph(c, item, right_x, right_y, col_w, st["bullet"], bullet_text="-") - 2

    metric_card(c, left_x, 86, 210, 92, "Primary result", "End-to-end pipeline", "mesh -> warped G-code")
    metric_card(c, left_x + 228, 86, 210, 92, "Main caution", "Needs print validation", "proxy metrics only")
    c.showPage()


def render_scope_page(c: canvas.Canvas, cfg: dict, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 7)
    y = draw_section_title(c, "Scope of Work", TOP_Y, st)
    col_gap = 24
    col_w = (CONTENT_W - col_gap) / 2
    left_x = MARGIN
    right_x = MARGIN + col_w + col_gap

    left_y = y
    left_y = draw_paragraph(c, "New Work by Author(s)", left_x, left_y, col_w, st["heading"]) - 2
    for item in cfg["new_work_by_author"]:
        left_y = draw_paragraph(c, item, left_x, left_y, col_w, st["bullet"], bullet_text="-") - 1

    right_y = y
    right_y = draw_paragraph(c, "Professional, Institutional, and Academic Resources and Support", right_x, right_y, col_w, st["heading"]) - 2
    for item in cfg["resources_and_support"]:
        style = st["placeholder"] if is_placeholder(item) else st["bullet"]
        bullet = "-" if style is st["bullet"] else None
        right_y = draw_paragraph(c, item, right_x, right_y, col_w, style, bullet_text=bullet) - 1
    c.showPage()


def render_references_page(c: canvas.Canvas, st: dict[str, ParagraphStyle]) -> None:
    draw_background(c, 8)
    y = draw_section_title(c, "References / Supplemental Information", TOP_Y, st)
    refs = [
        "Ottonello, E., Hugron, P.-A., Parmiggiani, A., and Lefebvre, S. QuickCurve: revisiting slightly non-planar 3D printing. arXiv:2406.03966, June 6, 2024.",
        "Ahlers, D., Wasserfall, F., Hendrich, N., and Zhang, J. 3D printing of nonplanar layers for smooth surface generation. IEEE CASE, 2019.",
        "Etienne, J., Ray, N., Panozzo, D., Hornus, S., Wang, C. C. L., Martinez, J., McMains, S., Alexa, M., Wyvill, B., and Lefebvre, S. CurviSlicer: slightly curved slicing for 3-axis printers. ACM TOG, 2019.",
        "Prusa Research. PrusaSlicer documentation and CLI workflow. Used here as the planar slicing backend before G-code warping.",
        "Supplemental information available from the project working directory and generated benchmark JSON outputs used in this draft.",
    ]
    current_y = y
    for ref in refs:
        current_y = draw_paragraph(c, ref, MARGIN, current_y, CONTENT_W, st["bullet"], bullet_text="-") - 2
    c.showPage()


def add_fit_page_open_action(pdf_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    first_ref = writer.pages[0].indirect_reference
    writer._root_object.update(  # noqa: SLF001
        {
            NameObject("/OpenAction"): ArrayObject([first_ref, NameObject("/Fit")]),
        }
    )
    with pdf_path.open("wb") as fp:
        writer.write(fp)


def validate_pdf(pdf_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) > 13:
        raise ValueError(f"Presentation has {len(reader.pages)} pages; limit is 13.")
    for idx, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if width <= height:
            raise ValueError(f"Page {idx} is not landscape.")
        if round(width, 1) != round(PAGE_W, 1) or round(height, 1) != round(PAGE_H, 1):
            raise ValueError(f"Page {idx} does not match landscape letter size.")


def build_pdf(cfg: dict, metrics: dict, charts: dict[str, Path]) -> Path:
    out_path = GENERATED_DIR / "flexislicer_csef_presentation_draft.pdf"
    c = canvas.Canvas(str(out_path), pagesize=landscape(letter), pageCompression=1)
    st = styles()

    render_title_page(c, cfg, st)
    render_intro_page(c, cfg, st)
    render_framework_page(c, metrics, st)
    render_findings_page_one(c, metrics, charts, st)
    render_findings_page_two(c, metrics, charts, st)
    render_conclusions_page(c, st)
    render_scope_page(c, cfg, st)
    render_references_page(c, st)
    c.save()
    add_fit_page_open_action(out_path)
    validate_pdf(out_path)
    return out_path


def main() -> None:
    cfg = load_config()
    metrics = load_metrics()
    charts = build_charts(metrics)
    pdf_path = build_pdf(cfg, metrics, charts)
    print(pdf_path)


if __name__ == "__main__":
    main()
