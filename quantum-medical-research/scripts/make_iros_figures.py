#!/usr/bin/env python3
"""Generate publication-ready IROS ELSR figures without Matplotlib."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "iros_elsr_2026"

KVASIR_STATS = ROOT / "endoscopy_guidance" / "results" / "residual_gain_kvasir_hard_enriched_300_paired_stats.csv"
CVC_STATS = ROOT / "endoscopy_guidance" / "results" / "residual_gain_cvc_external_r48_paired_stats.csv"
RESCUE_MANIFEST = ROOT / "docs" / "figures" / "kvasir_hard_enriched_switch_examples" / "manifest.csv"

FIG_W = 7.16 * inch
PIPE_H = 2.20 * inch
RESULT_H = 2.62 * inch

BLUE = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2a9d8f")
ORANGE = colors.HexColor("#e76f51")
GRAY = colors.HexColor("#52616b")
INK = colors.HexColor("#1d2733")
LIGHT_BLUE = colors.HexColor("#eaf3fb")
LIGHT_TEAL = colors.HexColor("#e9f6f3")
LIGHT_ORANGE = colors.HexColor("#fff1eb")
PALE_GRAY = colors.HexColor("#f5f7fa")
GRID = colors.HexColor("#d4dde6")


def register_fonts() -> tuple[str, str]:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    regular = font_dir / "Times New Roman.ttf"
    bold = font_dir / "Times New Roman Bold.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("TimesNewRoman", str(regular)))
        pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(bold)))
        return "TimesNewRoman", "TimesNewRoman-Bold"
    return "Times-Roman", "Times-Bold"


FONT, FONT_BOLD = register_fonts()


def load_row(path: Path, policy: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["policy"] == policy:
                return row
    raise ValueError(f"Policy {policy!r} not found in {path}")


def load_rescue() -> dict[str, str]:
    with RESCUE_MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # Prefer a visually clean example for paper/demo figures: high final Dice
    # and positive rescue gain are easier to interpret than the largest gain.
    return max(rows, key=lambda row: (float(row["sam_dice"]), float(row["delta_vs_unet"])))


def text(c: canvas.Canvas, x: float, y: float, msg: str, size: float, bold: bool = False, color=INK, center=False):
    c.setFillColor(color)
    c.setFont(FONT_BOLD if bold else FONT, size)
    if center:
        c.drawCentredString(x, y, msg)
    else:
        c.drawString(x, y, msg)


def multiline_center(c: canvas.Canvas, x: float, y: float, lines: list[str], size: float, bold: bool = True):
    leading = size * 1.18
    start = y + (len(lines) - 1) * leading / 2
    for i, line in enumerate(lines):
        text(c, x, start - i * leading, line, size, bold=bold, center=True)


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, lines: list[str], fill, stroke, size: float = 8.4):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    multiline_center(c, x + w / 2, y + h / 2 - 2, lines, size)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=GRAY):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.15)
    c.line(x1, y1, x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    s = 5.0
    pts = [
        (x2, y2),
        (x2 - s * ux + 0.55 * s * px, y2 - s * uy + 0.55 * s * py),
        (x2 - s * ux - 0.55 * s * px, y2 - s * uy - 0.55 * s * py),
    ]
    path = c.beginPath()
    path.moveTo(*pts[0])
    path.lineTo(*pts[1])
    path.lineTo(*pts[2])
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def save_metadata(c: canvas.Canvas) -> None:
    c.setAuthor("FAMS Lab")
    c.setCreator("FAMS Lab")
    c.setProducer("FAMS Lab")
    c.setSubject("IROS ELSR 2026 figure")


def make_pipeline() -> None:
    out = OUT / "fig1_quantum_guided_second_opinion_pipeline.pdf"
    c = canvas.Canvas(str(out), pagesize=(FIG_W, PIPE_H))
    save_metadata(c)
    c.setTitle("Quantum-guided second-opinion pipeline")

    y_mid = PIPE_H / 2
    box(c, 0.08 * inch, y_mid - 0.22 * inch, 0.82 * inch, 0.44 * inch, ["Endoscopic", "frame"], colors.white, colors.HexColor("#9aa7b4"))
    box(c, 1.25 * inch, 1.22 * inch, 0.98 * inch, 0.42 * inch, ["Fast default", "UNet"], LIGHT_BLUE, BLUE)
    box(c, 1.25 * inch, 0.48 * inch, 0.98 * inch, 0.42 * inch, ["Uncertainty", "features"], PALE_GRAY, colors.HexColor("#9aa7b4"))
    box(c, 2.70 * inch, y_mid - 0.22 * inch, 1.02 * inch, 0.44 * inch, ["Hard-case", "router"], PALE_GRAY, GRAY)
    box(c, 4.26 * inch, 1.22 * inch, 1.18 * inch, 0.42 * inch, ["Quantum-guided", "prompt diversity"], LIGHT_ORANGE, ORANGE, size=7.9)
    box(c, 4.26 * inch, 0.48 * inch, 1.18 * inch, 0.42 * inch, ["SAM second", "opinion masks"], LIGHT_TEAL, TEAL)
    box(c, 6.25 * inch, y_mid - 0.22 * inch, 0.78 * inch, 0.44 * inch, ["Final", "mask"], colors.white, colors.HexColor("#9aa7b4"))

    arrow(c, 0.90 * inch, y_mid, 1.25 * inch, 1.43 * inch)
    arrow(c, 0.90 * inch, y_mid, 1.25 * inch, 0.69 * inch)
    arrow(c, 2.23 * inch, 1.43 * inch, 2.70 * inch, 1.17 * inch)
    arrow(c, 2.23 * inch, 0.69 * inch, 2.70 * inch, 1.03 * inch)
    arrow(c, 3.72 * inch, y_mid, 4.26 * inch, 1.43 * inch, ORANGE)
    arrow(c, 4.85 * inch, 1.22 * inch, 4.85 * inch, 0.90 * inch, ORANGE)
    arrow(c, 5.44 * inch, 0.69 * inch, 6.25 * inch, 1.02 * inch, TEAL)
    arrow(c, 3.72 * inch, y_mid, 6.25 * inch, 1.18 * inch, BLUE)

    text(c, 1.72 * inch, 1.78 * inch, "default fast path", 7.6, color=BLUE, center=True)
    text(c, 4.85 * inch, 1.78 * inch, "activated only for difficult frames", 7.6, color=ORANGE, center=True)
    text(c, 4.85 * inch, 0.16 * inch, "alternate hypotheses ranked by predicted residual gain", 7.6, color=TEAL, center=True)
    c.showPage()
    c.save()


def draw_bar(c, x, y0, width, height, value, max_value, color):
    bar_h = max(0, value / max_value) * height
    c.setFillColor(color)
    c.rect(x, y0, width, bar_h, fill=1, stroke=0)
    text(c, x + width / 2, y0 + bar_h + 5, f"+{value:.3f}", 6.8, bold=True, color=INK, center=True)


def make_results_and_rescue() -> None:
    k_switch = load_row(KVASIR_STATS, "classical_histgb:absolute_sam_dice:meta_rf_gain_switch")
    k_oracle = load_row(KVASIR_STATS, "classical_histgb:absolute_sam_dice:oracle_best_of_two")
    c_switch = load_row(CVC_STATS, "classical_histgb:residual_gain:val_gain_switch")
    rescue = load_rescue()

    groups = [
        ("Kvasir\nlearned", float(k_switch["delta_vs_unet"]), float(k_switch["hard_delta_vs_unet"])),
        ("Kvasir\noracle", float(k_oracle["delta_vs_unet"]), float(k_oracle["hard_delta_vs_unet"])),
        ("CVC\nlearned", float(c_switch["delta_vs_unet"]), float(c_switch["hard_delta_vs_unet"])),
    ]

    out = OUT / "fig2_results_and_hard_case_rescue.pdf"
    c = canvas.Canvas(str(out), pagesize=(FIG_W, RESULT_H))
    save_metadata(c)
    c.setTitle("Second-opinion segmentation results and rescue example")

    text(c, 0.08 * inch, RESULT_H - 0.25 * inch, "A. Selective second-opinion gains", 9.2, bold=True)
    chart_x = 0.18 * inch
    chart_y = 0.58 * inch
    chart_w = 2.30 * inch
    chart_h = 1.52 * inch
    max_v = 0.28
    c.setStrokeColor(GRID)
    c.setLineWidth(0.45)
    for tick in [0.0, 0.07, 0.14, 0.21, 0.28]:
        y = chart_y + tick / max_v * chart_h
        c.line(chart_x, y, chart_x + chart_w, y)
        text(c, chart_x - 0.06 * inch, y - 2, f"{tick:.2f}", 6.6, color=GRAY, center=True)
    c.setStrokeColor(INK)
    c.line(chart_x, chart_y, chart_x + chart_w, chart_y)
    c.line(chart_x, chart_y, chart_x, chart_y + chart_h)
    text(c, chart_x + chart_w / 2, 0.19 * inch, "Dice gain over UNet", 7.6, color=GRAY, center=True)

    group_w = chart_w / 3
    bar_w = 0.14 * inch
    for i, (label, overall, hard) in enumerate(groups):
        center = chart_x + group_w * (i + 0.5)
        draw_bar(c, center - bar_w - 2, chart_y, bar_w, chart_h, overall, max_v, BLUE)
        draw_bar(c, center + 2, chart_y, bar_w, chart_h, hard, max_v, ORANGE)
        l1, l2 = label.split("\n")
        text(c, center, chart_y - 0.15 * inch, l1, 6.8, color=INK, center=True)
        text(c, center, chart_y - 0.27 * inch, l2, 6.8, color=INK, center=True)

    legend_y = RESULT_H - 0.47 * inch
    c.setFillColor(BLUE)
    c.rect(0.26 * inch, legend_y, 0.09 * inch, 0.07 * inch, fill=1, stroke=0)
    text(c, 0.39 * inch, legend_y - 1, "Overall", 7.3)
    c.setFillColor(ORANGE)
    c.rect(0.86 * inch, legend_y, 0.09 * inch, 0.07 * inch, fill=1, stroke=0)
    text(c, 0.99 * inch, legend_y - 1, "Hard frames", 7.3)

    text(c, 2.92 * inch, RESULT_H - 0.25 * inch, "B. Held-out hard-frame rescue", 9.2, bold=True)
    rescue_path = ROOT / rescue["figure"]
    img = Image.open(rescue_path)
    img_w, img_h = img.size
    target_w = 4.05 * inch
    target_h = target_w * img_h / img_w
    c.drawImage(ImageReader(img), 2.92 * inch, 0.64 * inch, width=target_w, height=target_h, preserveAspectRatio=True, mask="auto")
    text(
        c,
        2.92 * inch,
        0.31 * inch,
        f"UNet Dice {float(rescue['unet_dice']):.3f} -> switched SAM {float(rescue['sam_dice']):.3f}; gain +{float(rescue['delta_vs_unet']):.3f}",
        7.5,
        color=INK,
    )
    c.showPage()
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_pipeline()
    make_results_and_rescue()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
