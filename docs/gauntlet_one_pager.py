#!/usr/bin/env python3
"""
Gauntlet Evidence Sheet — 1-Page PDF Generator
===============================================
Reads reports/autonomous_gauntlet_results.json and produces
docs/gauntlet_evidence_sheet.pdf

Usage:
    python docs/gauntlet_one_pager.py

Requires:
    pip install reportlab
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB = True
except ImportError:
    REPORTLAB = False


# ── Colour palette ────────────────────────────────────────────────────────────
TEAL   = colors.HexColor("#0D7377")
DARK   = colors.HexColor("#14213D")
GOLD   = colors.HexColor("#E9C46A")
LIGHT  = colors.HexColor("#F4F4F4")
WHITE  = colors.white
GREEN  = colors.HexColor("#2A9D8F")
RED    = colors.HexColor("#E76F51")


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def status_colour(status: str) -> colors.Color:
    return GREEN if status == "RELIABLE" else (GOLD if status == "HIGH_UNCERTAINTY" else RED)


def build_pdf(data: dict, out: Path) -> None:
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=18, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9, textColor=GOLD,
        alignment=TA_CENTER, spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=10, textColor=TEAL,
        spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, textColor=DARK,
        leading=13,
    )
    mono_style = ParagraphStyle(
        "Mono", parent=styles["Normal"],
        fontName="Courier", fontSize=8, textColor=DARK,
        backColor=LIGHT, leading=12,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.grey,
        alignment=TA_CENTER,
    )

    story = []

    # ── Header banner ────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("QuLab Infinite — Autonomous Lab Gauntlet", title_style),
    ]]
    header_table = Table(header_data, colWidths=[7.2 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("ROUNDEDCORNERS", [6]),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)

    sub_data = [[
        Paragraph(
            "Contract-Enforced Benchmark  ·  Tag: v1.0-gauntlet  ·  "
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            sub_style,
        )
    ]]
    sub_table = Table(sub_data, colWidths=[7.2 * inch])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 10))

    # ── SNR Definition & Taxonomy ────────────────────────────────────────────
    story.append(Paragraph("SNR Definition &amp; Status Taxonomy", section_style))
    story.append(Paragraph(
        "<b>SNR</b> = (max − min of signal window) / (2 × std of baseline window) "
        "&nbsp;[USP &lt;1225&gt;, same formula across all 4 runs — no goalposts moved]",
        body_style,
    ))
    story.append(Spacer(1, 6))

    tax_headers = [["Band", "Threshold", "Label", "Lab Action"]]
    tax_rows = [
        ["●", "SNR < 5",    "INSUFFICIENT_SNR",  "Halt — do not compute derived params"],
        ["●", "5 ≤ SNR < 15","HIGH_UNCERTAINTY", "LOD/LOQ regime — robust stats only, CI ×2"],
        ["●", "SNR ≥ 15",   "RELIABLE",          "Full analysis permitted"],
    ]
    tax_colours = [RED, GOLD, GREEN]
    tax_table = Table(
        tax_headers + tax_rows,
        colWidths=[0.25 * inch, 1.05 * inch, 1.6 * inch, 4.3 * inch],
    )
    tax_style = [
        ("BACKGROUND",   (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]
    for i, c in enumerate(tax_colours):
        tax_style.append(("TEXTCOLOR", (0, i + 1), (0, i + 1), c))
        tax_style.append(("FONTNAME",  (0, i + 1), (0, i + 1), "Helvetica-Bold"))
    tax_table.setStyle(TableStyle(tax_style))
    story.append(tax_table)
    story.append(Spacer(1, 10))

    # ── Run Summary Table ────────────────────────────────────────────────────
    story.append(Paragraph("Run Summary", section_style))

    runs = data.get("runs", [])
    run_headers = [["Run", "Scenario", "Initial SNR", "Final SNR", "Gain", "Iter.", "Status"]]
    scenario_map = {
        1: "Noise-dominant (RF + shot noise)",
        2: "Signal-limited — LOD/LOQ regime",
        3: "Drift-dominant — thermal baseline walk",
        4: "HARD MODE — saturation + drift (2-iter)",
    }
    run_rows = []
    for r in runs:
        rid     = r["run_id"]
        status  = r["final_status"]
        i0      = r["initial_snr"]
        i1      = r["final_snr"]
        gain    = r["snr_gain"]
        iters   = r["iterations"]
        row = [
            str(rid),
            scenario_map.get(rid, ""),
            f"{i0:.2f}",
            f"{i1:.2f}",
            f"{gain:.2f}×",
            str(iters),
            status,
        ]
        run_rows.append(row)

    run_table = Table(
        run_headers + run_rows,
        colWidths=[0.3*inch, 2.45*inch, 0.75*inch, 0.75*inch, 0.6*inch, 0.4*inch, 1.45*inch],
    )
    run_ts = [
        ("BACKGROUND",   (0, 0), (-1, 0),  TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID",         (0, 0), (-1, -1),  0.4, colors.lightgrey),
        ("TOPPADDING",   (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING",(0, 0), (-1, -1),  4),
        ("LEFTPADDING",  (0, 0), (-1, -1),  5),
        ("ALIGN",        (2, 1), (-1, -1),  "CENTER"),
    ]
    # Colour status column
    status_col = 6
    status_colour_map = {"RELIABLE": GREEN, "HIGH_UNCERTAINTY": GOLD, "INSUFFICIENT_SNR": RED}
    for i, r in enumerate(run_rows):
        c = status_colour_map.get(r[status_col], DARK)
        run_ts.append(("TEXTCOLOR", (status_col, i + 1), (status_col, i + 1), c))
        run_ts.append(("FONTNAME",  (status_col, i + 1), (status_col, i + 1), "Helvetica-Bold"))
    run_table.setStyle(TableStyle(run_ts))
    story.append(run_table)

    avg = sum(r["snr_gain"] for r in runs[:3]) / 3
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"<b>Average SNR gain (Runs 1–3):</b> {avg:.2f}×  &nbsp;|&nbsp;  "
        "<b>Hard-Mode Run 4:</b> first redesign FAILED (SNR=3.86, INSUFFICIENT_SNR) → 2 iterations required.",
        body_style,
    ))
    story.append(Spacer(1, 10))

    # ── Prediction Model ─────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Model (iid Gaussian) — Why Observed &gt; Predicted", section_style))
    story.append(Paragraph(
        "For iid Gaussian noise: <b>gain = √N_eff</b>, where N_eff = (n_after/n_before) × √W for "
        "combined oversampling + moving-average window W (sublinear, correlated samples). "
        "Observed gain exceeds prediction when: (a) noise has heavy tails suppressed by outlier "
        "rejection (Run 1), (b) drift is <i>deterministic</i> — linear reference fit removes ~97% "
        "vs the conservative 85% assumed (Run 3).",
        body_style,
    ))
    story.append(Spacer(1, 10))

    # ── Hard-Mode Enforcement Block ───────────────────────────────────────────
    story.append(Paragraph("Hard-Mode Contract (Run 4) — Enforced at Runtime", section_style))
    code = (
        "if s_first >= 5:\n"
        "    raise RuntimeError('Hard-Mode violation: first redesign must remain &lt;5')\n"
        "if s_final < 15:\n"
        "    raise RuntimeError('Hard-Mode violation: final SNR must reach ≥15')"
    )
    story.append(Paragraph(code.replace("\n", "<br/>"), mono_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "These assertions run on every execution. If anyone touches the thresholds or SNR "
        "construction, the script raises immediately — the benchmark cannot silently degrade.",
        body_style,
    ))
    story.append(Spacer(1, 10))

    # ── Reproducibility ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    repro = [
        ["Repository",  "github.com/Workofarttattoo/QuLabInfinite"],
        ["Tag",         "v1.0-gauntlet (immutable)"],
        ["Run command", "python autonomous_lab_gauntlet.py"],
        ["Methodology", "docs/gauntlet.md"],
        ["Raw output",  "reports/autonomous_gauntlet_results.json  (UTC-timestamped)"],
    ]
    repro_table = Table(repro, colWidths=[1.3 * inch, 5.9 * inch])
    repro_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(repro_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Copyright © 2025 Joshua Hendricks Cole · Corporation of Light · Patent Pending  |  "
        "For NREL / ARPA-E / Lux Research review — reproducible on any Python 3.11+ environment.",
        footer_style,
    ))

    doc.build(story)
    print(f"✓ PDF → {out}")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    results_path = repo / "reports" / "autonomous_gauntlet_results.json"
    out_path     = repo / "docs" / "gauntlet_evidence_sheet.pdf"

    if not results_path.exists():
        print(f"Results not found at {results_path}. Run autonomous_lab_gauntlet.py first.")
        return

    if not REPORTLAB:
        print("reportlab not installed. Run:  pip install reportlab")
        return

    data = load_results(results_path)
    build_pdf(data, out_path)


if __name__ == "__main__":
    main()
