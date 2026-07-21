"""
ui/report_export.py

Builds a downloadable PDF performance report from a RunResult, for the
"export to PDF" button in ui/app.py. Uses reportlab (already a project
dependency-adjacent choice -- no new heavy deps) with a matplotlib-rendered
equity chart embedded as an image, since reportlab has no native chart
support and this only needs to run once per export click (not interactively),
unlike the live Plotly charts in the UI itself.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

from ui.format_utils import build_display_comparison, metric_winner, format_inr_compact


def _render_equity_chart_png(strategy_equity: pd.Series, buy_hold_equity: pd.Series, init_cash: float) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    ax.plot(strategy_equity.index, strategy_equity.values, color="#1A1A1A", linewidth=1.4, label="Strategy")
    ax.plot(buy_hold_equity.index, buy_hold_equity.values, color="#777777", linewidth=1.1,
            linestyle="--", label="Buy & hold")
    ax.axhline(init_cash, color="#BBBBBB", linewidth=0.8, linestyle=":")
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, color="#EEEEEE", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_pdf_report(result) -> bytes:
    """
    result: a ui.pipeline_runner.RunResult. Returns the PDF file's raw bytes
    (suitable for st.download_button's `data=` argument directly).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Silver Prediction — Run Report",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=2)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#555555"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    body = styles["Normal"]

    story = []
    story.append(Paragraph("Silver Prediction — Run Report", title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Contract {result.config.contract} · "
        f"Horizon {result.config.horizon}d",
        caption_style,
    ))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#DDDDDD")))

    # --- run summary -------------------------------------------------
    story.append(Paragraph("Run summary", h2))
    summary_lines = (
        f"{result.raw_row_count:,} raw rows ({result.source_counts}) &rarr; "
        f"{result.n_model_ready_rows:,} rows with complete features &rarr; "
        f"{result.n_predictions:,} walk-forward predictions<br/>"
        f"Date range: {result.date_range[0].date()} to {result.date_range[1].date()}"
    )
    story.append(Paragraph(summary_lines, body))
    if result.warning:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Note:</b> {result.warning}", ParagraphStyle(
            "Warn", parent=body, textColor=colors.HexColor("#8A6D00"))))

    # --- performance comparison table ---------------------------------
    story.append(Paragraph("Performance report — strategy vs buy &amp; hold", h2))
    display_df = build_display_comparison(result.comparison, currency_symbol="Rs. ")
    raw_metrics = result.comparison.T.index.tolist()

    table_data = [["Metric", "Strategy", "Buy & hold"]]
    row_colors = []  # (row_index, winner_col) for cell background highlighting
    for i, (label, raw_metric) in enumerate(zip(display_df.index, raw_metrics), start=1):
        table_data.append([label, display_df.loc[label, "strategy"], display_df.loc[label, "buy_and_hold"]])
        winner = metric_winner(result.comparison, raw_metric)
        if winner:
            row_colors.append((i, 1 if winner == "strategy" else 2))

    tbl = Table(table_data, colWidths=[7 * cm, 4.5 * cm, 4.5 * cm], hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1A1A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_idx, col_idx in row_colors:
        style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), colors.HexColor("#DCF3E3")))
        style_cmds.append(("TEXTCOLOR", (col_idx, row_idx), (col_idx, row_idx), colors.HexColor("#1B7A3E")))
        style_cmds.append(("FONTNAME", (col_idx, row_idx), (col_idx, row_idx), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)

    # --- equity chart ---------------------------------------------------
    story.append(Paragraph("Equity curve", h2))
    png_bytes = _render_equity_chart_png(result.strategy_equity, result.buy_hold_equity, result.config.init_cash)
    story.append(Image(io.BytesIO(png_bytes), width=17 * cm, height=17 * cm * 3.0 / 7.2))

    # --- PnL breakdown ---------------------------------------------------
    story.append(Paragraph("Trade-level PnL breakdown", h2))
    pnl = result.trade_pnl
    pnl_rows = [
        ["Trades", f"{pnl['num_trades']:,}"],
        ["Win / Loss", f"{pnl['num_wins']} / {pnl['num_losses']}"],
        ["Avg win", format_inr_compact(pnl["avg_win"], symbol="Rs. ") if pd.notna(pnl["avg_win"]) else "—"],
        ["Avg loss", format_inr_compact(pnl["avg_loss"], symbol="Rs. ") if pd.notna(pnl["avg_loss"]) else "—"],
        ["Avg win/loss ratio", f"{pnl['avg_win_loss_ratio']:.2f}" if pd.notna(pnl["avg_win_loss_ratio"]) else "—"],
        ["Expectancy/trade", format_inr_compact(pnl["expectancy_per_trade"], symbol="Rs. ") if pd.notna(pnl["expectancy_per_trade"]) else "—"],
    ]
    pnl_tbl = Table(pnl_rows, colWidths=[7 * cm, 6 * cm], hAlign="LEFT")
    pnl_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pnl_tbl)

    # --- paper broker replay ---------------------------------------------
    story.append(Paragraph("Paper broker replay", h2))
    broker_rows = [
        ["Orders placed", f"{result.broker_n_orders:,}"],
        ["Realised P&L", format_inr_compact(result.broker_pnl["realised"], symbol="Rs. ")],
        ["Total P&L", format_inr_compact(result.broker_pnl["total"], symbol="Rs. ")],
    ]
    broker_tbl = Table(broker_rows, colWidths=[7 * cm, 6 * cm], hAlign="LEFT")
    broker_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(broker_tbl)

    doc.build(story)
    buf.seek(0)
    return buf.read()
