"""
ui/format_utils.py

Small display-formatting helpers shared by ui/app.py and ui/report_export.py.
Kept dependency-free (no Streamlit, no Plotly) so the PDF exporter can reuse
the exact same number formatting the on-screen UI uses.
"""

from __future__ import annotations

import math

import pandas as pd

# Which comparison-table metrics count as "wins" going up vs down, for the
# green/red color-coding. Metrics not listed here (total_trades,
# avg_margin_pct_of_capital, max_margin_pct_of_capital) are informational,
# not a strategy-quality signal, so they render uncolored.
HIGHER_IS_BETTER = {"total_return_pct", "sharpe_ratio", "win_rate_pct", "profit_factor"}
LOWER_IS_BETTER = {"max_drawdown_pct", "total_fees_paid"}

METRIC_LABELS = {
    "total_return_pct": "Total return (%)",
    "sharpe_ratio": "Sharpe ratio",
    "max_drawdown_pct": "Max drawdown (%)",
    "win_rate_pct": "Win rate (%)",
    "profit_factor": "Profit factor",
    "total_trades": "Total trades",
    "total_fees_paid": "Total fees paid",
    "avg_margin_pct_of_capital": "Avg margin (% of capital)",
    "max_margin_pct_of_capital": "Max margin (% of capital)",
}

# Metrics that are already rupee amounts (vs. percentages/ratios/counts) --
# these get the compact ₹L/₹Cr treatment; everything else gets plain
# decimal/percent formatting.
CURRENCY_METRICS = {"total_fees_paid"}


def format_inr_compact(value: float, symbol: str = "\u20b9") -> str:
    """
    Compact Indian-style currency formatting: ₹3.28L instead of ₹328,300,
    ₹1.45Cr instead of ₹14,500,000. Falls back to plain formatting below
    ₹1,000 where compacting wouldn't save any readability.

    symbol: defaults to the ₹ glyph (fine anywhere rendered in a browser,
    e.g. the Streamlit UI). Pass symbol="Rs. " for reportlab-generated PDFs
    -- reportlab's built-in Helvetica font has no ₹ glyph and renders it as
    a blank/black box, so ui/report_export.py uses the ASCII-safe form.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}{symbol}{v / 1e7:.2f}Cr"
    if v >= 1e5:
        return f"{sign}{symbol}{v / 1e5:.2f}L"
    if v >= 1e3:
        return f"{sign}{symbol}{v / 1e3:.1f}K"
    return f"{sign}{symbol}{v:,.0f}"


def format_metric_value(metric: str, value: float, currency_symbol: str = "\u20b9") -> str:
    """Format a single comparison-table cell value based on what kind of
    metric it is (currency / percent / ratio / count)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if metric in CURRENCY_METRICS:
        return format_inr_compact(value, symbol=currency_symbol)
    if metric == "total_trades":
        return f"{int(value):,}"
    if metric.endswith("_pct"):
        return f"{value:,.2f}%"
    return f"{value:,.3f}"


def build_display_comparison(comparison: pd.DataFrame, currency_symbol: str = "\u20b9") -> pd.DataFrame:
    """
    Takes the raw `comparison` DataFrame from RunResult (rows=strategy/
    buy_and_hold, columns=metrics) and returns a metrics-as-rows,
    strategy/buy_and_hold-as-columns DataFrame of pre-formatted display
    strings -- this is the transpose that fixes the 9-columns-cut-off-at-100%
    layout, since 9 rows in a 2-column table always fits regardless of
    screen width.
    """
    transposed = comparison.T
    display_rows = {}
    for metric in transposed.index:
        row = transposed.loc[metric]
        display_rows[METRIC_LABELS.get(metric, metric)] = {
            col: format_metric_value(metric, row[col], currency_symbol=currency_symbol) for col in transposed.columns
        }
    return pd.DataFrame(display_rows).T


def metric_winner(comparison: pd.DataFrame, metric: str) -> str | None:
    """Returns 'strategy', 'buy_and_hold', or None (tie/uncolored metric)
    for a given raw metric name, used to color-code the transposed table."""
    if metric not in HIGHER_IS_BETTER and metric not in LOWER_IS_BETTER:
        return None
    strat_val = comparison.loc["strategy", metric]
    bh_val = comparison.loc["buy_and_hold", metric]
    if pd.isna(strat_val) or pd.isna(bh_val) or strat_val == bh_val:
        return None
    if metric in HIGHER_IS_BETTER:
        return "strategy" if strat_val > bh_val else "buy_and_hold"
    return "strategy" if strat_val < bh_val else "buy_and_hold"
