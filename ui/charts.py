"""
ui/charts.py

Interactive Plotly chart builders for ui/app.py, replacing the old static
matplotlib equity curve. Two charts:

    build_price_chart(price_series, trade_markers)
        Close-price line with BUY/SELL trade markers, TradingView-style
        range buttons (1M/3M/6M/YTD/1Y/All) + a range slider underneath.

    build_equity_chart(strategy_equity, buy_hold_equity, init_cash)
        Strategy vs buy-and-hold equity curves, same zoom/pan/range-button
        treatment, plus a flat reference line at the starting capital.

Both return a plain go.Figure -- ui/app.py renders them with
st.plotly_chart(fig, width="stretch"). Kept in one dark, monochrome-plus-
accent theme (white/gray/green/red) to match the rest of the UI rather than
Plotly's default palette.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BG = "#0A0A0A"
GRID = "#1A1A1A"
AXIS = "#AAAAAA"
LINE_WHITE = "#FFFFFF"
LINE_GRAY = "#777777"
BUY_GREEN = "#3DDC84"
SELL_RED = "#FF5C5C"

_RANGE_BUTTONS = dict(
    buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    font=dict(color="#000000"),
    activecolor="#FFFFFF",
    bgcolor="#2A2A2A",
)


def _base_layout(title: str, y_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#F2F2F2", size=15)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=AXIS, size=11),
        margin=dict(l=50, r=20, t=45, b=10),
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1, font=dict(color="#F2F2F2")),
        xaxis=dict(
            gridcolor=GRID, showline=True, linecolor="#2A2A2A", rangeslider=dict(visible=True, bgcolor="#141414"),
            rangeselector=_RANGE_BUTTONS,
        ),
        yaxis=dict(gridcolor=GRID, showline=True, linecolor="#2A2A2A", title=y_title, title_font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A1A1A", font=dict(color="#F2F2F2")),
    )


def _add_position_shading(fig: go.Figure, position_series: pd.Series) -> None:
    """
    Adds translucent background bands showing when the strategy was long
    (green tint), short (red tint), or flat (no tint) -- the "or whatever"
    part of "buy and sell and short or whatever": SELL in this system means
    going short (see backtest.vectorbt_backtest.signals_to_position), not
    just exiting a long, so the shading makes that explicit at a glance
    without having to read individual markers.
    """
    if position_series is None or position_series.empty:
        return
    pos = position_series.copy()
    change_points = pos[pos != pos.shift(1)].index.tolist()
    if not change_points:
        return
    boundaries = change_points + [pos.index[-1]]

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        val = pos.loc[seg_start]
        if val == 0:
            continue
        color = "rgba(61,220,132,0.08)" if val == 1 else "rgba(255,92,92,0.08)"
        fig.add_vrect(x0=seg_start, x1=seg_end, fillcolor=color, line_width=0, layer="below")


def build_price_chart(price_series: pd.Series, trade_markers: pd.DataFrame | None,
                       position_series: pd.Series | None = None) -> go.Figure:
    """
    price_series: close price indexed by date (model_ready['mcx_close']).
    trade_markers: DataFrame indexed by date with columns ['signal',
        'entry_price'] for non-HOLD rows (RunResult.trade_markers). May be
        None/empty if a run somehow produced no trades.
    position_series: 1/0/-1 per day (RunResult.position_series) -- when
        given, shades the background green while long and red while short
        so direction is visible even between markers, not just at them.
    """
    fig = go.Figure()

    _add_position_shading(fig, position_series)

    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values, mode="lines",
        line=dict(color=LINE_WHITE, width=1.3), name="MCX Silver (close)",
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.2f}<extra></extra>",
    ))

    if trade_markers is not None and not trade_markers.empty:
        buys = trade_markers[trade_markers["signal"] == "BUY"]
        sells = trade_markers[trade_markers["signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["entry_price"], mode="markers", name="BUY — go long",
                marker=dict(symbol="triangle-up", size=11, color=BUY_GREEN,
                            line=dict(width=1, color="#0A0A0A")),
                hovertemplate="%{x|%Y-%m-%d}<br>BUY — go LONG @ %{y:,.2f}<extra></extra>",
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["entry_price"], mode="markers", name="SELL — go short",
                marker=dict(symbol="triangle-down", size=11, color=SELL_RED,
                            line=dict(width=1, color="#0A0A0A")),
                hovertemplate="%{x|%Y-%m-%d}<br>SELL — go SHORT @ %{y:,.2f}<extra></extra>",
            ))

    fig.update_layout(**_base_layout("Price with trade markers", "Price (\u20b9)"))
    return fig


def build_price_only_chart(price_series: pd.Series) -> go.Figure:
    """
    Bare price line with no trade markers -- used by the UI's pre-run
    'preview data' chart (shows what's currently stored before you've run
    anything, so 'a chart view alone like before running').
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values, mode="lines",
        line=dict(color=LINE_WHITE, width=1.3), name="MCX Silver (close)",
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Stored price history (preview)", "Price (\u20b9)"))
    return fig


FORECAST_AMBER = "#FFA23D"


def build_forecast_chart(
    actual_price: pd.Series, predicted_price: pd.Series, start_date,
) -> go.Figure:
    """
    Two lines, one chart: the real close (white, solid) and the model's
    own multi-day-ahead guess (amber, dashed) starting exactly at
    `start_date` -- no orders, no P&L, just "here's reality, here's what
    the model thinks happens next." A vertical marker at `start_date`
    makes it obvious where real data stops and the forecast begins.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_price.index, y=actual_price.values, mode="lines",
        line=dict(color=LINE_WHITE, width=1.3), name="MCX Silver (actual)",
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:,.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=predicted_price.index, y=predicted_price.values, mode="lines+markers",
        line=dict(color=FORECAST_AMBER, width=1.8, dash="dash"),
        marker=dict(size=4, color=FORECAST_AMBER), name="Model forecast",
        hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:,.2f}<extra></extra>",
    ))

    fig.add_vline(x=start_date, line=dict(color="#555555", width=1, dash="dot"))

    fig.update_layout(**_base_layout("Actual vs. model forecast", "Price (\u20b9)"))
    return fig


def build_equity_chart(strategy_equity: pd.Series, buy_hold_equity: pd.Series, init_cash: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_equity.index, y=strategy_equity.values, mode="lines",
        line=dict(color=LINE_WHITE, width=1.6), name="Strategy",
        hovertemplate="%{x|%Y-%m-%d}<br>Strategy: \u20b9%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=buy_hold_equity.index, y=buy_hold_equity.values, mode="lines",
        line=dict(color=LINE_GRAY, width=1.2, dash="dash"), name="Buy & hold",
        hovertemplate="%{x|%Y-%m-%d}<br>Buy & hold: \u20b9%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=init_cash, line=dict(color="#333333", width=1, dash="dot"))

    fig.update_layout(**_base_layout("Equity curve — strategy vs buy-and-hold", "Portfolio value (\u20b9)"))
    return fig
