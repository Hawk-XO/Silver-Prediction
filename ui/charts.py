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


def build_price_chart(price_series: pd.Series, trade_markers: pd.DataFrame | None) -> go.Figure:
    """
    price_series: close price indexed by date (model_ready['mcx_close']).
    trade_markers: DataFrame indexed by date with columns ['signal',
        'entry_price'] for non-HOLD rows (RunResult.trade_markers). May be
        None/empty if a run somehow produced no trades.
    """
    fig = go.Figure()
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
                x=buys.index, y=buys["entry_price"], mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=10, color=BUY_GREEN,
                            line=dict(width=1, color="#0A0A0A")),
                hovertemplate="%{x|%Y-%m-%d}<br>BUY @ %{y:,.2f}<extra></extra>",
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["entry_price"], mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=10, color=SELL_RED,
                            line=dict(width=1, color="#0A0A0A")),
                hovertemplate="%{x|%Y-%m-%d}<br>SELL @ %{y:,.2f}<extra></extra>",
            ))

    fig.update_layout(**_base_layout("Price with trade markers", "Price (\u20b9)"))
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
