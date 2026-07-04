"""
features/indicators.py

Technical indicators (EMA, MACD, RSI, ATR, Bollinger %B, ADX) built on top of
the merged MCX/global-factors DataFrame (output of data.merge.merge_mcx_with_global).

Anti-leakage note (see PROJECT_NOTES.md Section 3)
---------------------------------------------------
PROJECT_NOTES.md requires every rolling/windowed feature to use only
information available strictly BEFORE time t — i.e. row t must not see
row t's own close/high/low.

The `ta` library's indicators (EMA, RSI, MACD, ATR, Bollinger, ADX) are all
CAUSAL functions of a price series: the value at position i is computed only
from prices at positions <= i. Because of that causality, the following two
approaches are mathematically equivalent:

    (a) shift the input price series by 1 bar, then run the indicator, or
    (b) run the indicator on the raw series, then shift the OUTPUT by 1 bar.

We use (b) here — compute on the raw OHLC, then `.shift(1)` the resulting
indicator column — because it lets us use the `ta` library's indicator
classes directly without hand-rolling shifted OHLC inputs (which some
indicators, like ADX/ATR, combine across open/high/low/close in ways that
are fiddly to shift consistently by hand). The net effect on every column
produced here is identical to shifting inputs first: the value assigned to
row t reflects only information available through row t-1.

Every function below returns a NEW DataFrame (copy) with additional columns
appended to `df`; it never mutates the input in place.
"""

from __future__ import annotations

import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

EMA_WINDOWS = (9, 21, 50, 200)


def add_ema(df: pd.DataFrame, price_col: str = "mcx_close", windows: tuple[int, ...] = EMA_WINDOWS) -> pd.DataFrame:
    """Add EMA columns `ema_{window}` for each window, shifted 1 bar (see module docstring)."""
    out = df.copy()
    for w in windows:
        ema = EMAIndicator(close=out[price_col], window=w, fillna=False).ema_indicator()
        out[f"ema_{w}"] = ema.shift(1)
    return out


def add_macd(
    df: pd.DataFrame,
    price_col: str = "mcx_close",
    window_fast: int = 12,
    window_slow: int = 26,
    window_sign: int = 9,
) -> pd.DataFrame:
    """Add `macd`, `macd_signal`, `macd_diff` columns, shifted 1 bar."""
    out = df.copy()
    macd_ind = MACD(
        close=out[price_col],
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_sign,
        fillna=False,
    )
    out["macd"] = macd_ind.macd().shift(1)
    out["macd_signal"] = macd_ind.macd_signal().shift(1)
    out["macd_diff"] = macd_ind.macd_diff().shift(1)
    return out


def add_rsi(df: pd.DataFrame, price_col: str = "mcx_close", window: int = 14) -> pd.DataFrame:
    """Add `rsi_{window}` column, shifted 1 bar."""
    out = df.copy()
    rsi = RSIIndicator(close=out[price_col], window=window, fillna=False).rsi()
    out[f"rsi_{window}"] = rsi.shift(1)
    return out


def add_atr(
    df: pd.DataFrame,
    high_col: str = "mcx_high",
    low_col: str = "mcx_low",
    close_col: str = "mcx_close",
    window: int = 14,
) -> pd.DataFrame:
    """Add `atr_{window}` column, shifted 1 bar."""
    out = df.copy()
    atr = AverageTrueRange(
        high=out[high_col], low=out[low_col], close=out[close_col], window=window, fillna=False
    ).average_true_range()
    out[f"atr_{window}"] = atr.shift(1)
    return out


def add_bollinger_percent_b(
    df: pd.DataFrame,
    price_col: str = "mcx_close",
    window: int = 20,
    window_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Add `bb_percent_b_{window}`: Bollinger %B, i.e. where price sits within
    the bands, scaled 0-1 (0 = lower band, 1 = upper band). Shifted 1 bar.
    """
    out = df.copy()
    bb = BollingerBands(close=out[price_col], window=window, window_dev=window_dev, fillna=False)
    out[f"bb_percent_b_{window}"] = bb.bollinger_pband().shift(1)
    return out


def add_adx(
    df: pd.DataFrame,
    high_col: str = "mcx_high",
    low_col: str = "mcx_low",
    close_col: str = "mcx_close",
    window: int = 14,
) -> pd.DataFrame:
    """Add `adx_{window}` column, shifted 1 bar."""
    out = df.copy()
    adx = ADXIndicator(
        high=out[high_col], low=out[low_col], close=out[close_col], window=window, fillna=False
    ).adx()
    out[f"adx_{window}"] = adx.shift(1)
    return out


def add_all_indicators(
    df: pd.DataFrame,
    high_col: str = "mcx_high",
    low_col: str = "mcx_low",
    close_col: str = "mcx_close",
) -> pd.DataFrame:
    """Convenience wrapper applying every indicator in this module with default settings."""
    out = df.copy()
    out = add_ema(out, price_col=close_col)
    out = add_macd(out, price_col=close_col)
    out = add_rsi(out, price_col=close_col)
    out = add_atr(out, high_col=high_col, low_col=low_col, close_col=close_col)
    out = add_bollinger_percent_b(out, price_col=close_col)
    out = add_adx(out, high_col=high_col, low_col=low_col, close_col=close_col)
    return out
