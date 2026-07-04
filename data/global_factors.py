"""
data/global_factors.py

Fetches global cross-asset series relevant to MCX Silver: COMEX silver
futures, USD/INR exchange rate, and the US Dollar Index (DXY). These are
used both as raw features and to construct derived features (COMEX-MCX
spread, INR-adjusted parity price) in the feature engineering phase.

Requires: yfinance (pip install yfinance)

Note on tickers: yfinance ticker symbols can shift over time as data
providers change symbology. If a fetch starts returning empty data, search
for the current ticker rather than assuming the ones below are permanent.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

TICKERS = {
    "comex_silver": "SI=F",      # COMEX Silver futures (continuous, front-month)
    "comex_gold": "GC=F",        # COMEX Gold futures (continuous, front-month)
    "usdinr": "USDINR=X",        # USD/INR spot exchange rate
    "dxy": "DX-Y.NYB",           # US Dollar Index
}


def _fetch_single(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(
            f"global_factors: yfinance returned no data for ticker '{ticker}'. "
            f"Check the ticker is still valid, or that start/end dates aren't "
            f"outside the available range."
        )
    # yfinance sometimes returns MultiIndex columns for single tickers depending
    # on version; flatten defensively.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        data.index = data.index.tz_convert("Asia/Kolkata")
    return data[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)


def fetch_comex_silver(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch COMEX Silver futures (SI=F) OHLCV, indexed by date (IST)."""
    return _fetch_single(TICKERS["comex_silver"], start, end)


def fetch_comex_gold(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch COMEX Gold futures (GC=F) OHLCV, indexed by date (IST).

    Used in features/cross_asset.py to build the Gold-Silver ratio, a
    commonly watched relative-value signal in precious metals markets.
    """
    return _fetch_single(TICKERS["comex_gold"], start, end)


def fetch_usdinr(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch USD/INR spot rate OHLCV, indexed by date (IST)."""
    return _fetch_single(TICKERS["usdinr"], start, end)


def fetch_dxy(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch US Dollar Index (DXY) OHLCV, indexed by date (IST)."""
    return _fetch_single(TICKERS["dxy"], start, end)


def fetch_all_global_factors(
    start: str | None = None, end: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper fetching all three global factor series at once.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'comex_silver', 'usdinr', 'dxy'. Each value is an OHLCV
        DataFrame indexed by date (IST). Any individual fetch that fails
        raises immediately rather than silently returning partial data —
        callers should decide whether a partial set is acceptable.
    """
    return {
        "comex_silver": fetch_comex_silver(start, end),
        "usdinr": fetch_usdinr(start, end),
        "dxy": fetch_dxy(start, end),
    }
