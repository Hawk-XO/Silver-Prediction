"""
data/mcx_loader.py

Loads raw MCX Silver futures OHLCV data from CSV. Designed to be tolerant of
column-naming inconsistencies across brokers/data vendors (Zerodha, Kite,
NCDEX-style exports, manually downloaded MCX bhavcopy files, etc.).

Usage:
    from data.mcx_loader import load_mcx_csv
    df = load_mcx_csv("path/to/silver_futures.csv")
"""

from __future__ import annotations

import pandas as pd

# Map of common column-name variants -> our canonical schema.
# Canonical schema: date, open, high, low, close, volume, open_interest, contract
_COLUMN_ALIASES = {
    "date": ["date", "timestamp", "datetime", "trading_date", "trade_date", "traddt", "date1"],
    "open": ["open", "open_price", "o", "openprice"],
    "high": ["high", "high_price", "h", "highprice"],
    "low": ["low", "low_price", "l", "lowprice"],
    "close": ["close", "close_price", "c", "ltp", "settle", "settlement_price", "closeprice", "settlementprice"],
    "volume": ["volume", "vol", "traded_qty", "total_traded_qty", "totaltradedqty", "totaltradedquantity", "vol (lots)", "volume (000's)", "volume (000s)"],
    "open_interest": ["open_interest", "oi", "openinterest", "oi (lots)"],
    "contract": ["contract", "symbol", "expiry_symbol", "instrument", "instrumentname"],
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close"]
OPTIONAL_COLUMNS = ["volume", "open_interest", "contract"]


def _build_rename_map(columns: list[str]) -> dict[str, str]:
    """Match incoming (lowercased) column names against known aliases."""
    lower_map = {c.lower().strip(): c for c in columns}
    rename_map = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                rename_map[lower_map[alias]] = canonical
                break
    return rename_map


def load_mcx_csv(path: str, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    """
    Load an MCX Silver futures OHLCV CSV into a canonical DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    tz : str
        Timezone to localize the date index to (default Asia/Kolkata, since
        MCX trades in IST).

    Returns
    -------
    pd.DataFrame
        Indexed by tz-aware `date`, columns: open, high, low, close,
        volume (if available), open_interest (if available),
        contract (if available).

    Raises
    ------
    ValueError
        If any required column (date, open, high, low, close) cannot be
        matched from the input file's headers.
    """
    raw = pd.read_csv(path)
    rename_map = _build_rename_map(list(raw.columns))
    df = raw.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"load_mcx_csv: could not find required column(s) {missing} in "
            f"{path}. Available columns after alias matching: {list(df.columns)}. "
            f"Add the actual header name(s) to _COLUMN_ALIASES in mcx_loader.py "
            f"if this is a new vendor format."
        )

    keep_cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize(tz)
    else:
        df["date"] = df["date"].dt.tz_convert(tz)

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)

    n_bad = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
    if n_bad > 0:
        df = df.dropna(subset=["open", "high", "low", "close"])

    return df
