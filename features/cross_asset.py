"""
features/cross_asset.py

Cross-asset relative-value features:

  1. Gold-Silver ratio (COMEX gold close / COMEX silver close) — a widely
     watched precious-metals relative-value signal.
  2. COMEX-MCX spread z-score — how far MCX Silver is trading from its
     COMEX*USDINR import-parity price, relative to its own recent history.
     (This reuses the same parity formula as data/mcx_proxy.py.)

Anti-leakage note
------------------
Both features below are, at their core, a same-day ratio/spread computed
from same-day closes. Per PROJECT_NOTES.md Section 3, a feature at row t may
only use information available strictly before t — so even a same-day ratio
counts as "today's value" and must be lagged. We therefore:

  - compute the raw ratio/spread series first (this is NOT yet a feature,
    just an intermediate quantity), then
  - `.shift(1)` it before it is either exposed directly or fed into a
    rolling z-score window.

This mirrors the shift-then-roll pattern used in features/rolling_stats.py.
"""

from __future__ import annotations

import pandas as pd

from data.mcx_proxy import TROY_OZ_TO_KG

DEFAULT_ZSCORE_WINDOWS = (10, 20)


def add_gold_silver_ratio(
    df: pd.DataFrame,
    gold_close: pd.Series,
    silver_close_col: str = "comex_close",
) -> pd.DataFrame:
    """
    Add `gold_silver_ratio` = COMEX gold close / COMEX silver close, shifted
    1 bar so row t reflects the ratio as of t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Merged MCX/global-factors frame (must contain `silver_close_col`,
        default 'comex_close' as produced by data/merge.py).
    gold_close : pd.Series
        COMEX gold close series (e.g. from data.global_factors.fetch_comex_gold),
        indexed by date. Will be aligned to `df`'s index and forward-filled
        up to 3 days to match the same staleness convention used in
        data/merge.py for other global factors.
    """
    out = df.copy()
    gold = gold_close.copy()
    gold.index = pd.to_datetime(gold.index).normalize()
    gold = gold[~gold.index.duplicated(keep="last")]

    aligned_gold = gold.reindex(out.index.normalize())
    aligned_gold.index = out.index
    aligned_gold = aligned_gold.ffill(limit=3)

    ratio = aligned_gold / out[silver_close_col]
    out["gold_silver_ratio"] = ratio.shift(1)
    return out


def add_comex_mcx_spread_zscore(
    df: pd.DataFrame,
    mcx_close_col: str = "mcx_close",
    comex_close_col: str = "comex_close",
    usdinr_close_col: str = "usdinr_close",
    windows: tuple[int, ...] = DEFAULT_ZSCORE_WINDOWS,
) -> pd.DataFrame:
    """
    Add the COMEX-MCX spread (MCX close minus zero-premium import-parity
    price implied by COMEX*USDINR) and its rolling z-score for each window.

    Adds columns: comex_mcx_spread, comex_mcx_spread_z_{w} for each w in
    `windows`.
    """
    out = df.copy()
    parity_price = out[comex_close_col] * out[usdinr_close_col] * TROY_OZ_TO_KG
    spread = out[mcx_close_col] - parity_price
    lagged_spread = spread.shift(1)

    out["comex_mcx_spread"] = lagged_spread
    for w in windows:
        roll_mean = lagged_spread.rolling(w).mean()
        roll_std = lagged_spread.rolling(w).std()
        out[f"comex_mcx_spread_z_{w}"] = (lagged_spread - roll_mean) / roll_std

    return out
