"""
data/merge.py

MCX (India) and COMEX/DXY (US) don't share the same trading calendar —
Indian and US market holidays don't line up, and MCX has occasional
muhurat/special sessions with no US equivalent. Naively joining on date
with an inner join silently drops legitimate MCX trading days whenever the
US market happened to be closed, which quietly shrinks your usable dataset
and can bias results toward days both markets were open.

This module aligns everything to the MCX trading calendar (since that's
what we're ultimately trading) and forward-fills global factors on days
where MCX was open but COMEX/DXY had no fresh quote (e.g. US holiday).
Forward-fill is a deliberate choice here, not an accident: on a US holiday,
the "last known" USDINR/COMEX/DXY level is the most defensible estimate of
that value at the time of MCX's session, since no new information arrived.
"""

from __future__ import annotations

import pandas as pd

MAX_FORWARD_FILL_DAYS = 3  # don't fill across gaps longer than this; flag instead


def merge_mcx_with_global(
    mcx_df: pd.DataFrame,
    comex_df: pd.DataFrame,
    usdinr_df: pd.DataFrame,
    dxy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge MCX Silver continuous series with COMEX Silver, USD/INR, and DXY,
    aligned to the MCX trading calendar.

    Parameters
    ----------
    mcx_df : pd.DataFrame
        Output of build_continuous_series() — must have a `close` column,
        indexed by tz-aware date.
    comex_df, usdinr_df, dxy_df : pd.DataFrame
        Outputs of the global_factors fetchers — must have a `close` column
        each, indexed by tz-aware date.

    Returns
    -------
    pd.DataFrame
        Indexed by MCX trading date. Columns:
            mcx_open, mcx_high, mcx_low, mcx_close, mcx_volume, mcx_oi (if present)
            comex_close, usdinr_close, dxy_close
            comex_stale, usdinr_stale, dxy_stale (bool — True if the value
                was forward-filled rather than a fresh quote on that date)
    """
    base = mcx_df.copy()
    base.index = base.index.normalize()  # drop intraday time component, keep date only

    rename_map = {
        "open": "mcx_open",
        "high": "mcx_high",
        "low": "mcx_low",
        "close": "mcx_close",
        "volume": "mcx_volume",
        "open_interest": "mcx_oi",
    }
    base = base.rename(columns={k: v for k, v in rename_map.items() if k in base.columns})
    keep = [c for c in rename_map.values() if c in base.columns]
    base = base[keep]

    result = base.copy()

    for name, gdf in [("comex", comex_df), ("usdinr", usdinr_df), ("dxy", dxy_df)]:
        g = gdf.copy()
        g.index = g.index.normalize()
        g = g[["close"]].rename(columns={"close": f"{name}_close"})
        g = g[~g.index.duplicated(keep="last")]

        result = result.join(g, how="left")

        # Track which rows are forward-filled (stale) vs. fresh, before filling.
        stale_flag = result[f"{name}_close"].isna()

        result[f"{name}_close"] = result[f"{name}_close"].ffill(limit=MAX_FORWARD_FILL_DAYS)
        result[f"{name}_stale"] = stale_flag

    unresolved = result[[c for c in result.columns if c.endswith("_close")]].isna().any(axis=1).sum()
    if unresolved > 0:
        # Gaps longer than MAX_FORWARD_FILL_DAYS remain NaN by design — surface
        # this rather than silently propagating stale data indefinitely.
        result.attrs["unresolved_gap_rows"] = int(unresolved)

    return result
