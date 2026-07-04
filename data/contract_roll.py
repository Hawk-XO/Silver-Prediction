"""
data/contract_roll.py

MCX Silver futures expire monthly. To build a usable continuous price series
for modeling, individual contract series must be stitched together across
expiries. Naively concatenating raw contract prices creates artificial jumps
at each roll date (the new front-month contract rarely trades at the exact
same price as the expiring one).

This module implements the **ratio-adjustment** (back-adjustment) method:
at each roll date, all historical prices before the roll are scaled by the
ratio (new_contract_price / old_contract_price) on that date, so the
percentage-return series remains continuous even though absolute price
levels are adjusted. This preserves return-based modeling integrity, which
matters since our target variable is a log return (see PROJECT_NOTES.md).

Expected input: a long-format DataFrame with columns
    date | contract | open | high | low | close | volume | open_interest
covering multiple overlapping monthly contracts (e.g. SILVER25JUL,
SILVER25AUG, ...). This is the raw shape you'd get from stacking multiple
load_mcx_csv() calls, one per contract, with the `contract` column populated.
"""

from __future__ import annotations

import pandas as pd


def _pick_front_month(day_df: pd.DataFrame, volume_col: str = "volume") -> str:
    """On a given date, the 'front month' contract is the one with the
    highest traded volume (a more reliable proxy than nearest expiry alone,
    since MCX liquidity sometimes lags the calendar-nearest contract)."""
    if volume_col in day_df.columns and day_df[volume_col].notna().any():
        return day_df.sort_values(volume_col, ascending=False).iloc[0]["contract"]
    return day_df.iloc[0]["contract"]


def build_continuous_series(
    multi_contract_df: pd.DataFrame,
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """
    Build a ratio-adjusted continuous price series from multiple overlapping
    MCX Silver contracts.

    Parameters
    ----------
    multi_contract_df : pd.DataFrame
        Long-format, index = date (tz-aware), must include a `contract`
        column plus OHLC(V/OI) columns. Typically built by concatenating
        multiple load_mcx_csv() outputs, each tagged with its contract name.
    price_cols : tuple[str, ...]
        Which columns to ratio-adjust. Volume/open_interest are carried
        through unadjusted (they're not comparable across the roll anyway;
        keep the front-month contract's raw values for those on each date).

    Returns
    -------
    pd.DataFrame
        Single continuous series indexed by date, ratio-adjusted OHLC,
        plus `contract` (which physical contract was front-month that day)
        and `is_roll_date` (bool flag marking days where the front month
        changed, useful for excluding/flagging in downstream feature
        engineering).
    """
    if "contract" not in multi_contract_df.columns:
        raise ValueError(
            "build_continuous_series requires a 'contract' column identifying "
            "which MCX contract each row belongs to."
        )

    df = multi_contract_df.copy()
    df = df.sort_index()

    # Determine the front-month contract for each date.
    front_month_by_date = (
        df.reset_index()
        .groupby("date", group_keys=False)
        .apply(lambda g: _pick_front_month(g), include_groups=False)
    )
    front_month_by_date.name = "front_contract"

    dates = front_month_by_date.index
    contracts = front_month_by_date.values

    # Build the raw front-month series (unadjusted) first.
    rows = []
    for date, contract in zip(dates, contracts):
        row = df.loc[(df.index == date) & (df["contract"] == contract)]
        if row.empty:
            continue
        rows.append(row.iloc[0])
    raw_front = pd.DataFrame(rows)
    raw_front.index = pd.DatetimeIndex([r.name for r in rows]) if len(rows) else pd.DatetimeIndex([])
    raw_front = raw_front.sort_index()

    if raw_front.empty:
        raise ValueError("build_continuous_series: no rows matched after selecting front-month contracts.")

    raw_front["is_roll_date"] = raw_front["contract"] != raw_front["contract"].shift(1)
    raw_front.loc[raw_front.index[0], "is_roll_date"] = False  # first row is not a "roll"

    # Ratio-adjustment: walk backward from the most recent contract.
    # At each roll date, compute ratio = new_contract_close / old_contract_close
    # using the OVERLAP date (the roll date itself, where both contracts
    # have a quote) and apply the cumulative ratio to all prior rows.
    adjusted = raw_front.copy()
    roll_dates = adjusted.index[adjusted["is_roll_date"]]

    cumulative_ratio = 1.0
    # Process roll dates from most recent to oldest, adjusting everything
    # before each roll date by the ratio observed at that roll.
    for roll_date in sorted(roll_dates, reverse=True):
        idx_pos = adjusted.index.get_loc(roll_date)
        if idx_pos == 0:
            continue
        new_contract = adjusted.iloc[idx_pos]["contract"]
        old_contract = adjusted.iloc[idx_pos - 1]["contract"]

        new_quote = df.loc[(df.index == roll_date) & (df["contract"] == new_contract)]
        old_quote = df.loc[(df.index == roll_date) & (df["contract"] == old_contract)]

        if new_quote.empty or old_quote.empty or old_quote.iloc[0]["close"] == 0:
            # Can't compute a clean ratio on this roll date (no overlapping
            # quote for the old contract) — skip adjustment for this roll
            # rather than risk a divide-by-zero or fabricated ratio.
            continue

        ratio = new_quote.iloc[0]["close"] / old_quote.iloc[0]["close"]
        cumulative_ratio *= ratio

        mask = adjusted.index < roll_date
        for col in price_cols:
            adjusted.loc[mask, col] = adjusted.loc[mask, col] * ratio

    keep_cols = list(price_cols) + ["contract", "is_roll_date"]
    for extra in ["volume", "open_interest"]:
        if extra in adjusted.columns:
            keep_cols.append(extra)

    return adjusted[keep_cols]
