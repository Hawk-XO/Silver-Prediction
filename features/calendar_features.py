"""
features/calendar_features.py

Calendar-derived features: day-of-week and days-to-expiry.

These are NOT subject to the shift(1)-before-rolling rule in
PROJECT_NOTES.md Section 3 — that rule exists to stop a feature from
peeking at price/return information from row t or later. Calendar features
derive purely from the row's own timestamp, which is known in advance (the
calendar for December doesn't depend on any price observed in December), so
there is no lookahead risk here regardless of shifting.

Days-to-expiry caveat
----------------------
We don't have a real MCX expiry calendar wired in yet (Phase 2's continuous
series drops the `contract`/`is_roll_date` columns once merged with global
factors — see data/merge.py). As an approximation, we treat the LAST
BUSINESS DAY OF THE CALENDAR MONTH as a stand-in for "expiry", since MCX
Silver contracts expire near month-end. This is a documented approximation,
not the real MCX expiry schedule (which varies contract to contract) —
replace with an actual expiry calendar (e.g. from contract_roll's
`is_roll_date` flag, threaded through from Phase 2) if calendar-accurate
expiry timing becomes important.
"""

from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `day_of_week` (0=Monday..6=Sunday) and `days_to_expiry` (approximate,
    see module docstring) columns.
    """
    out = df.copy()
    idx = out.index

    out["day_of_week"] = idx.dayofweek

    # Calendar month-end (naive), then snap to the nearest business day
    # on/before it, matching MCX's business-day trading calendar.
    naive_idx = idx.tz_localize(None) if idx.tz is not None else idx
    month_calendar_end = naive_idx.to_period("M").to_timestamp("M")
    approx_expiry = month_calendar_end + pd.offsets.BMonthEnd(0)
    # BMonthEnd(0) rolls FORWARD to the next business-month-end if the
    # anchor date isn't already one; since our anchor is the calendar
    # month-end itself, this correctly snaps backward/onto the last
    # business day of that same month rather than jumping to next month.
    needs_rollback = approx_expiry > month_calendar_end
    approx_expiry = pd.DatetimeIndex(
        [
            (e - pd.offsets.BMonthEnd(1)) if needs else e
            for e, needs in zip(approx_expiry, needs_rollback)
        ]
    )

    out["days_to_expiry"] = (approx_expiry - naive_idx).days

    return out
