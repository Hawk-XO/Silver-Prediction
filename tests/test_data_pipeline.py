"""
tests/test_data_pipeline.py

End-to-end smoke test for the Phase 2 ingestion pipeline:
synthetic multi-contract MCX data -> continuous series (roll-adjusted) ->
merge with synthetic global factors -> final schema check.

Deliberately uses synthetic data (data/synthetic.py) rather than live
yfinance calls, so this test runs offline and deterministically in CI.
"""

import pandas as pd
import pytest

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global


@pytest.fixture(scope="module")
def synthetic_bundle():
    return generate_full_synthetic_dataset(start_date="2024-01-01", n_days=120)


def test_synthetic_mcx_has_expected_columns(synthetic_bundle):
    mcx = synthetic_bundle["mcx_multi_contract"]
    for col in ["contract", "open", "high", "low", "close", "volume", "open_interest"]:
        assert col in mcx.columns


def test_continuous_series_builds_without_error(synthetic_bundle):
    mcx = synthetic_bundle["mcx_multi_contract"]
    continuous = build_continuous_series(mcx)

    assert not continuous.empty
    for col in ["open", "high", "low", "close", "contract", "is_roll_date"]:
        assert col in continuous.columns

    # No NaNs should be introduced by the ratio-adjustment itself.
    assert continuous[["open", "high", "low", "close"]].isna().sum().sum() == 0

    # There should be at least one roll date given 120 days / ~21-day roll period.
    assert continuous["is_roll_date"].sum() >= 1


def test_continuous_series_is_return_continuous_across_roll(synthetic_bundle):
    """
    The core correctness property of ratio-adjustment: log-returns across a
    roll date should NOT show an artificial jump equal to the raw contract
    price gap. We check that the adjusted return on the roll date is of a
    similar order of magnitude to a typical daily return, not an outlier
    driven by the contract switch itself.
    """
    mcx = synthetic_bundle["mcx_multi_contract"]
    continuous = build_continuous_series(mcx)

    log_returns = (continuous["close"] / continuous["close"].shift(1)).apply(
        lambda x: pd.NA if x <= 0 else x
    ).dropna().apply(lambda x: float(x))
    import numpy as np
    log_returns = np.log(log_returns)

    typical_daily_vol = log_returns.std()
    roll_positions = continuous.index[continuous["is_roll_date"]]

    for roll_date in roll_positions:
        pos = continuous.index.get_loc(roll_date)
        if pos == 0:
            continue
        ret_on_roll = log_returns.iloc[pos - 1] if pos - 1 < len(log_returns) else None
        if ret_on_roll is None:
            continue
        # Roll-date return should not be a wild multiple of typical daily vol
        # (a raw, unadjusted series would show exactly this kind of jump).
        assert abs(ret_on_roll) < 8 * typical_daily_vol, (
            f"Suspiciously large return on roll date {roll_date} — "
            f"ratio-adjustment may not be correctly removing the roll jump."
        )


def test_merge_aligns_to_mcx_calendar(synthetic_bundle):
    mcx = build_continuous_series(synthetic_bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(
        mcx,
        synthetic_bundle["comex_silver"],
        synthetic_bundle["usdinr"],
        synthetic_bundle["dxy"],
    )

    # Merged frame's row count must match the MCX continuous series exactly —
    # we align to MCX's calendar, not the intersection of all calendars.
    assert len(merged) == len(mcx)

    for col in ["mcx_close", "comex_close", "usdinr_close", "dxy_close"]:
        assert col in merged.columns

    for col in ["comex_stale", "usdinr_stale", "dxy_stale"]:
        assert col in merged.columns
        assert merged[col].dtype == bool


def test_merge_forward_fill_does_not_exceed_limit(synthetic_bundle):
    """Sanity check that stale-flag bookkeeping matches actual NaN positions
    pre-fill (verifies we're not silently mis-tracking staleness)."""
    mcx = build_continuous_series(synthetic_bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(
        mcx,
        synthetic_bundle["comex_silver"],
        synthetic_bundle["usdinr"],
        synthetic_bundle["dxy"],
    )
    # Since our synthetic global factors use the same business-day calendar
    # as MCX, we expect zero staleness here — this test would catch a
    # regression where the merge logic mis-aligns dates.
    assert merged["comex_stale"].sum() == 0
    assert merged["usdinr_stale"].sum() == 0
    assert merged["dxy_stale"].sum() == 0
