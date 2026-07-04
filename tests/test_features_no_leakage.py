"""
tests/test_features_no_leakage.py

Phase 3 leakage test suite, per PROJECT_NOTES.md Section 3:

    "Every feature-engineering module must ship with a pytest test
    asserting that no feature value at row t depends on data at row >= t."

Primary technique: TRUNCATION EQUIVALENCE.
-------------------------------------------
For a causal feature function f (one that, correctly, only looks backward),
computing f on a full series and computing f on a TRUNCATED prefix of that
same series must produce IDENTICAL values at every row that exists in both
outputs. If truncating the future changes a past row's feature value, that
row was leaking information from the truncated (future) part of the series.

We apply this at three levels:
  1. The full Phase 3 pipeline (features.pipeline.build_feature_matrix).
  2. Each individual feature module, to pinpoint failures precisely.
  3. A handful of hand-computed sanity checks on tiny fixed series, to catch
     off-by-one shift errors that truncation alone might not surface (e.g. a
     feature that's shifted by 1 in the WRONG direction would still pass a
     truncation-equivalence check, since it's still causal --- just shifted
     the wrong way relative to the intended row-alignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global

from features.indicators import add_all_indicators, add_rsi
from features.rolling_stats import add_return_rolling_stats
from features.cross_asset import add_gold_silver_ratio, add_comex_mcx_spread_zscore
from features.calendar_features import add_calendar_features
from features.target import add_forward_log_return_target
from features.pipeline import build_feature_matrix

N_DAYS = 180
CUTOFFS = [60, 100, 150]  # truncate-to-this-many-rows checkpoints


@pytest.fixture(scope="module")
def merged_bundle():
    bundle = generate_full_synthetic_dataset(start_date="2024-01-01", n_days=N_DAYS)
    continuous = build_continuous_series(bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(
        continuous, bundle["comex_silver"], bundle["usdinr"], bundle["dxy"]
    )
    gold_close = bundle["comex_gold"]["close"]
    return merged, gold_close


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """All columns added by feature engineering, excluding the raw merged
    input columns and the target (target is deliberately forward-looking)."""
    raw_cols = {
        "mcx_open", "mcx_high", "mcx_low", "mcx_close", "mcx_volume", "mcx_oi",
        "comex_close", "usdinr_close", "dxy_close",
        "comex_stale", "usdinr_stale", "dxy_stale",
    }
    return [c for c in df.columns if c not in raw_cols and c != "target"]


# ---------------------------------------------------------------------------
# 1. Full-pipeline truncation-equivalence test
# ---------------------------------------------------------------------------

def test_full_pipeline_no_lookahead(merged_bundle):
    merged, gold_close = merged_bundle
    full = build_feature_matrix(merged, gold_close=gold_close, include_target=False)
    cols = _feature_cols(full)
    assert len(cols) > 0, "sanity check: pipeline should have added feature columns"

    for cutoff in CUTOFFS:
        truncated_input = merged.iloc[:cutoff]
        truncated_gold = gold_close.loc[gold_close.index <= truncated_input.index[-1].tz_localize(None)] \
            if gold_close.index.tz is None else gold_close.loc[gold_close.index <= truncated_input.index[-1]]
        truncated = build_feature_matrix(truncated_input, gold_close=truncated_gold, include_target=False)

        last_date = truncated.index[-1]
        full_row = full.loc[last_date, cols]
        trunc_row = truncated.loc[last_date, cols]

        pd.testing.assert_series_equal(
            full_row.astype(float),
            trunc_row.astype(float),
            check_names=False,
            obj=f"feature row at cutoff={cutoff} (date={last_date})",
        )


# ---------------------------------------------------------------------------
# 2. Per-module truncation-equivalence tests
# ---------------------------------------------------------------------------

def test_indicators_no_lookahead(merged_bundle):
    merged, _ = merged_bundle
    full = add_all_indicators(merged)
    cols = [c for c in full.columns if c not in merged.columns]

    cutoff = 100
    truncated = add_all_indicators(merged.iloc[:cutoff])
    last_date = truncated.index[-1]

    pd.testing.assert_series_equal(
        full.loc[last_date, cols].astype(float),
        truncated.loc[last_date, cols].astype(float),
        check_names=False,
    )


def test_rolling_stats_no_lookahead(merged_bundle):
    merged, _ = merged_bundle
    full = add_return_rolling_stats(merged)
    cols = [c for c in full.columns if c not in merged.columns]

    cutoff = 100
    truncated = add_return_rolling_stats(merged.iloc[:cutoff])
    last_date = truncated.index[-1]

    pd.testing.assert_series_equal(
        full.loc[last_date, cols].astype(float),
        truncated.loc[last_date, cols].astype(float),
        check_names=False,
    )


def test_cross_asset_no_lookahead(merged_bundle):
    merged, gold_close = merged_bundle
    full = add_comex_mcx_spread_zscore(merged)
    full = add_gold_silver_ratio(full, gold_close=gold_close)
    cols = [c for c in full.columns if c not in merged.columns]

    cutoff = 100
    truncated_input = merged.iloc[:cutoff]
    truncated = add_comex_mcx_spread_zscore(truncated_input)
    truncated = add_gold_silver_ratio(truncated, gold_close=gold_close)
    last_date = truncated.index[-1]

    pd.testing.assert_series_equal(
        full.loc[last_date, cols].astype(float),
        truncated.loc[last_date, cols].astype(float),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 3. Hand-computed sanity checks (catch wrong-direction shifts)
# ---------------------------------------------------------------------------

def test_rsi_is_shifted_exactly_one_bar(merged_bundle):
    """Directly verify add_rsi's output equals the raw `ta` RSI shifted by
    exactly 1 bar — this would fail if the shift direction or amount were
    wrong, even though such a bug would still pass truncation-equivalence."""
    from ta.momentum import RSIIndicator

    merged, _ = merged_bundle
    out = add_rsi(merged, price_col="mcx_close", window=14)

    raw_rsi = RSIIndicator(close=merged["mcx_close"], window=14, fillna=False).rsi()
    expected = raw_rsi.shift(1)

    pd.testing.assert_series_equal(
        out["rsi_14"].astype(float), expected.astype(float), check_names=False
    )


def test_return_rolling_mean_excludes_current_day_return():
    """Hand-built 10-row series: ret_mean_3 at row t must equal the mean of
    the three log-returns realized strictly before t, never including the
    return that resolves ON day t."""
    dates = pd.bdate_range("2024-01-01", periods=10, tz="Asia/Kolkata")
    close = pd.Series([100, 101, 99, 102, 104, 103, 105, 108, 107, 110], index=dates)
    df = pd.DataFrame({"mcx_close": close})

    out = add_return_rolling_stats(df, price_col="mcx_close", windows=(3,))

    log_ret = np.log(close / close.shift(1))
    # Manually compute what ret_mean_3 SHOULD be at the last row: mean of
    # log-returns at t-1, t-2, t-3 (i.e. NOT including the return realized
    # on the last row itself).
    expected_last = log_ret.shift(1).rolling(3).mean().iloc[-1]
    actual_last = out["ret_mean_3"].iloc[-1]

    assert np.isclose(actual_last, expected_last)

    # And explicitly confirm it does NOT equal the "wrong" (leaky) version
    # that would include today's own return.
    leaky_version = log_ret.rolling(3).mean().iloc[-1]
    assert not np.isclose(actual_last, leaky_version)


def test_calendar_features_days_to_expiry_counts_down_within_month():
    dates = pd.bdate_range("2024-01-25", "2024-01-31", tz="Asia/Kolkata")
    df = pd.DataFrame({"mcx_close": range(len(dates))}, index=dates)
    out = add_calendar_features(df)

    # days_to_expiry should be non-increasing as we move through the month.
    assert (out["days_to_expiry"].diff().dropna() <= 0).all()
    # Last business day of January 2024 (Wed 31st) should have 0 days to expiry.
    assert out["days_to_expiry"].iloc[-1] == 0
    # day_of_week matches pandas' own weekday numbering.
    assert list(out["day_of_week"]) == list(dates.dayofweek)


# ---------------------------------------------------------------------------
# 4. Target is deliberately forward-looking — verify that explicitly
# ---------------------------------------------------------------------------

def test_target_forward_return_and_trailing_nans(merged_bundle):
    merged, _ = merged_bundle
    horizon = 2
    out = add_forward_log_return_target(merged, price_col="mcx_close", horizon=horizon)

    close = merged["mcx_close"]
    expected = np.log(close.shift(-horizon) / close)

    pd.testing.assert_series_equal(
        out["target"].astype(float), expected.astype(float), check_names=False
    )

    # The last `horizon` rows must be NaN (no future close exists yet).
    assert out["target"].iloc[-horizon:].isna().all()
    # Every other row should be non-NaN (synthetic data has no gaps).
    assert out["target"].iloc[:-horizon].isna().sum() == 0
