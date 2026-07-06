"""
tests/test_signal_engine.py

Phase 6 tests for signals/signal_engine.py:
  1. Confidence gating — low-confidence predictions produce HOLD.
  2. Volatility regime filter — suppresses signals in high-ATR regimes even
     when confidence is high.
  3. Cooldown — blocks a flip (BUY->SELL or SELL->BUY) within the cooldown
     window, but allows it once the window has passed, and never blocks
     same-direction re-affirmation.
  4. ATR-based stop-loss/target levels are computed correctly and only
     attached to BUY/SELL rows (never HOLD).
  5. No lookahead in the volatility-regime rolling threshold (truncation
     equivalence, same technique as tests/test_features_no_leakage.py).
  6. CSV logging writes the expected rows/columns.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from signals.signal_engine import (
    SignalConfig,
    generate_signals,
    compute_confidence,
    compute_volatility_regime_flag,
    log_signals_to_csv,
)


def _make_df(n=40, seed=0, atr_value=500.0, vol_value=0.008):
    dates = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "meta_pred": rng.normal(0, 0.01, n),
            "ret_std_20": np.full(n, vol_value),
            "atr_14": np.full(n, atr_value),
            "mcx_close": np.linspace(75000, 76000, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Confidence gating
# ---------------------------------------------------------------------------

def test_low_confidence_forces_hold():
    df = _make_df(n=20)
    # Tiny prediction relative to vol -> confidence near zero -> must HOLD.
    df["meta_pred"] = 0.0001
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=5)
    result = generate_signals(df, cfg)
    assert (result["signal"] == "HOLD").all()


def test_high_confidence_and_normal_regime_trades():
    df = _make_df(n=20)
    df["meta_pred"] = 0.05  # huge relative to vol=0.008 -> confidence >> threshold
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=5, cooldown_days=0)
    result = generate_signals(df, cfg)
    # First few rows have no rolling ATR threshold yet -> NaN comparison ->
    # high_vol_regime False by construction (flat ATR here anyway), so all
    # should trade BUY (positive prediction).
    assert (result["signal"] == "BUY").all()


def test_compute_confidence_handles_zero_volatility():
    df = _make_df(n=5)
    df["ret_std_20"] = 0.0
    conf = compute_confidence(df, "meta_pred", "ret_std_20")
    assert conf.isna().all()  # division by zero -> NaN, not inf/crash


# ---------------------------------------------------------------------------
# 2. Volatility regime filter
# ---------------------------------------------------------------------------

def test_high_atr_regime_suppresses_signal_despite_high_confidence():
    df = _make_df(n=30)
    df["meta_pred"] = 0.05  # would otherwise easily clear confidence threshold
    # Normal ATR for the first 20 rows, then a large spike for the rest.
    df.loc[df.index[:20], "atr_14"] = 400.0
    df.loc[df.index[20:], "atr_14"] = 4000.0  # 10x spike -> clearly "high vol regime"

    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=10, atr_regime_percentile=0.8, cooldown_days=0)
    result = generate_signals(df, cfg)

    # Rows just after the spike starts should be flagged high-vol and
    # forced to HOLD — check early in the spike, before the rolling window
    # itself becomes saturated with spike values (at which point 4000
    # stops looking anomalous relative to its own recent history).
    spike_rows = result.iloc[20:23]
    assert spike_rows["high_vol_regime"].all()
    assert (spike_rows["signal"] == "HOLD").all()


def test_volatility_regime_no_lookahead():
    """Truncation-equivalence: the rolling ATR-percentile threshold at row t
    must not change if we truncate the series to end right after t."""
    df = _make_df(n=50, seed=3)
    df["atr_14"] = np.linspace(300, 900, 50)  # trending ATR, not flat

    normalized_full, threshold_full, flag_full = compute_volatility_regime_flag(
        df, "atr_14", "mcx_close", window=10, percentile=0.8
    )

    cutoff = 35
    truncated = df.iloc[:cutoff]
    normalized_t, threshold_t, flag_t = compute_volatility_regime_flag(
        truncated, "atr_14", "mcx_close", window=10, percentile=0.8
    )

    last_date = truncated.index[-1]
    assert np.isclose(threshold_full.loc[last_date], threshold_t.loc[last_date], equal_nan=True)
    assert flag_full.loc[last_date] == flag_t.loc[last_date]


# ---------------------------------------------------------------------------
# 3. Cooldown
# ---------------------------------------------------------------------------

def test_cooldown_blocks_immediate_flip():
    dates = pd.bdate_range("2024-01-01", periods=6, tz="Asia/Kolkata")
    # BUY, then immediately want to SELL (flip) — should be blocked within cooldown.
    df = pd.DataFrame(
        {
            "meta_pred": [0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
            "ret_std_20": [0.008] * 6,
            "atr_14": [400.0] * 6,
            "mcx_close": [75000] * 6,
        },
        index=dates,
    )
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=3, cooldown_days=3)
    result = generate_signals(df, cfg)

    assert result["signal"].iloc[0] == "BUY"
    # Next `cooldown_days` rows wanting to flip to SELL should be blocked (HOLD).
    assert (result["signal"].iloc[1:4] == "HOLD").all()
    assert result["cooldown_blocked"].iloc[1:4].all()
    # Once cooldown has elapsed, the SELL should finally go through.
    assert result["signal"].iloc[4] == "SELL"


def test_cooldown_does_not_block_same_direction_reaffirmation():
    dates = pd.bdate_range("2024-01-01", periods=4, tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "meta_pred": [0.05, 0.05, 0.05, 0.05],  # same direction every day
            "ret_std_20": [0.008] * 4,
            "atr_14": [400.0] * 4,
            "mcx_close": [75000] * 4,
        },
        index=dates,
    )
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=2, cooldown_days=3)
    result = generate_signals(df, cfg)
    assert (result["signal"] == "BUY").all()
    assert not result["cooldown_blocked"].any()


# ---------------------------------------------------------------------------
# 4. ATR-based stop-loss / target levels
# ---------------------------------------------------------------------------

def test_stop_loss_and_target_only_set_for_buy_sell():
    df = _make_df(n=20)
    df["meta_pred"] = 0.0001  # everything HOLDs
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=5)
    result = generate_signals(df, cfg)
    assert result.loc[result["signal"] == "HOLD", "stop_loss"].isna().all()
    assert result.loc[result["signal"] == "HOLD", "target"].isna().all()


def test_stop_loss_and_target_direction_correct():
    dates = pd.bdate_range("2024-01-01", periods=2, tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "meta_pred": [0.05, -0.05],
            "ret_std_20": [0.008, 0.008],
            "atr_14": [500.0, 500.0],
            "mcx_close": [75000.0, 75000.0],
        },
        index=dates,
    )
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=1, cooldown_days=0,
                        stop_loss_atr_mult=1.5, target_atr_mult=2.5)
    result = generate_signals(df, cfg)

    buy_row = result.iloc[0]
    assert buy_row["signal"] == "BUY"
    assert buy_row["stop_loss"] == pytest.approx(75000 - 1.5 * 500)
    assert buy_row["target"] == pytest.approx(75000 + 2.5 * 500)

    sell_row = result.iloc[1]
    assert sell_row["signal"] == "SELL"
    assert sell_row["stop_loss"] == pytest.approx(75000 + 1.5 * 500)
    assert sell_row["target"] == pytest.approx(75000 - 2.5 * 500)


# ---------------------------------------------------------------------------
# 5. CSV logging
# ---------------------------------------------------------------------------

def test_log_signals_to_csv_writes_expected_rows_and_columns(tmp_path):
    df = _make_df(n=10)
    cfg = SignalConfig(confidence_threshold=0.5, atr_regime_window=3)
    signals_df = generate_signals(df, cfg)
    feature_snapshot = df[["meta_pred", "ret_std_20", "atr_14", "mcx_close"]]

    out_path = str(tmp_path / "nested" / "signal_log.csv")
    written_path = log_signals_to_csv(signals_df, feature_snapshot, path=out_path)

    assert written_path == out_path
    assert os.path.exists(out_path)

    logged = pd.read_csv(out_path)
    assert len(logged) == len(df)
    for col in ["date", "signal", "confidence", "stop_loss", "target", "mcx_close"]:
        assert col in logged.columns
