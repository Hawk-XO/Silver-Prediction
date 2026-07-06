"""
tests/test_vectorbt_backtest.py

Phase 7 tests for backtest/vectorbt_backtest.py:
  1. signals_to_position mapping is correct.
  2. Entries/exits are only flagged on days the position actually changes
     (not every day) — this is what keeps transaction costs realistic.
  3. A trivial all-BUY signal series produces a single long trade held to
     the end (no phantom intermediate trades).
  4. performance_report() returns the required metrics (Sharpe, max
     drawdown, win rate, profit factor) with sane types/ranges.
  5. compare_to_buy_and_hold() runs both legs and returns a 2-row table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.vectorbt_backtest import (
    BacktestConfig,
    signals_to_position,
    _position_to_entries_exits,
    run_strategy_backtest,
    run_buy_and_hold_backtest,
    performance_report,
    compare_to_buy_and_hold,
)


def _make_signals_df(signals: list[str], start_price: float = 75000.0, seed: int = 0) -> pd.DataFrame:
    n = len(signals)
    dates = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    rng = np.random.default_rng(seed)
    # A mildly noisy random walk so trades aren't all flat/zero P&L.
    price = start_price + np.cumsum(rng.normal(0, 50, n))
    return pd.DataFrame({"signal": signals, "entry_price": price}, index=dates)


def test_signals_to_position_mapping():
    signals = pd.Series(["BUY", "SELL", "HOLD", "BUY"])
    pos = signals_to_position(signals)
    assert list(pos) == [1, -1, 0, 1]


def test_entries_exits_only_on_position_change():
    position = pd.Series([0, 1, 1, 1, 0, -1, -1, 0])
    long_entries, long_exits, short_entries, short_exits = _position_to_entries_exits(position)

    assert list(long_entries) == [False, True, False, False, False, False, False, False]
    assert list(long_exits) == [False, False, False, False, True, False, False, False]
    assert list(short_entries) == [False, False, False, False, False, True, False, False]
    assert list(short_exits) == [False, False, False, False, False, False, False, True]


def test_all_buy_produces_single_long_trade():
    signals = ["BUY"] * 20
    df = _make_signals_df(signals)
    pf = run_strategy_backtest(df, config=BacktestConfig(init_cash=1_000_000))
    stats = pf.stats()
    assert stats["Total Trades"] == 1


def test_all_hold_produces_no_trades():
    signals = ["HOLD"] * 20
    df = _make_signals_df(signals)
    pf = run_strategy_backtest(df, config=BacktestConfig(init_cash=1_000_000))
    stats = pf.stats()
    assert stats["Total Trades"] == 0
    # No trading -> capital should be exactly preserved (no fees incurred).
    assert stats["End Value"] == pytest.approx(1_000_000)


def test_performance_report_has_required_metrics():
    signals = (["HOLD"] * 5 + ["BUY"] * 15 + ["HOLD"] * 5 + ["SELL"] * 15 + ["HOLD"] * 5)
    df = _make_signals_df(signals, seed=2)
    pf = run_strategy_backtest(df)
    report = performance_report(pf)

    required = {
        "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
        "profit_factor", "total_trades", "total_fees_paid",
        "avg_margin_pct_of_capital", "max_margin_pct_of_capital",
    }
    assert required.issubset(report.keys())
    assert report["total_trades"] >= 1
    assert report["max_drawdown_pct"] >= 0  # reported as a positive magnitude


def test_compare_to_buy_and_hold_returns_two_rows():
    signals = (["BUY"] * 10 + ["SELL"] * 10 + ["HOLD"] * 10)
    df = _make_signals_df(signals, seed=5)
    comparison = compare_to_buy_and_hold(df)
    assert list(comparison.index) == ["strategy", "buy_and_hold"]
    assert "sharpe_ratio" in comparison.columns


def test_fees_reduce_returns_relative_to_zero_fee_run():
    signals = (["BUY"] * 10 + ["SELL"] * 10 + ["BUY"] * 10)
    df = _make_signals_df(signals, seed=9)

    zero_cost = run_strategy_backtest(df, config=BacktestConfig(fees=0.0, slippage=0.0))
    with_cost = run_strategy_backtest(df, config=BacktestConfig(fees=0.01, slippage=0.01))

    zero_cost_value = zero_cost.stats()["End Value"]
    with_cost_value = with_cost.stats()["End Value"]
    assert with_cost_value < zero_cost_value
