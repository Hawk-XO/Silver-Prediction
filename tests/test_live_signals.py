"""
tests/test_live_signals.py

Tests data.db.live_signal_to_row() -- the pure transform from
signals.live_predict.LiveSignal to data.db.LiveSignalRow. Deliberately NOT
a round-trip test against a real database: upsert_live_signal() uses
MySQL-specific ON DUPLICATE KEY UPDATE syntax (via
sqlalchemy.dialects.mysql.insert), which doesn't run against SQLite, and
this project doesn't run a real MySQL instance in CI/the build sandbox.
The transform logic (date coercion, NaN confidence -> None) is exactly the
part with real branching to get wrong, so that's what's tested here,
following the same "test the pure function, not the DB round-trip" pattern
already used for upsert_ohlcv() elsewhere in this project.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from data.db import live_signal_to_row
from signals.live_predict import LiveSignal


def _make_live_signal(**overrides) -> LiveSignal:
    defaults = dict(
        date=pd.Timestamp("2026-07-21"),
        predicted_return=0.0123,
        signal="BUY",
        confidence=0.62,
        entry_price=98765.0,
        stop_loss=98000.0,
        target=99500.0,
        n_train_rows=850,
        n_total_rows_available=900,
    )
    defaults.update(overrides)
    return LiveSignal(**defaults)


def test_basic_fields_map_through_unchanged():
    live = _make_live_signal()
    row = live_signal_to_row(live, commodity="SILVERMIC")

    assert row.commodity == "SILVERMIC"
    assert row.signal == "BUY"
    assert row.predicted_return == pytest.approx(0.0123)
    assert row.confidence == pytest.approx(0.62)
    assert row.entry_price == pytest.approx(98765.0)
    assert row.stop_loss == pytest.approx(98000.0)
    assert row.target == pytest.approx(99500.0)
    assert row.n_train_rows == 850
    assert row.n_total_rows_available == 900


def test_pandas_timestamp_coerced_to_plain_date():
    live = _make_live_signal(date=pd.Timestamp("2026-07-21"))
    row = live_signal_to_row(live, commodity="SILVERMIC")

    assert row.date == dt.date(2026, 7, 21)
    assert not isinstance(row.date, pd.Timestamp)


def test_nan_confidence_becomes_none():
    live = _make_live_signal(confidence=float("nan"))
    row = live_signal_to_row(live, commodity="SILVERMIC")

    assert row.confidence is None


def test_none_stop_loss_and_target_pass_through_as_none():
    live = _make_live_signal(stop_loss=None, target=None)
    row = live_signal_to_row(live, commodity="SILVERMIC")

    assert row.stop_loss is None
    assert row.target is None


def test_sell_and_hold_signals_pass_through():
    for sig in ("SELL", "HOLD"):
        row = live_signal_to_row(_make_live_signal(signal=sig), commodity="SILVERMIC")
        assert row.signal == sig


def test_different_commodity_is_used_verbatim():
    row = live_signal_to_row(_make_live_signal(), commodity="GOLDMIC")
    assert row.commodity == "GOLDMIC"
