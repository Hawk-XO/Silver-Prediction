"""
signals/live_predict.py

Track C (auto EOD puller) piece: turns the freshest row in the MySQL store
into today's live BUY/SELL/HOLD signal.

Deliberately reuses the exact same building blocks as the backtest path
(data.pipeline_common.load_real_features, backtest.walk_forward.run_walk_forward,
signals.signal_engine.generate_signals) rather than a separate parallel
implementation -- less code, and it means "live" predictions are produced
by the identical purge/embargo-safe fold logic that's already tested in
tests/test_walk_forward_no_leakage.py, not a second hand-rolled path that
could quietly drift out of sync with it.

"Incremental" caveat
---------------------
This re-runs feature engineering and re-fits ARIMA/XGBoost/the meta-learner
on the *entire* stored history every time, not a true incremental update.
For MCX Silver's data volume (thousands of rows, not millions) a full
refit finishes in well under a minute -- see run_eod_job.py's docstring
for the measured runtime -- so it wasn't worth the added complexity of
warm-starting models from a saved state. If the history grows enough that
this stops being true, that's the place to revisit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.pipeline_common import load_real_features
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals


@dataclass
class LiveSignal:
    date: pd.Timestamp
    predicted_return: float          # meta_pred -- forward log return, horizon days
    signal: str                      # "BUY" / "SELL" / "HOLD"
    confidence: float
    entry_price: float
    stop_loss: float | None
    target: float | None
    n_train_rows: int
    n_total_rows_available: int


def generate_live_signal(
    commodity: str = "SILVER",
    min_train_size: int = 120,
    horizon: int = 1,
    confidence_threshold: float = 0.5,
    cooldown_days: int = 3,
    arima_exog_cols: list[str] | None = None,
    xgb_max_depth: int = 3,
    xgb_n_estimators: int = 100,
) -> LiveSignal:
    """
    Loads the full stored history for `commodity`, fits on everything with
    a resolved label, and predicts the newest (label-less) row.

    Raises ValueError if there isn't enough history yet to clear
    min_train_size -- same failure mode as ui/pipeline_runner.run_pipeline,
    deliberately not auto-shrunk here (a live trading signal shouldn't
    silently run on a shrunk training window the way an exploratory
    backtest run can -- see ui/pipeline_runner.py's auto-clamp comment for
    why that's fine for backtesting but not for this).
    """
    loaded = load_real_features(horizon=horizon, source=None, commodity=commodity)
    model_ready = loaded.model_ready
    feature_cols = loaded.feature_cols

    if len(model_ready) <= min_train_size:
        raise ValueError(
            f"Only {len(model_ready)} rows with complete features -- not enough to clear "
            f"min_train_size={min_train_size} yet. Needs more accumulated history before "
            f"live signals can start (see EOD_JOB.md)."
        )

    usable_exog_cols = [c for c in (arima_exog_cols or []) if c in feature_cols]
    wf_config = WalkForwardConfig(
        horizon=horizon,
        min_train_size=min_train_size,
        arima_exog_cols=usable_exog_cols,
        xgb_params={"n_estimators": xgb_n_estimators, "max_depth": xgb_max_depth},
    )

    # include_live_row=True -- also predict the newest row even though its
    # target is NaN (forward return isn't resolved yet). See
    # backtest/walk_forward.py's docstring for why this is leak-safe.
    wf_results = run_walk_forward(
        model_ready, feature_cols=feature_cols, target_col="target",
        config=wf_config, include_live_row=True,
    )
    if wf_results.empty:
        raise ValueError("Walk-forward produced zero rows, including the live one -- "
                          "unexpected, check the stored data for gaps.")

    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    signal_config = SignalConfig(confidence_threshold=confidence_threshold, cooldown_days=cooldown_days)
    signals_df = generate_signals(signal_input, signal_config)

    live_row = signals_df.iloc[-1]
    live_date = signals_df.index[-1]
    live_train_rows = int(wf_results.iloc[-1]["n_train_rows"])

    return LiveSignal(
        date=live_date,
        predicted_return=float(live_row["prediction"]),
        signal=str(live_row["signal"]),
        confidence=float(live_row["confidence"]) if pd.notna(live_row["confidence"]) else float("nan"),
        entry_price=float(live_row["entry_price"]),
        stop_loss=float(live_row["stop_loss"]) if pd.notna(live_row["stop_loss"]) else None,
        target=float(live_row["target"]) if pd.notna(live_row["target"]) else None,
        n_train_rows=live_train_rows,
        n_total_rows_available=len(model_ready),
    )
