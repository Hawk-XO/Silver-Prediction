"""
signals/tune_threshold.py

Grid-searches SignalConfig.confidence_threshold against real walk-forward
predictions and reports how each candidate value shifts (a) the raw
BUY/SELL/HOLD distribution and (b) full-period backtest metrics vs.
buy-and-hold.

Why this exists
----------------
run_real_data_pipeline.py hardcoded confidence_threshold=0.025, a value
tuned against synthetic.py's prediction magnitude (see that script's
comment). On real data this produced 139 BUY / 11 HOLD out of 150 rows --
a threshold that low accepts almost any nonzero prediction as "confident",
so the signal barely discriminates. This module anchors the search grid to
the ACTUAL distribution of confidence = |meta_pred| / ret_std_20 on this
dataset, instead of guessing more round numbers.

Honest caveat -- read this before trusting a chosen threshold
---------------------------------------------------------------
Picking the threshold that maximizes backtest return on the SAME data
you're about to report results on is not stronger evidence of a working
strategy -- it's a different kind of overfitting than the model itself
has. Use this grid to:
  1. rule out obviously broken values (near-all-HOLD, near-all-one-side),
  2. see how sensitive the backtest is to this one knob (if total_return
     swings wildly across nearby threshold values, that instability is
     itself a warning sign, not just noise to average away), and
  3. shortlist 2-3 reasonable candidates.
Then validate the shortlist OUT of this sample -- on a period/date range
that was not used to pick the threshold -- before trusting it. Don't just
read off the single best row and call it done.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold


def grid_search_threshold(
    signal_input: pd.DataFrame,
    thresholds: list[float] | None = None,
    cooldown_days: int = 3,
    bt_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    signal_input : pd.DataFrame
        Same shape run_real_data_pipeline.py builds: meta_pred joined with
        atr_14, ret_std_20, mcx_close.
    thresholds : list[float] | None
        Candidate confidence_threshold values to try. If None, derived from
        the 10th/25th/40th/50th/60th/75th/90th percentiles of this
        dataset's own confidence distribution -- so the grid is anchored to
        what "confident" actually means for real data, not synthetic data.
    cooldown_days : int
        Passed through to SignalConfig unchanged for every candidate.
    bt_config : BacktestConfig | None
        Passed through to compare_to_buy_and_hold unchanged for every
        candidate. Defaults match run_real_data_pipeline.py's settings.

    Returns
    -------
    pd.DataFrame, one row per threshold, sorted ascending by threshold.
    """
    if thresholds is None:
        conf = (
            signal_input["meta_pred"].abs() / signal_input["ret_std_20"]
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if conf.empty:
            raise ValueError(
                "Couldn't compute a confidence distribution from "
                "signal_input -- check meta_pred/ret_std_20 aren't all NaN."
            )
        quantiles = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
        thresholds = sorted(set(round(float(q), 5) for q in np.quantile(conf, quantiles)))

    bt_config = bt_config or BacktestConfig(
        fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000
    )

    rows = []
    for t in thresholds:
        cfg = SignalConfig(confidence_threshold=t, cooldown_days=cooldown_days)
        signals_df = generate_signals(signal_input, cfg)
        counts = signals_df["signal"].value_counts()
        comparison = compare_to_buy_and_hold(
            signals_df, price_col="entry_price", config=bt_config
        )
        rows.append({
            "confidence_threshold": t,
            "buy": int(counts.get("BUY", 0)),
            "sell": int(counts.get("SELL", 0)),
            "hold": int(counts.get("HOLD", 0)),
            "strategy_return_pct": comparison.loc["strategy", "total_return_pct"],
            "buy_hold_return_pct": comparison.loc["buy_and_hold", "total_return_pct"],
            "strategy_sharpe": comparison.loc["strategy", "sharpe_ratio"],
            "strategy_max_dd_pct": comparison.loc["strategy", "max_drawdown_pct"],
            "total_trades": comparison.loc["strategy", "total_trades"],
        })

    return pd.DataFrame(rows).sort_values("confidence_threshold").reset_index(drop=True)
