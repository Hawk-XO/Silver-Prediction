"""
run_real_data_pipeline.py

Same Phase 1-7 pipeline as run_phase7_demo.py, but sourced from REAL data:
  - data/db.py (MySQL: calibrated COMEX+USDINR proxy for deep history,
    real Kite Connect MCX prices for anything recent) instead of
    data/synthetic.py
  - Live COMEX Silver / Gold / USD/INR / DXY via data/global_factors.py
    instead of the synthetic bundle's fabricated versions

Everything downstream (feature engineering, walk-forward, signals, vectorbt
backtest, paper broker replay) is unchanged from run_phase7_demo.py — this
is intentional. The synthetic demo existed to validate the *pipeline logic*
in isolation from data-quality questions; this script is where we find out
what the same logic actually says about real MCX Silver.

Expect the numbers here to look very different from the synthetic demo —
that's the point. Real markets don't have the clean statistical properties
synthetic.py was generated with, so:
  - Walk-forward accuracy/hit-rate will likely be lower.
  - The confidence_threshold tuned for synthetic data (0.025, see
    run_phase7_demo.py's comment) may need re-tuning here — check the
    signal value_counts output; if almost everything is HOLD or almost
    nothing is, the threshold is miscalibrated for real data's actual
    prediction magnitude.
  - The 2015-2025 portion of the series is a *calibrated proxy*, not exact
    real MCX prices (see data/build_merged_history.py) — treat backtest
    results over that stretch as directionally informative, not precise.

Run with:
    python run_real_data_pipeline.py
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from data.db import load_ohlcv
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from data.global_factors import fetch_all_global_factors, fetch_comex_gold
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold, run_strategy_backtest, trade_pnl_breakdown
from backtest.atr_exit_backtest import AtrExitConfig, run_atr_exit_backtest
from broker.kite_paper_broker import PaperKiteBroker
from signals.tune_threshold import grid_search_threshold

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def _strip_tz(df_or_series):
    """MySQL-sourced data (via data/db.py) comes back tz-naive since MySQL's
    DATE column has no timezone concept. Live data from data/global_factors.py
    comes back tz-aware (Asia/Kolkata), since it's fetched fresh via yfinance.
    Pandas refuses to .join() a tz-naive index with a tz-aware one, so we
    normalize everything to tz-naive right before merging — dates already
    represent MCX trading days, the tz info isn't adding real information
    at this point anyway."""
    if df_or_series.index.tz is not None:
        df_or_series = df_or_series.copy()
        df_or_series.index = df_or_series.index.tz_localize(None)
    return df_or_series


def main(
    start_date: str | None = None,
    end_date: str | None = None,
    tune_threshold: bool = False,
    confidence_threshold: float = 0.025,
):
    print("Loading real data from MySQL (proxy + kite_api rows)...")
    multi_contract = load_ohlcv()
    if multi_contract.empty:
        raise RuntimeError(
            "mcx_silver_ohlcv is empty — run data/build_merged_history.py "
            "and/or data/kite_fetcher.py first."
        )
    # IMPORTANT: --start-date must NOT be applied to the raw load directly.
    # Feature warmup (~200 rows) + walk-forward min_train_size (120 rows)
    # need real history strictly BEFORE start_date to produce any
    # predictions at all inside the requested window — filtering the raw
    # load by start_date destroys exactly that lookback (this broke the
    # 2020-03-01 COVID-window test: only 213 raw rows survived, nowhere
    # near enough warmup+training). Instead we load from a buffered
    # earlier date, run the full pipeline including warmup/training on
    # that wider range, and only slice down to [start_date, end_date] once
    # we get to signal generation/backtesting below. --end-date is safe to
    # apply directly to the raw load since nothing downstream needs data
    # AFTER the window being tested.
    LOOKBACK_BUFFER_DAYS = 500  # ~320 trading days of warmup+min_train, plus slack for weekends/holidays
    load_start = pd.Timestamp(start_date) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS) if start_date else None
    if load_start is not None:
        multi_contract = multi_contract[multi_contract.index >= load_start]
        if multi_contract.empty:
            raise RuntimeError(f"No rows on/after {load_start.date()} (buffered from --start-date {start_date}).")
    if end_date:
        multi_contract = multi_contract[multi_contract.index <= pd.Timestamp(end_date)]
        if multi_contract.empty:
            raise RuntimeError(f"No rows on/before --end-date {end_date}.")

    print(f"  {len(multi_contract)} raw rows across "
          f"{multi_contract['contract'].nunique()} contract(s)/segments, "
          f"{multi_contract.index.min().date()} to {multi_contract.index.max().date()}.")
    print(f"  By source: {multi_contract['source'].value_counts().to_dict()}")

    continuous = build_continuous_series(multi_contract)

    range_start = continuous.index.min().strftime("%Y-%m-%d")
    range_end = continuous.index.max().strftime("%Y-%m-%d")
    print(f"\nFetching live global factors ({range_start} to {range_end})...")
    globals_ = fetch_all_global_factors(start=range_start, end=range_end)
    gold_close = fetch_comex_gold(start=range_start, end=range_end)["close"]

    continuous = _strip_tz(continuous)
    comex = _strip_tz(globals_["comex_silver"])
    usdinr = _strip_tz(globals_["usdinr"])
    dxy = _strip_tz(globals_["dxy"])
    gold_close = _strip_tz(gold_close)

    merged = merge_mcx_with_global(continuous, comex, usdinr, dxy)

    print("\nBuilding features and walk-forward predictions (Phases 1-5)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_df = build_feature_matrix(merged, gold_close=gold_close, horizon=1)

    feature_cols = get_feature_columns(features_df)
    model_ready = features_df.dropna(subset=feature_cols).copy()
    print(f"  {len(model_ready)} rows with complete features "
          f"(dropped {len(features_df) - len(model_ready)} warmup/NaN rows).")

    wf_config = WalkForwardConfig(
        horizon=1,
        min_train_size=120,
        arima_exog_cols=["ret_mean_5", "comex_mcx_spread_z_10"],
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    if len(model_ready) <= wf_config.min_train_size:
        raise RuntimeError(
            f"Only {len(model_ready)} rows survived feature warmup (need more "
            f"than min_train_size={wf_config.min_train_size} to generate ANY "
            f"walk-forward predictions). This usually means your MySQL data "
            f"doesn't span enough history yet — run "
            f"data/build_merged_history.py with an earlier --deep-start, or "
            f"check mcx_silver_ohlcv actually has the date range you expect."
        )
    wf_results = run_walk_forward(model_ready, feature_cols=feature_cols, target_col="target", config=wf_config)
    print(f"  {len(wf_results)} walk-forward predictions generated.")

    print("\nGenerating signals (Phase 6)...")
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )

    # Now that warmup + walk-forward training have used the full buffered
    # range, slice down to the actually-requested window for reporting.
    # (This is the step that makes --start-date a report-window filter,
    # not a data-availability filter — see the loading comment above.)
    pre_slice_n = len(signal_input)
    if start_date:
        signal_input = signal_input[signal_input.index >= pd.Timestamp(start_date)]
    if end_date:
        signal_input = signal_input[signal_input.index <= pd.Timestamp(end_date)]
    if (start_date or end_date) and len(signal_input) != pre_slice_n:
        print(f"  ({pre_slice_n} predictions generated total using buffered "
              f"lookback; {len(signal_input)} fall inside the requested "
              f"[{start_date or 'earliest'}, {end_date or 'latest'}] window "
              f"and are used below.)")
    if signal_input.empty:
        raise RuntimeError(
            f"No walk-forward predictions fall inside "
            f"[{start_date or 'earliest'}, {end_date or 'latest'}] — the "
            f"window may be entirely inside the warmup/training lookback. "
            f"Try a start_date further from the earliest available data."
        )

    if tune_threshold:
        print("\n=== Confidence-threshold grid search ===")
        print("(candidates are the 10th/25th/40th/50th/60th/75th/90th "
              "percentiles of THIS dataset's own confidence distribution — "
              "see signals/tune_threshold.py's module docstring for how to "
              "read this table without overfitting the threshold to it)")
        grid = grid_search_threshold(signal_input)
        print(grid.to_string(index=False))
        print(
            "\nPick a threshold from the table above (or a nearby value) "
            "and pass it as --confidence-threshold on your next run — this "
            "run stops here rather than silently picking one for you."
        )
        return grid, None

    # NOTE: 0.025 was tuned against synthetic.py's prediction magnitude, not
    # real data's. Run with --tune-threshold first if you haven't yet re-
    # calibrated this for real data — see signals/tune_threshold.py.
    signal_config = SignalConfig(confidence_threshold=confidence_threshold, cooldown_days=3)
    signals_df = generate_signals(signal_input, signal_config)
    print(signals_df["signal"].value_counts().to_string())

    # --- Phase 7a: vectorbt backtest vs. buy-and-hold ---
    print("\nRunning vectorbt backtest vs. buy-and-hold (Phase 7)...")
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)
    comparison = compare_to_buy_and_hold(signals_df, price_col="entry_price", config=bt_config)
    print("\n=== Performance report: strategy vs. buy-and-hold (REAL DATA) ===")
    print(comparison)

    print("\n=== Trade-level PnL breakdown (strategy only) ===")
    print("(diagnoses whether win-rate% is actually converting to money — "
          "e.g. a >50% win rate can still lose if avg_loss dwarfs avg_win)")
    strategy_pf = run_strategy_backtest(signals_df, price_col="entry_price", config=bt_config)
    breakdown = trade_pnl_breakdown(strategy_pf)
    for k, v in breakdown.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Same signals, but exits enforce stop_loss/target (not just signal changes) ===")
    print("(see backtest/atr_exit_backtest.py docstring for the close-only-price "
          "approximation this makes — compare these stats to the block above, "
          "which only ever exits on a signal flip)")
    atr_result = run_atr_exit_backtest(signals_df, AtrExitConfig(
        fees=bt_config.fees, slippage=bt_config.slippage, init_cash=bt_config.init_cash
    ))
    for k, v in atr_result["stats"].items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")
    print(f"  exit_reason_counts: {atr_result['exit_reason_counts']}")

    # --- Phase 7b: replay the same signals through the paper broker ---
    print("\nReplaying signals through the paper-trading broker...")
    broker = PaperKiteBroker(initial_cash=bt_config.init_cash)
    symbol = "SILVERMICFUT"
    lot_size = 1  # SILVERMIC's real lot size is much smaller than SILVER's — verify against current MCX contract specs before live use
    current_qty = 0

    for date, row in signals_df.iterrows():
        broker.update_market_price(symbol, row["entry_price"])
        desired = {"BUY": lot_size, "SELL": -lot_size, "HOLD": 0}[row["signal"]]
        delta = desired - current_qty
        if delta != 0:
            broker.place_order(
                symbol,
                transaction_type="BUY" if delta > 0 else "SELL",
                quantity=abs(delta),
                price=row["entry_price"],
                tag=str(date.date()),
            )
            current_qty = desired

    broker_pnl = broker.get_pnl()
    print(f"  Paper broker orders placed: {len(broker.get_orders())}")
    print(f"  Paper broker P&L: realised={broker_pnl['realised']:,.2f}  "
          f"unrealised={broker_pnl['unrealised']:,.2f}  total={broker_pnl['total']:,.2f}")

    return comparison, broker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date", default=None,
        help="Only REPORT signals/backtest results on/after this date "
             "(YYYY-MM-DD). Feature warmup and walk-forward training still "
             "use real history from well before this date (a ~500-day "
             "buffer is loaded automatically) so predictions near the start "
             "of your window aren't undertrained — this does NOT shrink "
             "the data used for training, only the reporting window. "
             "Example: --start-date 2020-03-01 --end-date 2020-12-31 to "
             "test just the COVID-crash period.",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Only use data on/before this date (YYYY-MM-DD), and only "
             "report signals/backtest results up to it. Combine with "
             "--start-date to test a bounded window -- e.g. a period "
             "without an outsized external shock, or specifically a period "
             "WITH one -- rather than always running through to today.",
    )
    parser.add_argument(
        "--tune-threshold", action="store_true",
        help="Instead of running the full backtest, grid-search "
             "confidence_threshold against this dataset's own confidence "
             "distribution and print a comparison table, then stop. Run "
             "this first on real data before trusting the default 0.025 "
             "(tuned against synthetic data, not this dataset).",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.025,
        help="SignalConfig.confidence_threshold to use for the full run "
             "(ignored if --tune-threshold is passed). Default 0.025 is "
             "the synthetic-data-tuned value from run_phase7_demo.py — "
             "override this once you've picked a value via --tune-threshold.",
    )
    args = parser.parse_args()
    main(
        start_date=args.start_date,
        end_date=args.end_date,
        tune_threshold=args.tune_threshold,
        confidence_threshold=args.confidence_threshold,
    )
