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

import pandas as pd

from data.db import load_ohlcv
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from data.global_factors import fetch_all_global_factors, fetch_comex_gold
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold
from broker.kite_paper_broker import PaperKiteBroker

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


def main(start_date: str | None = None):
    print("Loading real data from MySQL (proxy + kite_api rows)...")
    multi_contract = load_ohlcv()
    if multi_contract.empty:
        raise RuntimeError(
            "mcx_silver_ohlcv is empty — run data/build_merged_history.py "
            "and/or data/kite_fetcher.py first."
        )
    if start_date:
        multi_contract = multi_contract[multi_contract.index >= pd.Timestamp(start_date)]
        if multi_contract.empty:
            raise RuntimeError(f"No rows on/after --start-date {start_date}.")

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
    # NOTE: 0.025 was tuned against synthetic.py's prediction magnitude, not
    # real data's. Check the signal value_counts below — if it's almost all
    # HOLD or almost no HOLD at all, re-tune this threshold against the
    # actual distribution of meta_pred on this real dataset before trusting
    # the backtest that follows.
    signal_config = SignalConfig(confidence_threshold=0.025, cooldown_days=3)
    signals_df = generate_signals(signal_input, signal_config)
    print(signals_df["signal"].value_counts().to_string())

    # --- Phase 7a: vectorbt backtest vs. buy-and-hold ---
    print("\nRunning vectorbt backtest vs. buy-and-hold (Phase 7)...")
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)
    comparison = compare_to_buy_and_hold(signals_df, price_col="entry_price", config=bt_config)
    print("\n=== Performance report: strategy vs. buy-and-hold (REAL DATA) ===")
    print(comparison)

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
        help="Only use data on/after this date (YYYY-MM-DD). Useful for a "
             "quick ~6-month test run before committing to the full "
             "multi-year walk-forward, which retrains daily and can take a "
             "long time over years of history. "
             "Example: --start-date 2026-01-13 for roughly the last 6 months.",
    )
    args = parser.parse_args()
    main(start_date=args.start_date)
