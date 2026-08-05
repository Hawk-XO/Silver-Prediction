"""
run_phase7_demo.py

End-to-end check, Phases 1-7: synthetic data -> features -> walk-forward
predictions -> signals -> vectorbt backtest (vs. buy-and-hold) -> replay
the same signals through the paper-trading broker to sanity-check its
accounting against vectorbt's.

Run with:
    python run_phase7_demo.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold
from broker.kite_paper_broker import PaperKiteBroker

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def main():
    print("Building features and walk-forward predictions (Phases 1-5)...")
    bundle = generate_full_synthetic_dataset(start_date="2023-01-01", n_days=600)
    continuous = build_continuous_series(bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(continuous, bundle["comex_silver"], bundle["usdinr"], bundle["dxy"])
    gold_close = bundle["comex_gold"]["close"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_df = build_feature_matrix(merged, gold_close=gold_close, horizon=1)

    feature_cols = get_feature_columns(features_df)
    model_ready = features_df.dropna(subset=feature_cols).copy()

    wf_config = WalkForwardConfig(
        horizon=1,
        min_train_size=120,
        arima_exog_cols=["ret_mean_5", "comex_mcx_spread_z_10"],
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    wf_results = run_walk_forward(model_ready, feature_cols=feature_cols, target_col="target", config=wf_config)
    print(f"  {len(wf_results)} walk-forward predictions generated.")

    print("\nGenerating signals (Phase 6)...")
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    # See run_phase6_demo.py for why 0.025 rather than the 0.5 default —
    # this model's predictions are small relative to this data's daily vol.
    signal_config = SignalConfig(confidence_threshold=0.025, cooldown_days=3)
    signals_df = generate_signals(signal_input, signal_config)
    print(signals_df["signal"].value_counts().to_string())

    # --- Phase 7a: vectorbt backtest vs. buy-and-hold ---
    print("\nRunning vectorbt backtest vs. buy-and-hold (Phase 7)...")
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)
    comparison = compare_to_buy_and_hold(signals_df, price_col="entry_price", config=bt_config)
    print("\n=== Performance report: strategy vs. buy-and-hold ===")
    print(comparison)

    # --- Phase 7b: replay the same signals through the paper broker ---
    # Sanity-checks that the broker's own position/PnL accounting agrees
    # with what the vectorbt backtest implies, using a completely
    # independent code path (no vectorbt involved here at all).
    print("\nReplaying signals through the paper-trading broker...")
    broker = PaperKiteBroker(initial_cash=bt_config.init_cash)
    symbol = "SILVERFUT"
    lot_size = 5  # arbitrary demo lot size — MCX Silver's real lot size varies by contract, check current specs
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
    main()
