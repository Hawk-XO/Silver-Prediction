"""
run_phase6_demo.py

End-to-end check: synthetic data -> features -> walk-forward predictions
(Phase 5) -> BUY/SELL/HOLD signals with ATR stops and cooldown (Phase 6)
-> CSV audit log.

Run with:
    python run_phase6_demo.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals, log_signals_to_csv

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)


def main():
    print("Generating synthetic dataset and building features (Phases 1-3)...")
    bundle = generate_full_synthetic_dataset(start_date="2023-01-01", n_days=600)
    continuous = build_continuous_series(bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(continuous, bundle["comex_silver"], bundle["usdinr"], bundle["dxy"])
    gold_close = bundle["comex_gold"]["close"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_df = build_feature_matrix(merged, gold_close=gold_close, horizon=1)

    feature_cols = get_feature_columns(features_df)
    model_ready = features_df.dropna(subset=feature_cols).copy()

    print("Running walk-forward validation (Phase 5)...")
    config = WalkForwardConfig(
        horizon=1,
        min_train_size=120,
        arima_exog_cols=["ret_mean_5", "comex_mcx_spread_z_10"],
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    wf_results = run_walk_forward(model_ready, feature_cols=feature_cols, target_col="target", config=config)
    print(f"  {len(wf_results)} walk-forward predictions generated.")

    # --- Phase 6: signals need predictions + ATR/vol/price context side by
    # side. Join the walk-forward predictions back onto the feature matrix
    # (same dates) to get atr_14 / ret_std_20 / mcx_close alongside meta_pred.
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )

    print("\nGenerating signals (Phase 6)...")
    # NOTE on confidence_threshold: 0.5 (the SignalConfig default) means
    # "only trade when the predicted move is at least half of typical daily
    # volatility" — a reasonable default, but whether it's reachable at all
    # depends on how large this particular model's predictions are relative
    # to that data's volatility scale. Check empirically before assuming a
    # threshold is well-calibrated:
    #     from signals.signal_engine import compute_confidence
    #     compute_confidence(signal_input, "meta_pred", "ret_std_20").describe()
    # On this synthetic run the meta-learner's predictions are small relative
    # to daily vol (median confidence ~0.016), so 0.5 would produce zero
    # trades — not a bug, just this model+data combination being
    # low-conviction. We use a threshold near this data's own 75th
    # percentile so the demo actually produces some BUY/SELL signals to
    # inspect; tune it against your real data's confidence distribution.
    signal_config = SignalConfig(
        pred_col="meta_pred",
        confidence_threshold=0.025,
        atr_regime_window=60,
        atr_regime_percentile=0.80,
        stop_loss_atr_mult=1.5,
        target_atr_mult=2.5,
        cooldown_days=3,
    )
    signals_df = generate_signals(signal_input, signal_config)

    counts = signals_df["signal"].value_counts()
    print("\nSignal counts:")
    print(counts.to_string())
    print(f"\nSignals blocked by cooldown: {int(signals_df['cooldown_blocked'].sum())}")
    print(f"Rows suppressed by high-vol regime: {int(signals_df['high_vol_regime'].sum())}")

    print("\nSample of BUY/SELL signals (first 5):")
    trades = signals_df[signals_df["signal"] != "HOLD"]
    print(trades[["prediction", "confidence", "signal", "entry_price", "stop_loss", "target"]].head())

    # --- Log every signal with its full feature snapshot to CSV ---
    feature_snapshot = model_ready.loc[signals_df.index, feature_cols]
    out_path = log_signals_to_csv(signals_df, feature_snapshot, path="outputs/signal_log.csv")
    print(f"\nFull signal log (with feature snapshots) written to: {out_path}")

    return signals_df


if __name__ == "__main__":
    main()
