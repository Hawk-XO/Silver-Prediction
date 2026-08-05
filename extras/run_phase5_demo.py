"""
run_phase5_demo.py

One-shot script to check the ENTIRE pipeline works end to end, Phase 1
through Phase 5: synthetic data -> continuous series -> merge -> features
-> walk-forward validation -> metrics report.

Run it with:
    python run_phase5_demo.py

Swap in real data by replacing the "1. Build input data" section below with
your actual MCX/COMEX/USDINR/gold loaders (data.mcx_loader.load_mcx_csv,
data.global_factors.fetch_*) — everything downstream is unchanged.

Runtime: ~1-2 minutes on a laptop (refits ARIMA + XGBoost + Ridge on every
fold of the walk-forward loop, ~300 folds by default).
"""

from __future__ import annotations

import warnings

import pandas as pd

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from backtest.metrics import evaluate_walk_forward_results

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.5f}")


def main():
    # --- 1. Build input data (SWAP THIS for real loaders when you have data) ---
    print("Generating synthetic dataset (replace with real MCX/COMEX/USDINR/gold data when ready)...")
    bundle = generate_full_synthetic_dataset(start_date="2023-01-01", n_days=600)
    continuous = build_continuous_series(bundle["mcx_multi_contract"])
    merged = merge_mcx_with_global(
        continuous, bundle["comex_silver"], bundle["usdinr"], bundle["dxy"]
    )
    gold_close = bundle["comex_gold"]["close"]

    # --- 2. Feature engineering (Phase 3) ---
    print("Building features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_df = build_feature_matrix(merged, gold_close=gold_close, horizon=1)

    feature_cols = get_feature_columns(features_df)
    print(f"  {len(feature_cols)} feature columns: {feature_cols}")

    # Drop warm-up NaNs (e.g. ema_200 needs 200 prior bars) up front. Do NOT
    # drop trailing target-NaN rows here — run_walk_forward() already skips
    # test points whose target isn't realized yet.
    model_ready = features_df.dropna(subset=feature_cols).copy()
    print(f"  {len(features_df)} raw rows -> {len(model_ready)} rows after dropping feature warm-up NaNs")

    # --- 3. Walk-forward validation (Phase 5) ---
    print("\nRunning walk-forward validation (this is the slow part)...")
    # A small, fast exogenous set for ARIMA — keep this short, SARIMAX gets
    # slow fast as exog count grows inside a per-fold refit loop.
    arima_exog = ["ret_mean_5", "comex_mcx_spread_z_10"]

    config = WalkForwardConfig(
        horizon=1,
        min_train_size=120,
        arima_order=(1, 0, 0),
        arima_exog_cols=arima_exog,
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    results = run_walk_forward(model_ready, feature_cols=feature_cols, target_col="target", config=config)
    print(f"  {len(results)} walk-forward folds completed "
          f"({results.index[0].date()} to {results.index[-1].date()})")

    # --- 4. Metrics: directional accuracy, RMSE, naive-backtest Sharpe ---
    metrics = evaluate_walk_forward_results(results)
    print("\n=== Walk-forward results: model ensemble vs. persistence baseline ===")
    print(metrics)

    print("\nSample of raw predictions (first 5 folds):")
    print(results.head())

    return results, metrics


if __name__ == "__main__":
    main()
