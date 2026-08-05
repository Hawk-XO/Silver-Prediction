"""
export_predictor_data.py

Runs the project's REAL pipeline (features -> walk-forward ARIMA + XGBoost +
meta-learner -> signal engine -> vectorbt backtest) against the project's
own synthetic data generator (data/synthetic.py) and exports the results as
JSON for predictor_ui/silver_options_predictor.jsx to embed and render.

Why synthetic data: there's no live MCX feed reachable from this sandbox
(no Kite credentials wired to a real broker session here, no outbound
access to Zerodha's API). data/synthetic.py is the project's own built-in
data generator, used elsewhere (run_phase7_demo.py etc.) to exercise the
pipeline end-to-end -- this reuses exactly that, not a separate fake.

Everything downstream of the synthetic prices is the REAL pipeline: real
feature engineering, real ARIMA/XGBoost/meta-learner walk-forward fitting
(purge/embargo-safe, same as backtest/walk_forward.py's tested fold logic),
real signal generation, real vectorbt backtest vs. buy-and-hold. Nothing
about the modeling or backtest math is faked or simplified for the UI.

Honest expectation on synthetic data: data/synthetic.py's own docstring
says it has "no genuine predictive structure" (pure random walk). ~50%
directional accuracy on it is the CORRECT result -- it's what proves the
walk-forward harness isn't leaking future information into training. A
much-better-than-50% number here would actually be a red flag, not a win.

Run with:
    python export_predictor_data.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic import generate_full_synthetic_dataset
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold

OUT_PATH = Path(__file__).parent / "predictor_ui" / "predictor_data.json"


def main() -> None:
    print("Building features and walk-forward predictions...")
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

    print("Generating signals...")
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    signal_config = SignalConfig(confidence_threshold=0.025, cooldown_days=3)
    signals_df = generate_signals(signal_input, signal_config)
    print(signals_df["signal"].value_counts().to_string())

    print("Running vectorbt backtest vs. buy-and-hold...")
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)
    comparison = compare_to_buy_and_hold(signals_df, price_col="entry_price", config=bt_config)

    # --- Build the chart series: actual close price vs. the model's ---
    # --- predicted price, per the label definition in features/target.py:
    # ---     target[t] = log(close[t+h]) - log(close[t])
    # --- so predicted_price[t+h] = close[t] * exp(meta_pred[t]).
    actual_close = model_ready["mcx_close"]
    horizon = wf_config.horizon

    rows = []
    correct_direction = 0
    total_scored = 0
    abs_pct_errors = []

    for t, pred_row in wf_results.iterrows():
        t_pos = actual_close.index.get_loc(t)
        target_pos = t_pos + horizon
        if target_pos >= len(actual_close):
            continue  # no realized future price yet to compare against
        target_date = actual_close.index[target_pos]

        base_price = float(actual_close.iloc[t_pos])
        meta_pred = float(pred_row["meta_pred"])
        predicted_price = base_price * float(np.exp(meta_pred))
        actual_future_price = float(actual_close.iloc[target_pos])

        pred_direction = "up" if meta_pred > 0 else ("down" if meta_pred < 0 else "flat")
        actual_direction = "up" if actual_future_price > base_price else (
            "down" if actual_future_price < base_price else "flat"
        )
        is_correct = pred_direction == actual_direction and pred_direction != "flat"
        if pred_direction != "flat" and actual_direction != "flat":
            total_scored += 1
            if is_correct:
                correct_direction += 1

        abs_pct_errors.append(abs(actual_future_price - predicted_price) / actual_future_price)

        rows.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "actualPrice": round(actual_future_price, 2),
            "predictedPrice": round(predicted_price, 2),
            "baseDate": t.strftime("%Y-%m-%d"),
            "basePrice": round(base_price, 2),
            "signal": str(signals_df.loc[t, "signal"]) if t in signals_df.index else "HOLD",
            "predictedReturn": round(meta_pred, 6),
        })

    directional_accuracy = (correct_direction / total_scored) if total_scored else None
    avg_abs_pct_error = float(np.mean(abs_pct_errors)) if abs_pct_errors else None

    # --- Latest real signal (last row of signals_df -- the most recent ---
    # --- date the pipeline had features for) ---
    last_date = signals_df.index.max()
    last_row = signals_df.loc[last_date]
    latest_signal = {
        "date": last_date.strftime("%Y-%m-%d"),
        "signal": str(last_row["signal"]),
        "entryPrice": round(float(last_row["entry_price"]), 2),
        "stopLoss": None if pd.isna(last_row.get("stop_loss")) else round(float(last_row["stop_loss"]), 2),
        "target": None if pd.isna(last_row.get("target")) else round(float(last_row["target"]), 2),
        "predictedReturn": round(float(wf_results.loc[last_date, "meta_pred"]), 6)
                            if last_date in wf_results.index else None,
    }

    # --- Backtest comparison table (strategy vs. buy-and-hold) ---
    comparison_clean = comparison.replace({np.nan: None})
    backtest_comparison = {
        metric: {
            "strategy": comparison_clean.loc["strategy", metric],
            "buyAndHold": comparison_clean.loc["buy_and_hold", metric],
        }
        for metric in comparison_clean.columns
    }

    output = {
        "generatedFrom": "real pipeline (features -> ARIMA+XGBoost+meta-learner walk-forward -> signal engine -> vectorbt backtest) on data/synthetic.py",
        "nPredictions": len(rows),
        "directionalAccuracy": directional_accuracy,
        "avgAbsPctError": avg_abs_pct_error,
        "series": rows,
        "latestSignal": latest_signal,
        "signalCounts": signals_df["signal"].value_counts().to_dict(),
        "backtestComparison": backtest_comparison,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")
    print(f"Directional accuracy: {directional_accuracy:.1%}" if directional_accuracy else "N/A")
    print(f"Latest signal: {latest_signal}")


if __name__ == "__main__":
    main()
