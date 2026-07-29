"""
ui/forecast_runner.py

Backend for the "Forecast" page (ui/pages/3_forecast.py), replacing the
old market-sim/paper-broker page entirely -- no orders, no fake cash, no
P&L. Just: pick a start date, and draw the model's own multi-day-ahead
guess as a second line next to the real price.

Deliberately NOT a reimplementation of the feature/model stack -- reuses
the exact same building blocks as ui/pipeline_runner.py and
backtest/walk_forward.py (load_real_features's Phase 1-3 loading logic,
ARIMAModel + XGBoostModel + RidgeMetaLearner, the same purge/embargo rule).
The one new piece of logic is the roll-forward loop itself:

  1. Fit ARIMA + XGBoost + meta-learner ONCE on everything up to
     `start_date` (a single fit, not walk-forward -- "if I trusted the
     model on this one day, where would it think we're going").
  2. Roll forward `horizon_days` times. Each step: recompute technical/
     rolling-stat/cross-asset features off the price series so far
     (which includes the model's own prior predictions, fed back in as
     if they were real) -> predict the next day's return -> turn that
     into a price -> append it and repeat.
  3. ARIMA's own contribution to each step is computed once, in a single
     multi-step get_forecast() call, since a fitted SARIMAX result
     doesn't need to be "walked" one row at a time (unlike XGBoost's
     purely feature-driven prediction) -- see ARIMAModel.predict()'s
     docstring for why one call with `horizon_days` rows already returns
     the full multi-step path.

Honest limitation (documented, not hidden, and surfaced in the UI caption):
Global factors (COMEX silver, USD/INR, DXY, COMEX gold) are held constant
at their last real value for the whole forecast window, since forecasting
THOSE is a separate problem this page doesn't attempt to solve. Only the
MCX price itself (and everything derived purely from its own history --
EMA/RSI/MACD/ATR/Bollinger/ADX/rolling-return stats) evolves using the
model's own predictions. This is why the predicted line typically gets
smoother/flatter the further out it goes -- expected behavior, not a bug.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data.db import load_ohlcv
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from data.global_factors import fetch_all_global_factors, fetch_comex_gold
from features.pipeline import build_feature_matrix, get_feature_columns
from backtest.walk_forward import valid_train_indices
from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.meta_learner import RidgeMetaLearner
from sklearn.preprocessing import StandardScaler

# Same contract list as pipeline_runner.py, kept in sync deliberately.
AVAILABLE_CONTRACTS = ["SILVER", "SILVERM", "SILVERMIC"]

DEFAULT_ARIMA_EXOG_CHOICES = [
    "ret_mean_5", "ret_mean_10", "ret_mean_20",
    "comex_mcx_spread_z_10", "comex_mcx_spread_z_20", "gold_silver_ratio",
]

MIN_VIABLE_TRAIN_SIZE = 20


def _strip_tz(df_or_series):
    if df_or_series.index.tz is not None:
        df_or_series = df_or_series.copy()
        df_or_series.index = df_or_series.index.tz_localize(None)
    return df_or_series


@dataclass
class ForecastConfig:
    contract: str = "SILVERMIC"
    start_date: str = ""                   # required -- the "real data stops here" cutover point
    horizon_days: int = 20                  # trading days to project forward

    xgb_max_depth: int = 3
    xgb_n_estimators: int = 80
    arima_exog_cols: list[str] = field(default_factory=lambda: ["ret_mean_5", "comex_mcx_spread_z_10"])
    min_train_size: int = 120


@dataclass
class ForecastResult:
    config: ForecastConfig
    actual_price: pd.Series          # real mcx_close, full loaded window
    predicted_price: pd.Series       # from start_date onward (start_date's own point included as the anchor)
    start_date: pd.Timestamp
    n_train_rows: int
    warning: str = ""


def run_forecast(
    cfg: ForecastConfig,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> ForecastResult:
    def _report(message: str, pct: int) -> None:
        if progress_callback is not None:
            progress_callback(message, pct)

    if not cfg.start_date:
        raise ValueError("start_date is required.")
    start_ts = pd.Timestamp(cfg.start_date)

    # --- Phase 1-2: real data load + continuous series + global merge, ---
    # --- same as data.pipeline_common.load_real_features's first half.  ---
    _report("Loading data from MySQL (proxy + kite_api rows)...", 3)
    # Deliberately NOT capped by a recent lookback window (unlike
    # pipeline_runner.py's LOOKBACK_BUFFER_DAYS) -- that buffer exists there
    # because a walk-forward run re-fits once per fold across the whole
    # requested range, so a huge buffer would be wasteful. This page does
    # exactly ONE fit, so it should use every real row available before
    # start_date (e.g. all the way back to 2016), not just the last ~500
    # days -- more training history only helps a single fit, never hurts.
    multi_contract = load_ohlcv()
    if multi_contract.empty:
        raise RuntimeError(
            "mcx_silver_ohlcv is empty -- run data/build_merged_history.py "
            "and/or data/kite_fetcher.py first."
        )
    prefix = multi_contract["contract"].astype(str).str.split("_").str[0]
    multi_contract = multi_contract[prefix == cfg.contract]
    if multi_contract.empty:
        raise RuntimeError(f"No rows for commodity={cfg.contract!r}.")

    continuous = build_continuous_series(multi_contract)
    if continuous.index.max() < start_ts:
        raise ValueError(
            f"No stored data on/after {start_ts.date()} -- latest stored date is "
            f"{continuous.index.max().date()}. Pick an earlier start_date."
        )
    if continuous.index.min() > start_ts:
        raise ValueError(
            f"start_date {start_ts.date()} is before the earliest stored data "
            f"({continuous.index.min().date()})."
        )

    range_start = continuous.index.min().strftime("%Y-%m-%d")
    range_end = continuous.index.max().strftime("%Y-%m-%d")
    _report(f"Fetching global factors ({range_start} to {range_end})...", 8)
    globals_ = fetch_all_global_factors(start=range_start, end=range_end)
    gold_close = fetch_comex_gold(start=range_start, end=range_end)["close"]

    continuous = _strip_tz(continuous)
    comex = _strip_tz(globals_["comex_silver"])
    usdinr = _strip_tz(globals_["usdinr"])
    dxy = _strip_tz(globals_["dxy"])
    gold_close = _strip_tz(gold_close)

    merged = merge_mcx_with_global(continuous, comex, usdinr, dxy)

    # --- Phase 3: build the TRAINING feature matrix, real data only, up ---
    # --- to and including start_date.                                   ---
    _report("Building feature matrix for the training window...", 15)
    merged_train = merged[merged.index <= start_ts].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_train = build_feature_matrix(merged_train, gold_close=gold_close, horizon=1, include_target=True)
    feature_cols = get_feature_columns(features_train)
    model_ready = features_train.dropna(subset=feature_cols).copy()

    if model_ready.empty or model_ready.index.max() < start_ts:
        raise ValueError(
            "After feature warmup (ema_200 alone needs 200 prior bars), there's no "
            "complete-feature row at or before start_date -- pick a later start_date "
            "or make sure enough history is loaded before it."
        )

    effective_min_train_size = cfg.min_train_size
    min_train_auto_adjusted = False
    if len(model_ready) - 2 <= effective_min_train_size:  # -2: purge/embargo trims the tail (see below)
        effective_min_train_size = max(MIN_VIABLE_TRAIN_SIZE, len(model_ready) - 4)
        min_train_auto_adjusted = True
    if len(model_ready) - 2 <= effective_min_train_size:
        raise ValueError(
            f"Only {len(model_ready)} rows with complete features up to {start_ts.date()} -- "
            f"not enough to fit any model even at the {MIN_VIABLE_TRAIN_SIZE}-row floor. "
            f"Pick a later start_date."
        )

    warnings_list = []
    if min_train_auto_adjusted:
        warnings_list.append(
            f"Requested min_train_size={cfg.min_train_size} didn't fit before {start_ts.date()} "
            f"-- auto-reduced to {effective_min_train_size}. Predictions from this few training "
            f"rows are noisier than a longer run; pick an earlier start_date or a smaller "
            f"min_train_size yourself for a more deliberate trade-off."
        )

    usable_exog_cols = [c for c in cfg.arima_exog_cols if c in feature_cols]

    # --- Single fit: train/test split via the exact same purge+embargo ---
    # --- rule walk-forward uses (see backtest/walk_forward.py), with    ---
    # --- test_idx = the LAST row (start_date itself). horizon=1 since   ---
    # --- this is a day-by-day iterative rollout, not a direct N-day-    ---
    # --- ahead single model.                                            ---
    _report("Fitting ARIMA + XGBoost + meta-learner once on the training window...", 25)
    test_idx = len(model_ready) - 1
    train_idx = valid_train_indices(test_idx, horizon=1)
    if len(train_idx) < effective_min_train_size:
        train_idx = np.arange(0, max(test_idx - 1, 0))  # fall back to a 1-row embargo if the strict rule leaves too little

    train_df = model_ready.iloc[train_idx]
    anchor_row = model_ready.iloc[[test_idx]]  # start_date's own row -- features only, not its (leaky) target

    X_train, y_train = train_df[feature_cols], train_df["target"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_anchor_scaled = pd.DataFrame(
        scaler.transform(anchor_row[feature_cols]), columns=feature_cols, index=anchor_row.index
    )

    arima = ARIMAModel(exog_cols=usable_exog_cols)
    arima.fit(X_train_scaled, y_train)
    arima_train_pred = arima.predict(X_train_scaled)

    xgb_feature_cols = feature_cols + ["arima_pred"]
    X_train_xgb = X_train_scaled.copy()
    X_train_xgb["arima_pred"] = arima_train_pred
    xgb_model = XGBoostModel(
        feature_cols=xgb_feature_cols, n_estimators=cfg.xgb_n_estimators, max_depth=cfg.xgb_max_depth,
    )
    xgb_model.fit(X_train_xgb, y_train)
    xgb_train_pred = xgb_model.predict(X_train_xgb)

    meta = RidgeMetaLearner()
    base_train_preds = pd.DataFrame({"arima": arima_train_pred, "xgb": xgb_train_pred}, index=X_train.index)
    meta.fit(base_train_preds, y_train)

    # --- ARIMA's full multi-step path in one call: future exog held ---
    # --- constant at start_date's own (scaled) values -- see module   ---
    # --- docstring's "honest limitation" note.                        ---
    if usable_exog_cols:
        future_exog = pd.concat([X_anchor_scaled[usable_exog_cols]] * cfg.horizon_days, ignore_index=True)
    else:
        future_exog = pd.DataFrame(index=range(cfg.horizon_days))
    arima_future_preds = arima.predict(future_exog)  # length horizon_days, steps 1..horizon_days from start_date

    # --- Day-by-day rollout for the technical/rolling-stat features, ---
    # --- feeding each day's predicted price back in as if real.       ---
    _report("Rolling the forecast forward...", 50)
    future_dates = pd.bdate_range(start=start_ts, periods=cfg.horizon_days + 1)[1:]

    working_merged = merged_train.copy()
    last_comex, last_usdinr, last_dxy = (
        working_merged["comex_close"].iloc[-1], working_merged["usdinr_close"].iloc[-1], working_merged["dxy_close"].iloc[-1],
    )
    last_gold = gold_close.reindex(gold_close.index.union([start_ts])).ffill().loc[start_ts]
    working_gold = gold_close[gold_close.index <= start_ts].copy()

    last_price = working_merged["mcx_close"].iloc[-1]
    predicted_prices = {start_ts: last_price}

    for step, next_date in enumerate(future_dates, start=1):
        if progress_callback is not None:
            pct = 50 + int(45 * step / max(cfg.horizon_days, 1))
            _report(f"Forecasting {next_date.date()} ({step}/{cfg.horizon_days})...", min(pct, 95))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step_features = build_feature_matrix(
                working_merged, gold_close=working_gold, horizon=1, include_target=False,
            )
        last_row = step_features.iloc[[-1]]
        if last_row[feature_cols].isna().any(axis=None):
            # Shouldn't happen (real history already satisfied warmup before
            # start_date) but guard rather than silently feeding NaNs in.
            warnings_list.append(
                f"Stopped early at {next_date.date()} -- a feature went NaN mid-rollout."
            )
            break

        X_step_scaled = pd.DataFrame(
            scaler.transform(last_row[feature_cols]), columns=feature_cols, index=last_row.index
        )
        xgb_row = X_step_scaled.copy()
        xgb_row["arima_pred"] = arima_future_preds[step - 1]
        xgb_pred = xgb_model.predict(xgb_row)[0]
        meta_pred = meta.predict(pd.DataFrame({"arima": [arima_future_preds[step - 1]], "xgb": [xgb_pred]}))[0]

        next_price = last_price * float(np.exp(meta_pred))
        predicted_prices[next_date] = next_price

        working_merged.loc[next_date] = {
            "mcx_open": next_price, "mcx_high": next_price, "mcx_low": next_price, "mcx_close": next_price,
            "mcx_volume": working_merged["mcx_volume"].iloc[-1] if "mcx_volume" in working_merged.columns else np.nan,
            **({"mcx_oi": working_merged["mcx_oi"].iloc[-1]} if "mcx_oi" in working_merged.columns else {}),
            "comex_close": last_comex, "usdinr_close": last_usdinr, "dxy_close": last_dxy,
            "comex_stale": True, "usdinr_stale": True, "dxy_stale": True,
        }
        working_gold.loc[next_date] = last_gold
        last_price = next_price

    _report("Done.", 100)
    predicted_series = pd.Series(predicted_prices).sort_index()
    predicted_series.index.name = "date"

    return ForecastResult(
        config=cfg,
        actual_price=merged["mcx_close"].dropna(),
        predicted_price=predicted_series,
        start_date=start_ts,
        n_train_rows=len(train_df),
        warning=" | ".join(warnings_list),
    )
