"""
backtest/walk_forward.py

Expanding-window walk-forward validation harness, per PROJECT_NOTES.md
Section 4:

  - expanding window: train on all rows up to day N, test exclusively on
    day N+1, roll forward one day at a time.
  - purging: drop any training row whose label window overlaps the test
    period.
  - embargo: buffer of `h` days after each test point before a row becomes
    eligible for training again.
  - scalers refit per fold, train-only.

Purge + embargo, unified
-------------------------
Row `t`'s target uses close[t+h] (see features/target.py) — its label
window is [t, t+h]. For a test point at position T, a training row j is
SAFE to train on only if its entire label window resolves strictly before
T, i.e. `j + h < T`. This single condition does double duty:
  - PURGE: excludes rows whose label overlaps the test window itself.
  - EMBARGO: as T advances fold-to-fold, row T-1 (etc.) only re-enters
    eligibility once a later T' satisfies T-1 + h < T', i.e. after an
    `h`-day gap — exactly the embargo buffer PROJECT_NOTES.md asks for.

So `valid_train_idx(T) = [0, 1, ..., T-h-1]` for every fold; we don't need
a second, separate embargo step.

Execution-timing simplification (documented)
----------------------------------------------
Section 4 also notes execution should respect next-open timing (a signal
from day t's close executes at day t+1's open, never at t's own close).
This first-pass harness evaluates a simplified close-to-close naive
backtest (position sized by predicted-return sign, PnL = position *
actual forward return) rather than modeling open-price fills — good enough
to sanity-check that predictions carry directional signal; refine in
Phase 6/7 once real execution mechanics (signals/, broker wrapper) exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.arima_model import ARIMAModel
from models.xgboost_model import XGBoostModel
from models.meta_learner import RidgeMetaLearner


@dataclass
class WalkForwardConfig:
    horizon: int = 1
    min_train_size: int = 100
    arima_order: tuple[int, int, int] = (1, 0, 0)
    arima_exog_cols: list[str] = field(default_factory=list)
    xgb_params: dict = field(default_factory=dict)
    meta_alpha: float = 1.0


def valid_train_indices(test_idx: int, horizon: int) -> np.ndarray:
    """Rows [0, test_idx - horizon - 1] — see module docstring for the
    purge+embargo derivation."""
    upper = test_idx - horizon  # exclusive
    if upper <= 0:
        return np.array([], dtype=int)
    return np.arange(0, upper)


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """
    Run the expanding-window walk-forward loop over `df` (already
    NaN-free in `feature_cols` + `target_col` — see
    features_ready_for_walk_forward() below to prepare it).

    Returns
    -------
    pd.DataFrame indexed by the same dates as the test rows, columns:
        y_true, arima_pred, xgb_pred, meta_pred, persistence_pred,
        n_train_rows
    """
    cfg = config or WalkForwardConfig()
    n = len(df)
    y = df[target_col]

    records = []
    for test_idx in range(cfg.min_train_size, n):
        if pd.isna(y.iloc[test_idx]):
            continue  # tail rows with no realized future close yet

        train_idx = valid_train_indices(test_idx, cfg.horizon)
        if len(train_idx) < cfg.min_train_size:
            continue

        train_df = df.iloc[train_idx]
        test_row = df.iloc[[test_idx]]

        X_train, y_train = train_df[feature_cols], train_df[target_col]

        # --- Scalers refit per fold, train-only (Section 4 requirement) ---
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(test_row[feature_cols]), columns=feature_cols, index=test_row.index
        )

        # --- Base model 1: ARIMA (exog = a small subset of scaled features) ---
        arima = ARIMAModel(order=cfg.arima_order, exog_cols=cfg.arima_exog_cols)
        arima.fit(X_train_scaled, y_train)
        arima_train_pred = arima.predict(X_train_scaled)
        arima_test_pred = arima.predict(X_test_scaled)[0]

        # --- Base model 2: XGBoost, with ARIMA's train prediction as an extra feature ---
        X_train_xgb = X_train_scaled.copy()
        X_train_xgb["arima_pred"] = arima_train_pred
        X_test_xgb = X_test_scaled.copy()
        X_test_xgb["arima_pred"] = arima_test_pred

        xgb_feature_cols = feature_cols + ["arima_pred"]
        xgb_model = XGBoostModel(feature_cols=xgb_feature_cols, **cfg.xgb_params)
        xgb_model.fit(X_train_xgb, y_train)
        xgb_train_pred = xgb_model.predict(X_train_xgb)
        xgb_test_pred = xgb_model.predict(X_test_xgb)[0]

        # --- Meta-learner: Ridge stack of ARIMA + XGBoost (see caveat in models/meta_learner.py) ---
        base_train_preds = pd.DataFrame(
            {"arima": arima_train_pred, "xgb": xgb_train_pred}, index=X_train.index
        )
        base_test_preds = pd.DataFrame(
            {"arima": [arima_test_pred], "xgb": [xgb_test_pred]}, index=test_row.index
        )
        meta = RidgeMetaLearner(alpha=cfg.meta_alpha)
        meta.fit(base_train_preds, y_train)
        meta_test_pred = meta.predict(base_test_preds)[0]

        # --- Persistence baseline: last label whose window is itself
        # already safely resolved (same purge boundary as training) ---
        persistence_source_idx = test_idx - cfg.horizon - 1
        persistence_pred = (
            y.iloc[persistence_source_idx] if persistence_source_idx >= 0 else np.nan
        )

        records.append(
            {
                "date": df.index[test_idx],
                "y_true": y.iloc[test_idx],
                "arima_pred": arima_test_pred,
                "xgb_pred": xgb_test_pred,
                "meta_pred": meta_test_pred,
                "persistence_pred": persistence_pred,
                "n_train_rows": len(train_idx),
            }
        )

    results = pd.DataFrame.from_records(records).set_index("date")
    return results
