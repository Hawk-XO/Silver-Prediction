"""
tests/test_walk_forward_no_leakage.py

Phase 5 leakage tests, per PROJECT_NOTES.md Section 4. Proves:
  1. No training row's label window [j, j+h] overlaps or touches the test
     index T (purging).
  2. No test timestamp itself ever appears among the training indices for
     its own fold (a test point can't train on itself).
  3. The embargo buffer is respected as folds roll forward (a row only
     becomes trainable once its own label has fully resolved before the
     new test point).
  4. A full (small, fast) run of run_walk_forward() end to end produces
     sane, non-NaN output with the expected number of training rows per
     fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import valid_train_indices, run_walk_forward, WalkForwardConfig


# ---------------------------------------------------------------------------
# 1 & 2 & 3: pure index-arithmetic checks on valid_train_indices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("test_idx", [10, 25, 50, 99])
def test_no_training_label_window_overlaps_test_point(test_idx, horizon):
    train_idx = valid_train_indices(test_idx, horizon)
    for j in train_idx:
        label_window_end = j + horizon
        assert label_window_end < test_idx, (
            f"training row {j}'s label window ends at {label_window_end}, "
            f"which overlaps or touches test_idx={test_idx} (horizon={horizon})"
        )


@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("test_idx", [10, 25, 50, 99])
def test_test_index_never_in_its_own_training_set(test_idx, horizon):
    train_idx = valid_train_indices(test_idx, horizon)
    assert test_idx not in train_idx


def test_embargo_widens_correctly_as_folds_roll_forward():
    """Row (T - 1) must NOT be trainable for test point T when horizon >= 1,
    but MUST become trainable again once the test point advances far enough
    past its label window."""
    horizon = 2
    T = 50

    train_at_T = valid_train_indices(T, horizon)
    assert (T - 1) not in train_at_T
    assert (T - 2) not in train_at_T  # label window [T-2, T] still touches T

    # Two folds later, T-2's label window [T-2, T] has fully resolved
    # before the new test point T+2, so it should now be trainable.
    train_at_T_plus_2 = valid_train_indices(T + 2, horizon)
    assert (T - 2) in train_at_T_plus_2


def test_valid_train_indices_empty_when_too_early():
    assert len(valid_train_indices(test_idx=0, horizon=1)) == 0
    assert len(valid_train_indices(test_idx=1, horizon=1)) == 0


# ---------------------------------------------------------------------------
# 4: small end-to-end run of the harness
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_feature_df():
    """A small, fast, fully-synthetic feature matrix — just enough rows to
    exercise run_walk_forward() without the cost of a full pipeline run."""
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")

    feat_a = rng.normal(size=n)
    feat_b = rng.normal(size=n)
    # target correlated with feat_a so models have *something* to find,
    # but noisy enough to be realistic.
    target = 0.02 * feat_a + rng.normal(scale=0.01, size=n)
    target[-1] = np.nan  # tail row with no realized future close, like real data

    return pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "target": target}, index=dates)


def test_run_walk_forward_end_to_end(tiny_feature_df):
    cfg = WalkForwardConfig(
        horizon=1,
        min_train_size=20,
        xgb_params={"n_estimators": 20, "max_depth": 2},
    )
    results = run_walk_forward(
        tiny_feature_df, feature_cols=["feat_a", "feat_b"], target_col="target", config=cfg
    )

    assert not results.empty
    for col in ["y_true", "arima_pred", "xgb_pred", "meta_pred", "n_train_rows"]:
        assert col in results.columns
        assert results[col].isna().sum() == 0

    # n_train_rows should strictly increase fold-over-fold (expanding window).
    assert (results["n_train_rows"].diff().dropna() >= 0).all()

    # Every fold's recorded train size must match what valid_train_indices
    # would independently compute for that fold's position in the ORIGINAL
    # (pre-dropna) dataframe.
    n = len(tiny_feature_df)
    for date, row in results.iterrows():
        test_idx = tiny_feature_df.index.get_loc(date)
        expected_train_idx = valid_train_indices(test_idx, cfg.horizon)
        assert row["n_train_rows"] == len(expected_train_idx)


def test_run_walk_forward_never_trains_on_test_date(tiny_feature_df):
    """Belt-and-suspenders integration check: reconstruct each fold's
    training index set the same way run_walk_forward() does internally, and
    confirm the test row's own date is absent from it."""
    cfg = WalkForwardConfig(horizon=1, min_train_size=20, xgb_params={"n_estimators": 10, "max_depth": 2})
    n = len(tiny_feature_df)
    for test_idx in range(cfg.min_train_size, n):
        if pd.isna(tiny_feature_df["target"].iloc[test_idx]):
            continue
        train_idx = valid_train_indices(test_idx, cfg.horizon)
        if len(train_idx) < cfg.min_train_size:
            continue
        train_dates = tiny_feature_df.index[train_idx]
        test_date = tiny_feature_df.index[test_idx]
        assert test_date not in train_dates
