"""
backtest/metrics.py

Evaluation metrics for walk-forward results (backtest.walk_forward.run_walk_forward
output): directional accuracy, RMSE, and Sharpe of a naive sign-based backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of rows where sign(pred) == sign(actual). Rows where actual
    is exactly 0 are excluded (no direction to be right or wrong about)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def naive_backtest_returns(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Strategy return per period = sign(prediction) * actual forward return.
    See backtest/walk_forward.py module docstring for the close-to-close
    execution-timing simplification this implies."""
    position = np.sign(y_pred)
    return position * y_true


def sharpe_ratio(strategy_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    std = strategy_returns.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(strategy_returns.mean() / std * np.sqrt(periods_per_year))


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Bundle of directional accuracy, RMSE, and naive-backtest Sharpe for one prediction column."""
    strat_returns = naive_backtest_returns(y_true, y_pred)
    return {
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "sharpe": sharpe_ratio(strat_returns),
        "n_predictions": int(len(y_true)),
    }


def evaluate_walk_forward_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Given the output of run_walk_forward(), compute metrics for every
    prediction column (arima_pred, xgb_pred, meta_pred, persistence_pred)
    against y_true, side by side.
    """
    pred_cols = [c for c in results.columns if c.endswith("_pred")]
    rows = {}
    for col in pred_cols:
        rows[col.replace("_pred", "")] = evaluate_predictions(results["y_true"], results[col])
    return pd.DataFrame(rows).T
