"""
models/arima_model.py

Thin wrapper around statsmodels' SARIMAX, fit directly on the (already
stationary-ish) log-return target, with optional exogenous regressors drawn
from the engineered feature set (e.g. USDINR/COMEX-derived features).

Kept deliberately simple for a first working pass — low-order ARIMA, no
seasonal terms. Tune `order` once the pipeline is running end to end.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """fit/predict wrapper matching the interface used across models/.

    Parameters
    ----------
    order : tuple[int, int, int]
        (p, d, q). Default (1, 0, 0) — the target is already a return
        series, so d=0 is appropriate; keep p/q small to stay fast inside
        a per-fold walk-forward loop.
    exog_cols : list[str] | None
        Feature columns to use as exogenous regressors. None/[] = pure
        univariate ARIMA on the target series.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0), exog_cols: list[str] | None = None):
        self.order = order
        self.exog_cols = exog_cols or []
        self._result = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMAModel":
        exog = X[self.exog_cols].to_numpy() if self.exog_cols else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels is chatty about convergence on short series
            model = SARIMAX(
                y.to_numpy(),
                exog=exog,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("ARIMAModel.predict called before fit().")
        exog = X[self.exog_cols].to_numpy() if self.exog_cols else None
        forecast = self._result.get_forecast(steps=len(X), exog=exog)
        return np.asarray(forecast.predicted_mean)
