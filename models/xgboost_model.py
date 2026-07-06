"""
models/xgboost_model.py

Thin wrapper around xgboost.XGBRegressor. Takes the engineered feature
columns (see features.pipeline.get_feature_columns) plus, optionally, an
ARIMA-residual column appended by the caller before fit/predict — the
Phase 4 plan calls for "XGBoost + ARIMA residuals as input", which the
walk-forward harness wires up by computing ARIMA's prediction first, then
adding it as an extra feature column here.

Small default hyperparameters (shallow trees, modest n_estimators) so a
walk-forward loop that refits every fold stays fast for a first pass —
tune upward once the pipeline is confirmed working end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=1,
    verbosity=0,
    random_state=42,
)


class XGBoostModel:
    def __init__(self, feature_cols: list[str], **xgb_params):
        self.feature_cols = list(feature_cols)
        params = dict(DEFAULT_PARAMS)
        params.update(xgb_params)
        self.params = params
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(X[self.feature_cols], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostModel.predict called before fit().")
        return np.asarray(self._model.predict(X[self.feature_cols]))
