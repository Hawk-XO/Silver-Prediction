"""
models/meta_learner.py

Ridge-regression stacker combining base-model predictions (ARIMA, XGBoost,
optionally LSTM) into a single blended prediction.

Honest caveat (documented, not hidden)
----------------------------------------
For a proper stack you'd fit the meta-learner on OUT-OF-FOLD base-model
predictions (e.g. via nested cross-validation within the training window),
so the meta-learner never sees a base model's in-sample fit. This first
pass fits the meta-learner directly on the base models' TRAIN-set
predictions instead — simpler, and fine for "does the pipeline work"
purposes, but it lets each base model's in-sample fit leak a little
optimism into the blend weights. Tighten this later (e.g. add an internal
train/holdout split before stacking) once the overall pipeline is running
and you're ready to squeeze out that bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class RidgeMetaLearner:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None

    def fit(self, base_predictions: pd.DataFrame, y: pd.Series) -> "RidgeMetaLearner":
        """`base_predictions`: DataFrame, one column per base model, aligned to `y`."""
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(base_predictions.to_numpy(), y.to_numpy())
        return self

    def predict(self, base_predictions: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeMetaLearner.predict called before fit().")
        return np.asarray(self._model.predict(base_predictions.to_numpy()))
