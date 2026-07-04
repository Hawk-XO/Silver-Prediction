"""
features/target.py

Constructs the prediction target: forward log return, per PROJECT_NOTES.md
Section 2:

    target[t] = log(close[t + h]) - log(close[t])

WARNING — this is a LABEL, not a feature
------------------------------------------
`target` is computed using close[t + h], which is information from the
FUTURE relative to row t. That's intentional and correct for a label, but
it means the target column must NEVER be used as a model input feature, and
any row where `target` is NaN (the last `h` rows, where t+h falls past the
end of the data) must be dropped before training — not filled or bfilled.

The walk-forward harness (Phase 5) is responsible for purging training rows
whose label window [t, t+h] overlaps the test period, per Section 4. This
module only constructs the raw target column; it does not do any
train/test splitting itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 1


def add_forward_log_return_target(
    df: pd.DataFrame,
    price_col: str = "mcx_close",
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Add a `target` column: forward log return over `horizon` days.

    The last `horizon` rows will have `target = NaN` (no future close exists
    yet to compute against) — this is expected, not a bug.
    """
    out = df.copy()
    close = out[price_col]
    out["target"] = np.log(close.shift(-horizon) / close)
    out.attrs["target_horizon"] = horizon
    return out
