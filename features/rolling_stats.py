"""
features/rolling_stats.py

Rolling statistics (mean, std, skew) of log-returns over multiple windows.

Follows the exact pattern mandated in PROJECT_NOTES.md Section 3:

    df['roll_mean_5'] = df['close'].shift(1).rolling(5).mean()   # RIGHT

i.e. `.shift(1)` is applied to the return series BEFORE the rolling window,
so the stat assigned to row t is computed entirely from returns realized
strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (5, 10, 20)


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def add_return_rolling_stats(
    df: pd.DataFrame,
    price_col: str = "mcx_close",
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling mean/std/skew of log-returns for each window in `windows`.

    Adds columns: ret_mean_{w}, ret_std_{w}, ret_skew_{w} for each w.

    Note: `close.shift(1)` inside `_log_returns` produces the return realized
    AT day t-1->t (i.e. it is "known" as of the close of day t). We then
    apply a further `.shift(1)` before rolling so that the stat for row t
    only pools returns realized up through day t-1, never the return that
    resolves on day t itself.
    """
    out = df.copy()
    returns = _log_returns(out[price_col])
    lagged_returns = returns.shift(1)

    for w in windows:
        out[f"ret_mean_{w}"] = lagged_returns.rolling(w).mean()
        out[f"ret_std_{w}"] = lagged_returns.rolling(w).std()
        out[f"ret_skew_{w}"] = lagged_returns.rolling(w).skew()

    return out
