"""
features/pipeline.py

Orchestrates the full Phase 3 feature-engineering pipeline on top of the
Phase 2 merged MCX/global-factors DataFrame (data.merge.merge_mcx_with_global
output).

Usage
-----
    from data.merge import merge_mcx_with_global
    from data.global_factors import fetch_comex_gold
    from features.pipeline import build_feature_matrix

    merged = merge_mcx_with_global(mcx_continuous, comex, usdinr, dxy)
    gold = fetch_comex_gold(start=..., end=...)   # optional
    features_df = build_feature_matrix(merged, gold_close=gold["close"])

The Gold-Silver ratio feature is skipped (with a warning) if `gold_close`
is not supplied, since it's the one feature here that needs data outside
the Phase 2 merged frame.
"""

from __future__ import annotations

import warnings

import pandas as pd

from features.indicators import add_all_indicators
from features.rolling_stats import add_return_rolling_stats
from features.cross_asset import add_gold_silver_ratio, add_comex_mcx_spread_zscore
from features.calendar_features import add_calendar_features
from features.target import add_forward_log_return_target, DEFAULT_HORIZON

# Columns coming straight from the Phase 2 merged frame (price LEVELS and
# raw staleness flags) — excluded from `get_feature_columns()` because
# PROJECT_NOTES.md Section 2 says not to fit models on non-stationary price
# levels directly. Keep them in the DataFrame (useful for debugging/plots),
# just don't hand them to a model as-is.
RAW_INPUT_COLS = {
    "mcx_open", "mcx_high", "mcx_low", "mcx_close", "mcx_volume", "mcx_oi",
    "comex_close", "usdinr_close", "dxy_close",
    "comex_stale", "usdinr_stale", "dxy_stale",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of engineered feature columns in `df` — i.e. everything
    build_feature_matrix() added, excluding raw price-level inputs and the
    `target` label. This is the column list Phase 4/5 (models, walk-forward)
    should train on.
    """
    return [c for c in df.columns if c not in RAW_INPUT_COLS and c != "target"]


def build_feature_matrix(
    merged_df: pd.DataFrame,
    gold_close: pd.Series | None = None,
    horizon: int = DEFAULT_HORIZON,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature matrix from a Phase 2 merged DataFrame.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of data.merge.merge_mcx_with_global — must contain
        mcx_open/high/low/close, comex_close, usdinr_close.
    gold_close : pd.Series | None
        COMEX gold close series (e.g. fetch_comex_gold(...)["close"]). If
        None, the gold_silver_ratio feature is skipped.
    horizon : int
        Forward-return horizon for the target column (default 1, per
        PROJECT_NOTES.md).
    include_target : bool
        If False, skip adding the target column (useful for pure
        inference/prediction on the latest row, where no future close
        exists to label against).

    Returns
    -------
    pd.DataFrame
        `merged_df` plus all engineered feature columns (and `target` if
        `include_target=True`). Rows at the start of the series will have
        NaNs from warm-up windows (e.g. ema_200 needs 200 prior bars); rows
        at the end will have NaN `target` (see features/target.py). Callers
        should drop NaNs appropriately for their use case rather than this
        function silently doing it, since "appropriate" differs between
        training (drop both ends) and live inference (only drop the
        feature warm-up NaNs, keep the target-less latest row).
    """
    out = merged_df.copy()

    out = add_all_indicators(out, high_col="mcx_high", low_col="mcx_low", close_col="mcx_close")
    out = add_return_rolling_stats(out, price_col="mcx_close")
    out = add_comex_mcx_spread_zscore(out)

    if gold_close is not None:
        out = add_gold_silver_ratio(out, gold_close=gold_close)
    else:
        warnings.warn(
            "build_feature_matrix: gold_close not provided — skipping "
            "gold_silver_ratio feature. Pass gold_close= to include it.",
            stacklevel=2,
        )

    out = add_calendar_features(out)

    if include_target:
        out = add_forward_log_return_target(out, price_col="mcx_close", horizon=horizon)

    return out
