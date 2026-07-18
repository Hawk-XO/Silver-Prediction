"""
data/pipeline_common.py

Shared "load real data -> build feature matrix" logic used by both
run_real_data_pipeline.py and research/run_feature_search.py. Factored out
so the two scripts can't silently drift apart (e.g. one script fixing a
timezone bug that the other doesn't get).

This module owns exactly the Phase 1-3 portion of the pipeline: raw MySQL
+ live-data loading, continuous-contract building, merging, and feature
matrix construction. It does NOT own walk-forward/signals/backtest -- those
stay specific to each caller (run_real_data_pipeline.py runs the full
production flow once; the search harness runs walk-forward repeatedly per
feature-subset candidate on top of the same loaded features).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.db import load_ohlcv
from data.contract_roll import build_continuous_series
from data.merge import merge_mcx_with_global
from data.global_factors import fetch_all_global_factors, fetch_comex_gold
from features.pipeline import build_feature_matrix, get_feature_columns

# Same buffer as run_real_data_pipeline.py used inline -- see that script's
# original comment (preserved below) for why this can't just be applied as
# a raw --start-date filter.
LOOKBACK_BUFFER_DAYS = 500  # ~320 trading days of warmup+min_train, plus slack for weekends/holidays


def _strip_tz(df_or_series):
    """MySQL-sourced data (via data/db.py) comes back tz-naive since MySQL's
    DATE column has no timezone concept. Live data from data/global_factors.py
    comes back tz-aware (Asia/Kolkata), since it's fetched fresh via yfinance.
    Pandas refuses to .join() a tz-naive index with a tz-aware one, so we
    normalize everything to tz-naive right before merging -- dates already
    represent MCX trading days, the tz info isn't adding real information
    at this point anyway."""
    if df_or_series.index.tz is not None:
        df_or_series = df_or_series.copy()
        df_or_series.index = df_or_series.index.tz_localize(None)
    return df_or_series


@dataclass
class LoadedFeatures:
    """Everything downstream code (walk-forward, feature search) needs
    after Phase 1-3. `model_ready` is already NaN-dropped on feature_cols
    (warmup rows removed) -- it's the direct input to run_walk_forward()."""
    features_df: pd.DataFrame     # full feature matrix, including warmup NaNs
    model_ready: pd.DataFrame     # NaN-dropped on feature_cols, ready for walk-forward
    feature_cols: list[str]
    raw_row_count: int
    source_counts: dict


def load_real_features(
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int = 1,
    verbose: bool = True,
    source: str | None = None,
    commodity: str | None = None,
) -> LoadedFeatures:
    """
    Load real MySQL + live-data history and build the full feature matrix,
    identical to Phase 1-3 of run_real_data_pipeline.main(). Kept separate
    from walk-forward so callers that need to run walk-forward multiple
    times (the feature-search harness) only pay the load+feature-engineering
    cost once per date-range, not once per candidate feature subset.

    source: filter to one provenance -- 'proxy' | 'kite_api' | 'manual_csv'.
        None (default) loads everything and lets build_continuous_series()
        do its normal blending (real kite_api/manual_csv rows preferred,
        proxy fills gaps) -- this is what the UI's "calibrated" data-source
        option maps to, since it's the best-available blended dataset
        rather than a literal separate calibration pass. See
        data/mcx_proxy.py's calibrate_premium() if you actually want to
        adjust the proxy's premium assumption instead of just filtering
        which rows get used.
    commodity: filter to one commodity's contracts by prefix match against
        the part of `contract` before the first underscore (e.g. 'SILVERMIC'
        matches 'SILVERMIC_26FEB2027' but NOT 'SILVER_...' or 'SILVERM_...' --
        plain substring matching would incorrectly match 'SILVER' against
        'SILVERM...' and 'SILVERMIC...' contracts too). None (default) keeps
        all commodities/contracts as loaded.

    IMPORTANT: --start-date must NOT be applied to the raw load directly.
    Feature warmup (~200 rows) + walk-forward min_train_size (120 rows)
    need real history strictly BEFORE start_date to produce any predictions
    at all inside the requested window -- filtering the raw load by
    start_date destroys exactly that lookback. Instead we load from a
    buffered earlier date and let callers slice `model_ready` down to their
    actual window of interest afterwards (this function intentionally
    returns the wider buffered range -- see module docstring).
    """
    if verbose:
        print("Loading real data from MySQL (proxy + kite_api rows)...")
    multi_contract = load_ohlcv(source=source)
    if multi_contract.empty:
        raise RuntimeError(
            "mcx_silver_ohlcv is empty (or nothing matches source="
            f"{source!r}) — run data/build_merged_history.py and/or "
            "data/kite_fetcher.py first."
        )

    if commodity:
        prefix = multi_contract["contract"].astype(str).str.split("_").str[0]
        multi_contract = multi_contract[prefix == commodity]
        if multi_contract.empty:
            raise RuntimeError(f"No rows for commodity={commodity!r} (checked exact prefix "
                                f"before '_' in the contract column, not substring match).")

    load_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
        if start_date else None
    )
    if load_start is not None:
        multi_contract = multi_contract[multi_contract.index >= load_start]
        if multi_contract.empty:
            raise RuntimeError(f"No rows on/after {load_start.date()} (buffered from --start-date {start_date}).")
    if end_date:
        multi_contract = multi_contract[multi_contract.index <= pd.Timestamp(end_date)]
        if multi_contract.empty:
            raise RuntimeError(f"No rows on/before --end-date {end_date}.")

    if verbose:
        print(f"  {len(multi_contract)} raw rows across "
              f"{multi_contract['contract'].nunique()} contract(s)/segments, "
              f"{multi_contract.index.min().date()} to {multi_contract.index.max().date()}.")
        print(f"  By source: {multi_contract['source'].value_counts().to_dict()}")

    raw_row_count = len(multi_contract)
    source_counts = multi_contract["source"].value_counts().to_dict()

    continuous = build_continuous_series(multi_contract)

    range_start = continuous.index.min().strftime("%Y-%m-%d")
    range_end = continuous.index.max().strftime("%Y-%m-%d")
    if verbose:
        print(f"Fetching live global factors ({range_start} to {range_end})...")
    globals_ = fetch_all_global_factors(start=range_start, end=range_end)
    gold_close = fetch_comex_gold(start=range_start, end=range_end)["close"]

    continuous = _strip_tz(continuous)
    comex = _strip_tz(globals_["comex_silver"])
    usdinr = _strip_tz(globals_["usdinr"])
    dxy = _strip_tz(globals_["dxy"])
    gold_close = _strip_tz(gold_close)

    merged = merge_mcx_with_global(continuous, comex, usdinr, dxy)

    if verbose:
        print("Building feature matrix...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_df = build_feature_matrix(merged, gold_close=gold_close, horizon=horizon)

    feature_cols = get_feature_columns(features_df)
    model_ready = features_df.dropna(subset=feature_cols).copy()
    if verbose:
        print(f"  {len(model_ready)} rows with complete features "
              f"(dropped {len(features_df) - len(model_ready)} warmup/NaN rows).")

    return LoadedFeatures(
        features_df=features_df,
        model_ready=model_ready,
        feature_cols=feature_cols,
        raw_row_count=raw_row_count,
        source_counts=source_counts,
    )
