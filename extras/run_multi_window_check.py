"""
run_multi_window_check.py

Robustness check for candidates that survived (or nearly survived) the
search/validate split in run_feature_search.py. Splits the search+validation
history into several separate, non-overlapping chronological chunks and
re-scores each candidate on every chunk independently.

Why this exists: the two-window search/validate split can still be fooled.
A candidate can, by chance, do well on both the search window AND the
validation window and still not represent a real, regime-general edge --
it just needs to have gotten lucky twice instead of once. Slicing into
more (smaller) windows makes that much less likely to happen by chance,
and directly shows whether a candidate's edge is CONSISTENT (positive in
most windows, small spread) or was concentrated in one lucky regime.

IMPORTANT: this script only ever touches the search+validation portion of
model_ready (same fractions as run_feature_search.py's default 0.5/0.3
split) -- it reconstructs and reuses that split's boundary rather than
inventing a new one, and NEVER includes the holdout window. This isn't a
backdoor around the holdout guard; it's meant to run BEFORE you decide
whether a candidate is even worth spending holdout on.

Run with:
    python run_multi_window_check.py
    python run_multi_window_check.py --n-windows 6
    python run_multi_window_check.py --candidates only_rsi all_features
"""

from __future__ import annotations

import argparse

import pandas as pd

from data.pipeline_common import load_real_features
from backtest.walk_forward import WalkForwardConfig
from backtest.vectorbt_backtest import BacktestConfig
from research.feature_groups import default_search_candidates
from research.search_harness import run_multi_window_check
from run_feature_search import chronological_three_way_split

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# Candidates worth re-checking based on the search/validate run: the 6
# search-stage survivors that got carried into validation, plus the
# baseline. Override with --candidates if you want a different set (e.g.
# after re-running search/validate with different fractions).
DEFAULT_CANDIDATE_NAMES = [
    "all_features", "only_rsi", "only_ema", "only_adx",
    "only_bollinger", "drop_return_stats", "only_atr",
]


def main(
    n_windows: int = 4,
    candidate_names: list[str] | None = None,
    search_frac: float = 0.5,
    validation_frac: float = 0.3,
    min_train_size: int = 60,
):
    loaded = load_real_features()
    model_ready = loaded.model_ready
    feature_cols = loaded.feature_cols

    # Reuse the exact same split boundary run_feature_search.py used, then
    # discard the holdout piece -- this script must never see it.
    search_window, validation_window, holdout_window = chronological_three_way_split(
        model_ready, search_frac, validation_frac
    )
    non_holdout = pd.concat([search_window, validation_window])
    print(f"Non-holdout history: {len(non_holdout)} rows "
          f"({non_holdout.index.min().date()} to {non_holdout.index.max().date()}). "
          f"Holdout ({len(holdout_window)} rows, {holdout_window.index.min().date() if len(holdout_window) else '-'} "
          f"to {holdout_window.index.max().date() if len(holdout_window) else '-'}) is excluded and untouched.")

    all_candidates = default_search_candidates(feature_cols)
    names = candidate_names or [n for n in DEFAULT_CANDIDATE_NAMES if n in all_candidates]
    missing = [n for n in names if n not in all_candidates]
    if missing:
        raise ValueError(f"--candidates named {missing} which don't exist. "
                          f"Available: {list(all_candidates.keys())}")
    candidates = {n: all_candidates[n] for n in names}

    wf_config = WalkForwardConfig(
        horizon=1, min_train_size=min_train_size,
        arima_exog_cols=["ret_mean_5", "comex_mcx_spread_z_10"],
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)

    edges = [round(i * len(non_holdout) / n_windows) for i in range(n_windows + 1)]
    print(f"\nSplitting into {n_windows} chronological windows:")
    for i in range(n_windows):
        chunk = non_holdout.iloc[edges[i]:edges[i + 1]]
        print(f"  window_{i}: {len(chunk)} rows, {chunk.index.min().date()} to {chunk.index.max().date()}")

    print(f"\n{len(candidates)} candidates: {list(candidates.keys())}\n")

    results = run_multi_window_check(non_holdout, candidates, n_windows, wf_config, bt_config)

    print("\n=== MULTI-WINDOW ROBUSTNESS CHECK (strategy_sharpe per window) ===")
    print(results.to_string())

    print("\nReading this table:")
    print("  - mean_sharpe: average edge across all windows (higher = better, but see std)")
    print("  - std_sharpe: how much the edge varies window to window (lower = more consistent)")
    print("  - min_sharpe: worst-case window (a candidate that's only good 'on average' but")
    print("    occasionally very negative is a very different risk profile than one that's")
    print("    consistently mediocre)")
    print("  - frac_windows_positive: fraction of windows with positive Sharpe. A real, ")
    print("    regime-general edge should clear >0.5 here, ideally higher.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-windows", type=int, default=4,
                         help="Number of non-overlapping chronological chunks to split the "
                              "search+validation history into. Default 4.")
    parser.add_argument("--candidates", nargs="+", default=None,
                         help="Space-separated candidate names to check. Defaults to the "
                              "search-stage survivors from the last run_feature_search.py pass "
                              f"plus baseline: {DEFAULT_CANDIDATE_NAMES}")
    parser.add_argument("--search-frac", type=float, default=0.5,
                         help="Must match the --search-frac used in run_feature_search.py so the "
                              "holdout boundary lines up. Default 0.5.")
    parser.add_argument("--validation-frac", type=float, default=0.3,
                         help="Must match the --validation-frac used in run_feature_search.py. Default 0.3.")
    parser.add_argument("--min-train-size", type=int, default=60,
                         help="WalkForwardConfig.min_train_size. Default 60 (lower than "
                              "run_feature_search.py's 120 default since each window here is "
                              "smaller -- with 4 windows over ~2150 rows that's ~540 rows/window, "
                              "and min_train_size=120 would eat a large fraction of that).")
    args = parser.parse_args()

    main(
        n_windows=args.n_windows,
        candidate_names=args.candidates,
        search_frac=args.search_frac,
        validation_frac=args.validation_frac,
        min_train_size=args.min_train_size,
    )
