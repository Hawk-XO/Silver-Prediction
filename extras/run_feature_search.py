"""
run_feature_search.py

CLI entry point for the feature-subset search/validate/holdout harness
(research/search_harness.py + research/feature_groups.py).

What this does
---------------
1. Loads real data + builds the full feature matrix once (data.pipeline_common,
   same loader run_real_data_pipeline.py uses -- can't drift apart).
2. Splits model_ready chronologically into three NON-OVERLAPPING windows:
     search      -- earliest portion. Try candidates freely here.
     validation  -- middle portion. Only search survivors get re-scored here.
     holdout     -- most recent portion. Exactly ONE candidate, ONCE, ever
                     (enforced by research/search_harness.py's JSON ledger).
3. Runs search -> prints ranked table.
4. Runs validation on the top-N search survivors -> prints ranked table with
   a `survived` column.
5. If exactly one candidate survived validation, offers to spend the
   holdout evaluation on it (requires --run-holdout; won't happen silently).
   If more than one survived, stops and asks you to pick, rather than
   picking the best-on-validation-Sharpe one for you -- that selection
   step is itself a place multiple-comparisons bias can creep back in if
   automated silently.

Each window needs enough rows to clear WalkForwardConfig.min_train_size
before it can produce even one prediction -- if a window is too small,
that candidate's result for that window comes back with n_predictions=0
and trusted=False rather than a misleadingly-precise number.

Run with:
    python run_feature_search.py
    python run_feature_search.py --top-n-to-validate 5 --run-holdout
    python run_feature_search.py --search-frac 0.5 --validation-frac 0.3
"""

from __future__ import annotations

import argparse

import pandas as pd

from data.pipeline_common import load_real_features
from backtest.walk_forward import WalkForwardConfig
from backtest.vectorbt_backtest import BacktestConfig
from research.feature_groups import default_search_candidates
from research.search_harness import run_search, run_validation, run_holdout_once, DEFAULT_LEDGER_PATH

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def chronological_three_way_split(
    model_ready: pd.DataFrame, search_frac: float, validation_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split model_ready by ROW POSITION (already date-sorted, since it comes
    straight out of a time-indexed pipeline) into three chronological,
    non-overlapping chunks. holdout_frac is implicitly 1 - search_frac -
    validation_frac.

    Deliberately NOT a random split -- shuffling would leak future
    information into "earlier" windows and defeat the entire purpose of
    walk-forward validation in the first place.
    """
    n = len(model_ready)
    search_end = int(n * search_frac)
    validation_end = int(n * (search_frac + validation_frac))

    search_window = model_ready.iloc[:search_end]
    validation_window = model_ready.iloc[search_end:validation_end]
    holdout_window = model_ready.iloc[validation_end:]

    return search_window, validation_window, holdout_window


def main(
    search_frac: float = 0.5,
    validation_frac: float = 0.3,
    top_n_to_validate: int = 6,
    run_holdout: bool = False,
    force_holdout: bool = False,
    ledger_path: str = DEFAULT_LEDGER_PATH,
    min_train_size: int = 120,
    candidate_names: list[str] | None = None,
):
    loaded = load_real_features()
    model_ready = loaded.model_ready
    feature_cols = loaded.feature_cols

    search_window, validation_window, holdout_window = chronological_three_way_split(
        model_ready, search_frac, validation_frac
    )
    print(f"\nChronological split: search={len(search_window)} rows "
          f"({search_window.index.min().date() if len(search_window) else '-'} to "
          f"{search_window.index.max().date() if len(search_window) else '-'}), "
          f"validation={len(validation_window)} rows "
          f"({validation_window.index.min().date() if len(validation_window) else '-'} to "
          f"{validation_window.index.max().date() if len(validation_window) else '-'}), "
          f"holdout={len(holdout_window)} rows "
          f"({holdout_window.index.min().date() if len(holdout_window) else '-'} to "
          f"{holdout_window.index.max().date() if len(holdout_window) else '-'}).")

    wf_config = WalkForwardConfig(
        horizon=1, min_train_size=min_train_size,
        arima_exog_cols=["ret_mean_5", "comex_mcx_spread_z_10"],
        xgb_params={"n_estimators": 80, "max_depth": 3},
    )
    bt_config = BacktestConfig(fees=0.0003, slippage=0.0005, margin_pct=0.15, init_cash=1_000_000)

    candidates = default_search_candidates(feature_cols)
    if candidate_names:
        missing = [n for n in candidate_names if n not in candidates]
        if missing:
            raise ValueError(f"--candidates named {missing} which don't exist. "
                              f"Available: {list(candidates.keys())}")
        candidates = {n: candidates[n] for n in candidate_names}
        if "all_features" not in candidates:
            candidates["all_features"] = default_search_candidates(feature_cols)["all_features"]
    print(f"\n{len(candidates)} candidates: {list(candidates.keys())}")
    n_folds_est = max(0, len(search_window) - min_train_size)
    print(f"Estimated ~{n_folds_est} walk-forward folds per candidate on the search window "
          f"({n_folds_est * len(candidates)} total fits ahead -- this is the slow part, "
          f"progress prints per-candidate as it goes).")

    print("\n=== STAGE 1: SEARCH (free exploration, search window) ===")
    search_results = run_search(search_window, candidates, wf_config, bt_config)
    print(search_results[["n_features", "n_predictions", "trusted", "strategy_sharpe",
                           "strategy_total_return_pct", "strategy_win_rate_pct", "note"]].to_string())

    top_names = search_results.head(top_n_to_validate).index.tolist()
    print(f"\nCarrying top {len(top_names)} search survivors into validation: {top_names}")
    validate_candidates = {name: candidates[name] for name in top_names}
    if "all_features" not in validate_candidates and "all_features" in candidates:
        validate_candidates["all_features"] = candidates["all_features"]  # always include baseline for comparison

    print("\n=== STAGE 2: VALIDATION (separate, later, non-overlapping window) ===")
    validation_results = run_validation(
        validation_window, validate_candidates, baseline_name="all_features",
        wf_config=wf_config, bt_config=bt_config,
    )
    print(validation_results[["n_features", "n_predictions", "trusted", "strategy_sharpe",
                               "strategy_total_return_pct", "survived", "note"]].to_string())

    survivors = validation_results[validation_results["survived"]]
    survivor_names = [n for n in survivors.index if n != "all_features" or "all_features" in top_names]
    print(f"\n{len(survivors)} candidate(s) survived validation: {list(survivors.index)}")

    if len(survivors) == 0:
        print("\nNo candidate beat the guardrails on the validation window -- "
              "none of these feature-group ablations show a real, replicating "
              "edge over the baseline. That's a legitimate result, not a bug: "
              "it means the weak-signal problem (139/150 BUY earlier) isn't "
              "fixed by dropping/isolating these particular feature groups. "
              "Stopping here -- holdout is not spent.")
        return search_results, validation_results, None

    if len(survivors) > 1:
        print("\nMore than one candidate survived validation. Not auto-picking "
              "one for holdout -- that selection step is itself a place "
              "multiple-comparisons bias can creep back in. Review the table "
              "above and re-run with a narrower `candidates` dict (edit this "
              "script or call research.search_harness.run_holdout_once "
              "directly) once you've chosen.")
        return search_results, validation_results, None

    winner_name = survivors.index[0]
    winner_cols = candidates[winner_name]
    print(f"\nSingle validation survivor: '{winner_name}' ({len(winner_cols)} features).")

    if not run_holdout:
        print("Not spending holdout evaluation (pass --run-holdout to do so once "
              "you're confident this is the candidate you want to test).")
        return search_results, validation_results, None

    print(f"\n=== STAGE 3: HOLDOUT (exactly once, ledger-guarded at {ledger_path}) ===")
    holdout_result = run_holdout_once(
        holdout_window, winner_name, winner_cols, wf_config, bt_config,
        ledger_path=ledger_path, force=force_holdout,
    )
    print(f"\nHoldout result for '{winner_name}':")
    for k, v in holdout_result.__dict__.items():
        print(f"  {k}: {v}")

    return search_results, validation_results, holdout_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-frac", type=float, default=0.5,
                         help="Fraction of model_ready (chronologically earliest) used for the search window. Default 0.5.")
    parser.add_argument("--validation-frac", type=float, default=0.3,
                         help="Fraction of model_ready used for the validation window (immediately after search). "
                              "Default 0.3. Remaining fraction (1 - search_frac - validation_frac) is holdout.")
    parser.add_argument("--top-n-to-validate", type=int, default=6,
                         help="How many top search-window candidates to carry into validation. Default 6.")
    parser.add_argument("--run-holdout", action="store_true",
                         help="If exactly one candidate survives validation, spend the one-time holdout "
                              "evaluation on it. Without this flag, the script stops after validation.")
    parser.add_argument("--force-holdout", action="store_true",
                         help="Override the holdout ledger's one-candidate-ever guard. Logged when used. "
                              "Only pass this if you deliberately want to break holdout discipline.")
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH,
                         help=f"Path to the holdout ledger JSON file. Default {DEFAULT_LEDGER_PATH}.")
    parser.add_argument("--min-train-size", type=int, default=120,
                         help="WalkForwardConfig.min_train_size. Default 120 (matches production). "
                              "Lower this (e.g. 60) for a much faster smoke-test run on real data -- "
                              "fold count is roughly (window_rows - min_train_size), so this directly "
                              "controls runtime. Don't trust results from a lowered value for real "
                              "decisions, only for checking the harness runs correctly.")
    parser.add_argument("--candidates", nargs="+", default=None,
                         help="Space-separated subset of candidate names to run instead of all 19 "
                              "(e.g. --candidates drop_ema only_return_stats). 'all_features' is "
                              "always included automatically as the baseline. Use this for a fast "
                              "first pass before committing to the full ~75min run.")
    args = parser.parse_args()

    main(
        search_frac=args.search_frac,
        validation_frac=args.validation_frac,
        top_n_to_validate=args.top_n_to_validate,
        run_holdout=args.run_holdout,
        force_holdout=args.force_holdout,
        ledger_path=args.ledger_path,
        min_train_size=args.min_train_size,
        candidate_names=args.candidates,
    )
