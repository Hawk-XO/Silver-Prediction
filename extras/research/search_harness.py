"""
research/search_harness.py

Core search-validate-holdout logic for comparing feature-subset candidates
(research/feature_groups.py output) against real MCX Silver data, without
falling into the "try everything, eyeball the winner" trap that produces
confident-looking illusions on a signal we already know is weak (139/150
BUY signals on the unfiltered real-data run -- see PROJECT_NOTES.md /
prior session notes).

Three-stage discipline
-----------------------
1. SEARCH  -- score every candidate freely on an early chronological
   window. No guardrails here; this is where you're allowed to try lots
   of things.
2. VALIDATE -- re-score only the SEARCH survivors on a separate, later,
   non-overlapping window. A combo that doesn't replicate here is not
   real, no matter how good it looked in search.
3. HOLDOUT -- your single remaining candidate gets scored EXACTLY ONCE
   against the most recent window, which nothing above this point was
   ever allowed to see. Enforced by a JSON ledger (run_holdout_once) that
   refuses a second evaluation of a *different* candidate unless the
   caller explicitly passes force=True -- silently re-running holdout
   until something looks good defeats the entire point of having one.

Each candidate's confidence_threshold is set deterministically (median of
that candidate's own confidence distribution on the window being scored)
rather than hand-tuned per combo, so comparisons across candidates are
fair and this process doesn't introduce yet another layer of overfitting.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from backtest.metrics import evaluate_predictions
from signals.signal_engine import SignalConfig, generate_signals, compute_confidence
from backtest.vectorbt_backtest import BacktestConfig, compare_to_buy_and_hold

DEFAULT_LEDGER_PATH = "research/holdout_ledger.json"

# Minimum walk-forward predictions a candidate needs on a window before its
# metrics are trusted at all -- below this, Sharpe/hit-rate are noise, not
# signal, and we say so rather than quietly returning a number.
MIN_PREDICTIONS_FOR_TRUST = 30


@dataclass
class CandidateResult:
    name: str
    n_features: int
    n_predictions: int
    directional_accuracy: float
    rmse: float
    prediction_sharpe: float          # naive sign(pred)*actual sharpe, from backtest.metrics
    strategy_sharpe: float            # from full signal-engine -> vectorbt path
    strategy_total_return_pct: float
    strategy_win_rate_pct: float
    strategy_profit_factor: float
    buy_and_hold_total_return_pct: float
    confidence_threshold_used: float
    signal_counts: dict
    trusted: bool                     # False if n_predictions < MIN_PREDICTIONS_FOR_TRUST
    note: str = ""


def _feature_subset_hash(feature_cols: list[str]) -> str:
    """Stable hash of a feature subset's contents (order-independent) --
    used by the holdout ledger so renaming a candidate can't accidentally
    dodge the one-time-use guard, and so genuinely re-deriving the same
    subset under a different name is still caught."""
    key = ",".join(sorted(feature_cols))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run_candidate_on_window(
    model_ready_window: pd.DataFrame,
    feature_cols: list[str],
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    name: str = "candidate",
) -> CandidateResult:
    """
    Run walk-forward + signal generation + vectorbt-vs-buy-and-hold for one
    feature subset on one (already-sliced) window of model_ready rows.

    Confidence threshold is NOT a parameter here on purpose -- it's derived
    from the median of this candidate's own confidence distribution on this
    window (see module docstring), so no caller can accidentally hand-tune
    per candidate and bias the comparison.
    """
    wf_cfg = wf_config or WalkForwardConfig()
    bt_cfg = bt_config or BacktestConfig()

    import time as _time
    _t0 = _time.time()
    _n_folds_est = max(0, len(model_ready_window) - wf_cfg.min_train_size)
    print(f"  [{name}] {len(feature_cols)} features, ~{_n_folds_est} walk-forward "
          f"folds on {len(model_ready_window)} rows -- starting...", flush=True)

    # ARIMA's exog_cols must be a subset of THIS candidate's own feature_cols
    # -- a group-ablation candidate (e.g. "drop_cross_asset") can legitimately
    # remove a column the caller's wf_config still names as an exog regressor.
    # Silently dropping the missing ones (rather than raising) is correct
    # here: it's exactly what "this group is gone" should mean for ARIMA too.
    usable_exog_cols = [c for c in wf_cfg.arima_exog_cols if c in feature_cols]
    if usable_exog_cols != wf_cfg.arima_exog_cols:
        from dataclasses import replace
        wf_cfg = replace(wf_cfg, arima_exog_cols=usable_exog_cols)

    if len(model_ready_window) <= wf_cfg.min_train_size:
        return CandidateResult(
            name=name, n_features=len(feature_cols), n_predictions=0,
            directional_accuracy=np.nan, rmse=np.nan, prediction_sharpe=np.nan,
            strategy_sharpe=np.nan, strategy_total_return_pct=np.nan,
            strategy_win_rate_pct=np.nan, strategy_profit_factor=np.nan,
            buy_and_hold_total_return_pct=np.nan, confidence_threshold_used=np.nan,
            signal_counts={}, trusted=False,
            note=f"window has {len(model_ready_window)} rows, <= min_train_size="
                 f"{wf_cfg.min_train_size} -- cannot produce a single walk-forward prediction",
        )

    wf_results = run_walk_forward(
        model_ready_window, feature_cols=feature_cols, target_col="target", config=wf_cfg
    )
    print(f"  [{name}] walk-forward done in {_time.time() - _t0:.1f}s -- "
          f"{len(wf_results)} predictions.", flush=True)

    if wf_results.empty:
        return CandidateResult(
            name=name, n_features=len(feature_cols), n_predictions=0,
            directional_accuracy=np.nan, rmse=np.nan, prediction_sharpe=np.nan,
            strategy_sharpe=np.nan, strategy_total_return_pct=np.nan,
            strategy_win_rate_pct=np.nan, strategy_profit_factor=np.nan,
            buy_and_hold_total_return_pct=np.nan, confidence_threshold_used=np.nan,
            signal_counts={}, trusted=False,
            note="walk-forward produced zero predictions on this window",
        )

    pred_metrics = evaluate_predictions(wf_results["y_true"], wf_results["meta_pred"])

    # --- Deterministic confidence threshold: median of THIS candidate's
    # own confidence distribution on THIS window ---
    signal_input = wf_results[["meta_pred"]].join(
        model_ready_window[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    raw_confidence = compute_confidence(
        signal_input, pred_col="meta_pred", volatility_col="ret_std_20",
    )
    median_confidence = float(raw_confidence.median()) if raw_confidence.notna().any() else 0.5
    # Guard against a degenerate all-NaN or all-zero confidence distribution
    # collapsing the threshold to 0 (which would make every row "confident").
    if not np.isfinite(median_confidence) or median_confidence <= 0:
        median_confidence = 0.5

    signal_config = SignalConfig(confidence_threshold=median_confidence, cooldown_days=3)
    signals_df = generate_signals(signal_input, signal_config)
    signal_counts = signals_df["signal"].value_counts().to_dict()

    n_trades = int((signals_df["signal"] != "HOLD").sum())
    if n_trades == 0:
        return CandidateResult(
            name=name, n_features=len(feature_cols), n_predictions=len(wf_results),
            directional_accuracy=pred_metrics["directional_accuracy"], rmse=pred_metrics["rmse"],
            prediction_sharpe=pred_metrics["sharpe"], strategy_sharpe=np.nan,
            strategy_total_return_pct=np.nan, strategy_win_rate_pct=np.nan,
            strategy_profit_factor=np.nan, buy_and_hold_total_return_pct=np.nan,
            confidence_threshold_used=median_confidence, signal_counts=signal_counts,
            trusted=len(wf_results) >= MIN_PREDICTIONS_FOR_TRUST,
            note="every row HOLD at this candidate's own median-confidence threshold "
                 "-- no trades to backtest",
        )

    comparison = compare_to_buy_and_hold(signals_df, price_col="entry_price", config=bt_cfg)

    trusted = len(wf_results) >= MIN_PREDICTIONS_FOR_TRUST
    print(f"  [{name}] done in {_time.time() - _t0:.1f}s -- "
          f"{len(wf_results)} predictions, {n_trades} trades.", flush=True)
    return CandidateResult(
        name=name,
        n_features=len(feature_cols),
        n_predictions=len(wf_results),
        directional_accuracy=pred_metrics["directional_accuracy"],
        rmse=pred_metrics["rmse"],
        prediction_sharpe=pred_metrics["sharpe"],
        strategy_sharpe=float(comparison.loc["strategy", "sharpe_ratio"]),
        strategy_total_return_pct=float(comparison.loc["strategy", "total_return_pct"]),
        strategy_win_rate_pct=float(comparison.loc["strategy", "win_rate_pct"]),
        strategy_profit_factor=float(comparison.loc["strategy", "profit_factor"]),
        buy_and_hold_total_return_pct=float(comparison.loc["buy_and_hold", "total_return_pct"]),
        confidence_threshold_used=median_confidence,
        signal_counts=signal_counts,
        trusted=trusted,
        note="" if trusted else f"only {len(wf_results)} predictions (< {MIN_PREDICTIONS_FOR_TRUST}) -- treat as noise",
    )


def _results_to_frame(results: list[CandidateResult]) -> pd.DataFrame:
    rows = {r.name: {
        "n_features": r.n_features,
        "n_predictions": r.n_predictions,
        "trusted": r.trusted,
        "directional_accuracy": r.directional_accuracy,
        "prediction_sharpe": r.prediction_sharpe,
        "strategy_sharpe": r.strategy_sharpe,
        "strategy_total_return_pct": r.strategy_total_return_pct,
        "strategy_win_rate_pct": r.strategy_win_rate_pct,
        "strategy_profit_factor": r.strategy_profit_factor,
        "buy_and_hold_total_return_pct": r.buy_and_hold_total_return_pct,
        "confidence_threshold_used": r.confidence_threshold_used,
        "note": r.note,
    } for r in results}
    df = pd.DataFrame(rows).T
    return df.sort_values("strategy_sharpe", ascending=False)


def run_search(
    model_ready_search_window: pd.DataFrame,
    candidates: dict[str, list[str]],
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Stage 1: score every candidate freely on the search window. Returns a
    DataFrame ranked by strategy_sharpe (descending), NOT filtered -- no
    guardrails at this stage, that's what run_validation is for.
    """
    results = [
        run_candidate_on_window(model_ready_search_window, cols, wf_config, bt_config, name=name)
        for name, cols in candidates.items()
    ]
    return _results_to_frame(results)


def run_validation(
    model_ready_validation_window: pd.DataFrame,
    candidates: dict[str, list[str]],
    baseline_name: str = "all_features",
    min_sharpe: float = 0.0,
    max_relative_drop_vs_baseline: float = 0.5,
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Stage 2: re-score `candidates` (normally: the search-stage survivors
    you chose to carry forward, e.g. top-N by search-window Sharpe) on a
    separate, later, non-overlapping window. Adds a `survived` column:

        survived = strategy_sharpe > min_sharpe
                   AND strategy_sharpe doesn't fall more than
                       max_relative_drop_vs_baseline below the baseline
                       candidate's validation-window Sharpe (if baseline
                       is present in `candidates` and itself survives)
                   AND trusted (>= MIN_PREDICTIONS_FOR_TRUST predictions)

    A combo that looked great in search but doesn't clear this bar here is
    exactly the false positive this whole harness exists to catch.
    """
    df = run_search(model_ready_validation_window, candidates, wf_config, bt_config)  # same scoring logic, different window

    survived = (df["strategy_sharpe"] > min_sharpe) & df["trusted"]

    if baseline_name in df.index and np.isfinite(df.loc[baseline_name, "strategy_sharpe"]):
        baseline_sharpe = df.loc[baseline_name, "strategy_sharpe"]
        if baseline_sharpe > 0:
            floor = baseline_sharpe * (1 - max_relative_drop_vs_baseline)
            survived = survived & (df["strategy_sharpe"] >= floor)
        # if baseline itself is <= 0, the relative-drop check is meaningless
        # (there's nothing positive to fall a fraction below), so we skip it
        # and rely on the absolute min_sharpe bar above.

    df["survived"] = survived
    return df.sort_values(["survived", "strategy_sharpe"], ascending=[False, False])


def run_holdout_once(
    model_ready_holdout_window: pd.DataFrame,
    candidate_name: str,
    feature_cols: list[str],
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    ledger_path: str = DEFAULT_LEDGER_PATH,
    force: bool = False,
) -> CandidateResult:
    """
    Stage 3: score exactly ONE candidate against the holdout window,
    exactly once, ever (per feature-subset content -- see
    _feature_subset_hash). Guarded by a JSON ledger at `ledger_path`.

    Calling this a second time with a DIFFERENT feature subset raises
    RuntimeError unless force=True is passed explicitly -- the whole point
    of a holdout set is that you don't get to keep trying until one
    candidate looks good on it. Re-calling with the SAME subset (identical
    hash) just returns the ledger's recorded result again, unchanged --
    that's not re-spending the holdout, it's just re-reading what you
    already learned.

    force=True still records the new attempt in the ledger (with a
    `forced: true` flag) rather than silently overwriting history, so a
    human reviewing the ledger later can see holdout discipline was broken
    and when.
    """
    subset_hash = _feature_subset_hash(feature_cols)
    ledger = _load_ledger(ledger_path)

    if ledger.get("entries"):
        prior = ledger["entries"][0]
        if prior["subset_hash"] == subset_hash:
            print(f"[holdout ledger] '{candidate_name}' already evaluated on holdout "
                  f"(hash {subset_hash}, recorded {prior['timestamp']}) -- returning "
                  f"the original result rather than re-running.")
            return _candidate_result_from_dict(prior["result"])
        if not force:
            raise RuntimeError(
                f"Holdout already spent on a DIFFERENT candidate: "
                f"'{prior['candidate_name']}' (hash {prior['subset_hash']}, "
                f"recorded {prior['timestamp']}). Evaluating '{candidate_name}' "
                f"(hash {subset_hash}) here would mean trying candidates against "
                f"holdout until one looks good -- exactly what this guard exists "
                f"to prevent. Pass force=True if you deliberately want to override "
                f"this (it will be logged as such)."
            )
        print(f"[holdout ledger] FORCE override: spending holdout again on "
              f"'{candidate_name}' despite prior entry for "
              f"'{prior['candidate_name']}'. This is logged.")

    result = run_candidate_on_window(
        model_ready_holdout_window, feature_cols, wf_config, bt_config, name=candidate_name
    )

    entry = {
        "candidate_name": candidate_name,
        "subset_hash": subset_hash,
        "feature_cols": sorted(feature_cols),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forced": force and bool(ledger.get("entries")),
        "result": result.__dict__,
    }
    ledger.setdefault("entries", []).insert(0, entry)
    _save_ledger(ledger_path, ledger)

    return result


def split_into_n_windows(df: pd.DataFrame, n_windows: int) -> list[pd.DataFrame]:
    """
    Split a chronologically-sorted DataFrame into `n_windows` roughly-equal,
    non-overlapping, chronological chunks. Used by run_multi_window_check
    to test whether a candidate's edge is consistent across market regimes
    or was a one-off fit to a single window's quirks.

    Deliberately row-count-based (not calendar-based) so each chunk gets a
    comparable number of walk-forward folds -- a calendar-equal split could
    hand one chunk 800 rows (dense trading) and another 200 (data gaps),
    making their Sharpe estimates incomparable in reliability.
    """
    n = len(df)
    edges = [round(i * n / n_windows) for i in range(n_windows + 1)]
    return [df.iloc[edges[i]:edges[i + 1]] for i in range(n_windows)]


def run_multi_window_check(
    non_holdout_df: pd.DataFrame,
    candidates: dict[str, list[str]],
    n_windows: int = 4,
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Robustness check: run each candidate on `n_windows` separate,
    non-overlapping chronological chunks of `non_holdout_df` (the caller
    is responsible for making sure this excludes the holdout window --
    see run_multi_window_check.py, which slices search+validation only).

    Returns a DataFrame indexed by candidate name with per-window
    strategy_sharpe columns (window_0, window_1, ...) plus summary columns:
        mean_sharpe, std_sharpe, min_sharpe, frac_windows_positive

    A candidate whose search/validation result was a real, regime-general
    edge should show positive Sharpe in most windows with a small std. A
    candidate that only won because of one lucky window will show high
    std and a low frac_windows_positive here -- exactly the pattern that
    distinguishes "found a genuine edge" from "found a coincidence,"
    which the search/validate split alone can miss if the coincidence
    happened to hold up across both of those windows too.
    """
    windows = split_into_n_windows(non_holdout_df, n_windows)
    rows = {}
    for name, cols in candidates.items():
        sharpes = []
        for i, window in enumerate(windows):
            result = run_candidate_on_window(window, cols, wf_config, bt_config, name=f"{name}[w{i}]")
            sharpes.append(result.strategy_sharpe)
        arr = np.array(sharpes, dtype=float)
        valid = arr[~np.isnan(arr)]
        row = {f"window_{i}": s for i, s in enumerate(sharpes)}
        row["mean_sharpe"] = float(valid.mean()) if len(valid) else np.nan
        row["std_sharpe"] = float(valid.std()) if len(valid) > 1 else np.nan
        row["min_sharpe"] = float(valid.min()) if len(valid) else np.nan
        row["frac_windows_positive"] = float((valid > 0).mean()) if len(valid) else np.nan
        rows[name] = row

    df = pd.DataFrame(rows).T
    return df.sort_values("mean_sharpe", ascending=False)


def _candidate_result_from_dict(d: dict) -> CandidateResult:
    return CandidateResult(**d)


def _load_ledger(path: str) -> dict:
    if not os.path.exists(path):
        return {"entries": []}
    with open(path) as f:
        return json.load(f)


def _save_ledger(path: str, ledger: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, default=str)


def reset_ledger(ledger_path: str = DEFAULT_LEDGER_PATH) -> None:
    """Explicit, deliberately-named escape hatch for starting a fresh
    holdout ledger (e.g. a genuinely new project phase with new holdout
    data). Never called implicitly by run_holdout_once."""
    _save_ledger(ledger_path, {"entries": []})
