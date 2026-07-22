"""
ui/pipeline_runner.py

Plain-Python backend for the Streamlit control UI (ui/app.py). Deliberately
has zero Streamlit imports -- every knob the UI exposes (contract, date
range, horizon, data source, model hyperparams, signal threshold/cooldown)
is a field on RunConfig, and run_pipeline() is a pure function of that
config. This means the whole pipeline can be smoke-tested from a plain
python/pytest shell without ever starting the Streamlit server, and the UI
layer itself stays thin (just widgets -> RunConfig -> run_pipeline -> render
RunResult).

Mirrors run_phase7_demo.py / run_real_data_pipeline.py's Phase 1-7 flow
exactly (same functions, same order) -- this is not a reimplementation,
it's those same pipeline calls wired up to configurable params instead of
hardcoded ones.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from data.pipeline_common import load_real_features
from data.kite_fetcher import fetch_missing_range
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import (
    BacktestConfig, run_strategy_backtest, run_buy_and_hold_backtest,
    performance_report, trade_pnl_breakdown, signals_to_position,
)
from broker.kite_paper_broker import PaperKiteBroker

# Commodities as they appear before the first underscore in the `contract`
# column (see data/db.py's schema docstring, e.g. 'SILVERMIC_26FEB2027').
AVAILABLE_CONTRACTS = ["SILVER", "SILVERM", "SILVERMIC"]

# The UI used to expose a data-source picker (proxy / calibrated / manual
# bhavcopy). We only ever run against the full blended dataset now, so
# that's hardcoded here -- source=None means load_real_features() loads
# everything from MySQL and lets build_continuous_series do its normal
# real-data-preferred blending (real kite_api/manual_csv rows preferred,
# proxy fills gaps). See load_real_features's docstring for details.
DATA_SOURCE = None

# Feature columns commonly useful as ARIMA exogenous regressors -- exposed
# as a multiselect in the UI rather than free text, since arima_exog_cols
# must be a subset of whatever feature_cols this run actually produces.
DEFAULT_ARIMA_EXOG_CHOICES = [
    "ret_mean_5", "ret_mean_10", "ret_mean_20",
    "comex_mcx_spread_z_10", "comex_mcx_spread_z_20", "gold_silver_ratio",
]


@dataclass
class RunConfig:
    contract: str = "SILVERMIC"
    start_date: str | None = None          # None = earliest available
    end_date: str | None = None            # None = latest available
    horizon: int = 1

    xgb_max_depth: int = 3
    xgb_n_estimators: int = 80
    arima_exog_cols: list[str] = field(default_factory=lambda: ["ret_mean_5", "comex_mcx_spread_z_10"])
    min_train_size: int = 120

    confidence_threshold: float = 0.5
    cooldown_days: int = 3

    fees: float = 0.00015
    slippage: float = 0.0003
    margin_pct: float = 0.12
    init_cash: float = 1_000_000.0
    lot_size: int = 5   # arbitrary demo lot size for the paper-broker replay -- verify current MCX lot size before using for real sizing decisions


@dataclass
class RunResult:
    config: RunConfig
    raw_row_count: int
    source_counts: dict
    n_model_ready_rows: int
    n_predictions: int
    signal_counts: dict
    comparison: pd.DataFrame               # strategy vs buy-and-hold, from compare_to_buy_and_hold
    trade_pnl: dict                        # from trade_pnl_breakdown
    strategy_equity: pd.Series             # for the equity curve chart
    buy_hold_equity: pd.Series
    broker_pnl: dict
    broker_n_orders: int
    date_range: tuple                      # (min_date, max_date) of predictions actually used
    warning: str = ""                      # non-fatal notes (e.g. very few predictions)
    price_series: Optional[pd.Series] = None    # mcx_close over model_ready's range, for the price chart
    trade_markers: Optional[pd.DataFrame] = None  # non-HOLD rows (signal, entry_price), for BUY/SELL markers
    position_series: Optional[pd.Series] = None  # 1=long, -1=short, 0=flat, per day -- for chart background shading


def run_pipeline(
    cfg: RunConfig,
    verbose: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> RunResult:
    """
    progress_callback: optional callable(message, pct) called at each stage
    boundary, plus once per walk-forward fold (the slow part) so callers
    (the Streamlit UI) can drive a real progress bar instead of a fake
    timer. Purely a UI hook -- no effect on results, defaults to None for
    every other caller (scripts, tests, feature-search harness).
    """
    def _report(message: str, pct: int) -> None:
        if progress_callback is not None:
            progress_callback(message, pct)

    _report("Loading data from MySQL (proxy + kite_api rows)...", 3)
    loaded = load_real_features(
        start_date=cfg.start_date, end_date=cfg.end_date, horizon=cfg.horizon,
        verbose=verbose, source=DATA_SOURCE, commodity=cfg.contract,
    )
    model_ready = loaded.model_ready
    feature_cols = loaded.feature_cols
    _report(f"Loaded {loaded.raw_row_count:,} raw rows, building feature matrix...", 10)

    if cfg.start_date:
        model_ready = model_ready[model_ready.index >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        model_ready = model_ready[model_ready.index <= pd.Timestamp(cfg.end_date)]

    # Engineered-feature warmup (the biggest is ema_200 -- see
    # features/indicators.py's EMA_WINDOWS -- which needs 200 prior bars
    # before it stops being NaN) already ate into model_ready above, via
    # load_real_features()'s internal dropna(subset=feature_cols). On a
    # short date range (e.g. "last 1 year" ~= 250 trading days) that alone
    # can leave too few rows for the requested min_train_size. Rather than
    # hard-failing, shrink min_train_size to fit -- leaving a floor of
    # MIN_VIABLE_TRAIN_SIZE rows to still fit a model on, and surfacing a
    # warning so it's clear predictions are running on less training data
    # than requested (and will be noisier for it).
    MIN_VIABLE_TRAIN_SIZE = 20
    effective_min_train_size = cfg.min_train_size
    min_train_auto_adjusted = False
    if len(model_ready) <= effective_min_train_size:
        effective_min_train_size = max(MIN_VIABLE_TRAIN_SIZE, len(model_ready) - 20)
        min_train_auto_adjusted = True
    if len(model_ready) <= effective_min_train_size:
        raise ValueError(
            f"Only {len(model_ready)} rows in [{cfg.start_date}, {cfg.end_date}] after "
            f"feature warmup (ema_200 alone needs 200 prior bars) -- not enough to fit any "
            f"model even at the {MIN_VIABLE_TRAIN_SIZE}-row floor. Widen the date range."
        )

    usable_exog_cols = [c for c in cfg.arima_exog_cols if c in feature_cols]

    wf_config = WalkForwardConfig(
        horizon=cfg.horizon,
        min_train_size=effective_min_train_size,
        arima_exog_cols=usable_exog_cols,
        xgb_params={"n_estimators": cfg.xgb_n_estimators, "max_depth": cfg.xgb_max_depth},
    )

    # Walk-forward is by far the slowest stage (refits ARIMA + XGBoost per
    # fold) so it gets the bulk of the progress bar's range, 12% -> 80%,
    # ticked once per fold via progress_callback.
    def _wf_progress(done: int, total: int) -> None:
        pct = 12 + int(68 * done / max(total, 1))
        _report(f"Walk-forward validation -- fold {done:,}/{total:,}...", min(pct, 80))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf_results = run_walk_forward(
            model_ready, feature_cols=feature_cols, target_col="target", config=wf_config,
            progress_callback=_wf_progress if progress_callback is not None else None,
        )

    if wf_results.empty:
        raise ValueError("Walk-forward produced zero predictions on this configuration -- "
                          "try a wider date range or a smaller min_train_size.")

    warnings_list = []
    if min_train_auto_adjusted:
        warnings_list.append(
            f"Requested min_train_size={cfg.min_train_size} didn't fit this date range "
            f"({len(model_ready)} rows after warmup) -- auto-reduced to "
            f"{effective_min_train_size} so it could still run. Results with this few "
            f"training rows are noisier than a longer run; widen the date range or lower "
            f"min_train_size yourself for a more deliberate trade-off."
        )
    elif len(wf_results) < 30:
        warnings_list.append(f"Only {len(wf_results)} predictions -- treat these results as noise, not a reliable signal.")

    _report(f"Got {len(wf_results):,} predictions, generating trade signals...", 82)
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    signal_config = SignalConfig(confidence_threshold=cfg.confidence_threshold, cooldown_days=cfg.cooldown_days)
    signals_df = generate_signals(signal_input, signal_config)

    bt_config = BacktestConfig(
        fees=cfg.fees, slippage=cfg.slippage, margin_pct=cfg.margin_pct, init_cash=cfg.init_cash,
    )

    # A confidence_threshold that's too strict for this particular
    # prediction magnitude/date range can leave every row HOLD (nothing to
    # backtest). Rather than hard-failing, retry with progressively lower
    # thresholds against the SAME predictions already computed above (cheap
    # -- generate_signals() is just a filter, no retraining needed) and
    # surface a warning about the substitution, same pattern as the
    # min_train_size auto-clamp above.
    n_trades = int((signals_df["signal"] != "HOLD").sum())
    if n_trades == 0:
        for candidate in [cfg.confidence_threshold * f for f in (0.5, 0.25, 0.1)] + [0.0]:
            candidate_config = SignalConfig(confidence_threshold=candidate, cooldown_days=cfg.cooldown_days)
            candidate_signals = generate_signals(signal_input, candidate_config)
            candidate_trades = int((candidate_signals["signal"] != "HOLD").sum())
            if candidate_trades > 0:
                signals_df = candidate_signals
                n_trades = candidate_trades
                warnings_list.append(
                    f"confidence_threshold={cfg.confidence_threshold:g} produced zero trades (every row "
                    f"HOLD) -- auto-lowered to {candidate:.3f} so it could still run. Results at this "
                    f"threshold are less selective than requested; set a lower threshold yourself for a "
                    f"more deliberate trade-off, or widen the date range."
                )
                break

    if n_trades == 0:
        raise ValueError(
            "Every row is HOLD even at confidence_threshold=0.0 -- the volatility-regime filter "
            "(ATR percentile) is suppressing every row on this configuration, not the confidence "
            "threshold. Try a different date range."
        )

    signal_counts = signals_df["signal"].value_counts().to_dict()
    warning = " | ".join(warnings_list)

    _report("Backtesting strategy vs buy-and-hold...", 87)
    strategy_pf = run_strategy_backtest(signals_df, price_col="entry_price", config=bt_config)
    bh_pf = run_buy_and_hold_backtest(signals_df["entry_price"], config=bt_config)
    comparison = pd.DataFrame({
        "strategy": performance_report(strategy_pf, bt_config),
        "buy_and_hold": performance_report(bh_pf, bt_config),
    }).T
    trade_pnl = trade_pnl_breakdown(strategy_pf)

    _report("Replaying through the paper broker...", 94)
    # --- Paper-broker replay (Phase 7b) -- independent accounting check ---
    broker = PaperKiteBroker(initial_cash=bt_config.init_cash)
    symbol = f"{cfg.contract}FUT"
    current_qty = 0
    for date, row in signals_df.iterrows():
        broker.update_market_price(symbol, row["entry_price"])
        desired = {"BUY": cfg.lot_size, "SELL": -cfg.lot_size, "HOLD": 0}[row["signal"]]
        delta = desired - current_qty
        if delta != 0:
            broker.place_order(symbol, transaction_type="BUY" if delta > 0 else "SELL",
                                quantity=abs(delta), price=row["entry_price"], tag=str(date.date()))
            current_qty = desired
    broker_pnl = broker.get_pnl()

    _report("Done.", 100)
    return RunResult(
        config=cfg,
        raw_row_count=loaded.raw_row_count,
        source_counts=loaded.source_counts,
        n_model_ready_rows=len(model_ready),
        n_predictions=len(wf_results),
        signal_counts=signal_counts,
        comparison=comparison,
        trade_pnl=trade_pnl,
        strategy_equity=strategy_pf.value(),
        buy_hold_equity=bh_pf.value(),
        broker_pnl=broker_pnl,
        broker_n_orders=len(broker.get_orders()),
        date_range=(signals_df.index.min(), signals_df.index.max()),
        warning=warning,
        price_series=model_ready["mcx_close"] if "mcx_close" in model_ready.columns else None,
        trade_markers=signals_df[signals_df["signal"] != "HOLD"][["signal", "entry_price"]].copy(),
        position_series=signals_to_position(signals_df["signal"]),
    )


# ---------------------------------------------------------------------------
# Startup data-freshness check (4th requested feature): once per session,
# the UI calls check_and_fetch_missing_data() before the first RUN so a
# stale table (e.g. hasn't run since last Friday) gets today's missing days
# pulled in automatically instead of silently running a backtest on old
# data. Deliberately separate from run_pipeline() itself -- this only needs
# to happen once per session/day, not on every RUN click, and it needs its
# own progress reporting (a short "fetching missing data" status) distinct
# from the pipeline's own progress bar.
# ---------------------------------------------------------------------------

import datetime as _dt


@dataclass
class FreshnessResult:
    """Outcome of check_and_fetch_missing_data(), for the UI to render."""
    checked: bool                  # False if the freshness check itself failed (e.g. DB unreachable)
    latest_before: object          # pd.Timestamp | None -- latest date already stored
    missing_days: int              # trading-day-ish gap (calendar days, see docstring below)
    fetched: bool                  # True if a fetch was actually attempted
    rows_upserted: int
    message: str                   # human-readable summary for the UI


def check_and_fetch_missing_data(
    commodity: str = "SILVERMIC",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> FreshnessResult:
    """
    Checks the latest stored date for `commodity` against today's date; if
    they don't match (there's a gap -- could be just a weekend, could be
    several missed days), fetches the missing range via
    data.kite_fetcher.fetch_missing_range() (which itself falls back to the
    COMEX+USDINR proxy if Kite isn't configured) and upserts it into MySQL.

    Never raises for a "no fetch possible" situation -- e.g. if both Kite
    and the proxy fail, this returns a FreshnessResult with fetched=False
    and rows_upserted=0 plus an explanatory message, so the UI can show that
    message and still let the user run the pipeline against whatever's
    already stored, rather than crashing the app.
    """
    def _log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    from data.db import get_latest_date

    try:
        latest = get_latest_date(commodity=commodity)
    except Exception as e:
        return FreshnessResult(
            checked=False, latest_before=None, missing_days=0,
            fetched=False, rows_upserted=0,
            message=f"Couldn't check data freshness (DB unreachable?): {e}",
        )

    today = _dt.date.today()

    if latest is None:
        # No data at all yet for this commodity -- not this function's job
        # to do a full multi-year backfill (that's kite_fetcher.backfill_history,
        # a deliberate one-time/manual step); just report it so the UI can
        # tell the user to run that first.
        return FreshnessResult(
            checked=True, latest_before=None, missing_days=0,
            fetched=False, rows_upserted=0,
            message=(
                f"No {commodity} data stored yet -- run data/kite_fetcher.py "
                f"--mode backfill (or the manual CSV/proxy build) once before "
                f"using this UI."
            ),
        )

    latest_date = latest.date()
    if latest_date >= today:
        return FreshnessResult(
            checked=True, latest_before=latest, missing_days=0,
            fetched=False, rows_upserted=0,
            message=f"Data already up to date (latest stored: {latest_date}).",
        )

    missing_days = (today - latest_date).days
    from_date = latest_date + _dt.timedelta(days=1)

    _log(f"Latest stored data is {latest_date} ({missing_days} day(s) behind today, "
         f"{today}) -- fetching missing data first...")

    try:
        n = fetch_missing_range(commodity, from_date, today, log=_log)
    except Exception as e:
        return FreshnessResult(
            checked=True, latest_before=latest, missing_days=missing_days,
            fetched=False, rows_upserted=0,
            message=f"Fetch attempt failed ({e}) -- proceeding with existing data (up to {latest_date}).",
        )

    if n == 0:
        message = (
            f"No new rows found for {from_date} to {today} (likely just a "
            f"weekend/holiday gap, or {today} hasn't closed yet) -- proceeding "
            f"with existing data (up to {latest_date})."
        )
    else:
        message = f"Fetched and stored {n} new row(s) covering {from_date} to {today}."

    return FreshnessResult(
        checked=True, latest_before=latest, missing_days=missing_days,
        fetched=True, rows_upserted=n, message=message,
    )


def load_preview_price_series(commodity: str = "SILVERMIC", lookback_days: int | None = None) -> pd.Series | None:
    """
    Cheap "just show me the price" load for the UI's pre-run preview chart --
    deliberately NOT load_real_features() (which also merges in live global
    factors via yfinance and runs the full feature-engineering pass; too
    slow to call just to draw a line chart). This only loads raw OHLCV rows
    for `commodity` from MySQL and stitches them into one continuous close
    series via data.contract_roll.build_continuous_series -- the same roll
    logic the real pipeline uses, just without the feature/model layers on
    top.

    lookback_days: None (default) shows EVERYTHING stored, however far back
    it goes -- a previous version defaulted this to 730 days, which silently
    clipped anything older than 2 years even though it was sitting right
    there in MySQL. Pass an explicit number only if you actually want a
    shorter preview window.

    Returns None if there's no data at all yet for this commodity (fresh DB,
    nothing fetched) -- the UI shows a placeholder message in that case
    rather than an empty chart.
    """
    from data.db import load_ohlcv
    from data.contract_roll import build_continuous_series

    raw = load_ohlcv()
    if raw.empty:
        return None

    raw = raw[raw["contract"].str.split("_").str[0] == commodity]
    if raw.empty:
        return None

    if lookback_days is not None:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
        raw = raw[raw.index >= cutoff]
        if raw.empty:
            return None

    continuous = build_continuous_series(raw)
    return continuous["close"]
