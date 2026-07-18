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
from backtest.walk_forward import run_walk_forward, WalkForwardConfig
from signals.signal_engine import SignalConfig, generate_signals
from backtest.vectorbt_backtest import (
    BacktestConfig, run_strategy_backtest, run_buy_and_hold_backtest,
    performance_report, trade_pnl_breakdown,
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

    fees: float = 0.0003
    slippage: float = 0.0005
    margin_pct: float = 0.15
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

    if len(model_ready) <= cfg.min_train_size:
        raise ValueError(
            f"Only {len(model_ready)} rows in [{cfg.start_date}, {cfg.end_date}] after "
            f"warmup -- not enough to clear min_train_size={cfg.min_train_size}. Widen the "
            f"date range or lower min_train_size."
        )

    usable_exog_cols = [c for c in cfg.arima_exog_cols if c in feature_cols]

    wf_config = WalkForwardConfig(
        horizon=cfg.horizon,
        min_train_size=cfg.min_train_size,
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

    warning = ""
    if wf_results.empty:
        raise ValueError("Walk-forward produced zero predictions on this configuration -- "
                          "try a wider date range or a smaller min_train_size.")
    if len(wf_results) < 30:
        warning = f"Only {len(wf_results)} predictions -- treat these results as noise, not a reliable signal."

    _report(f"Got {len(wf_results):,} predictions, generating trade signals...", 82)
    signal_input = wf_results[["meta_pred"]].join(
        model_ready[["atr_14", "ret_std_20", "mcx_close"]], how="left"
    )
    signal_config = SignalConfig(confidence_threshold=cfg.confidence_threshold, cooldown_days=cfg.cooldown_days)
    signals_df = generate_signals(signal_input, signal_config)
    signal_counts = signals_df["signal"].value_counts().to_dict()

    bt_config = BacktestConfig(
        fees=cfg.fees, slippage=cfg.slippage, margin_pct=cfg.margin_pct, init_cash=cfg.init_cash,
    )

    n_trades = int((signals_df["signal"] != "HOLD").sum())
    if n_trades == 0:
        raise ValueError("Every row is HOLD at this confidence threshold -- no trades to backtest. "
                          "Lower --confidence-threshold or widen the date range.")

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
    )
