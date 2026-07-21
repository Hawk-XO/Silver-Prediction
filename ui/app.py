"""
ui/app.py

Track B control UI. Streamlit dashboard wrapping ui/pipeline_runner.run_pipeline
-- every widget here just builds a RunConfig and calls that one function.
No pipeline logic lives in this file; see ui/pipeline_runner.py for that.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Streamlit adds the script's OWN directory (ui/) to sys.path, not the
# project root -- so `from ui.pipeline_runner import ...` fails with
# "No module named 'ui'" when launched as `streamlit run ui/app.py`,
# since the `ui` package itself isn't importable from inside `ui/`.
# Inserting the project root (this file's parent's parent) fixes it
# regardless of the working directory `streamlit run` was invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.pipeline_runner import (
    RunConfig, run_pipeline, DEFAULT_ARIMA_EXOG_CHOICES,
    check_and_fetch_missing_data,
)
from ui.charts import build_price_chart, build_equity_chart
from ui.format_utils import build_display_comparison, metric_winner
from ui.report_export import build_pdf_report

st.set_page_config(page_title="Silver Prediction — Control", layout="wide", page_icon="◆")

# --- Minimal monochrome styling on top of the dark theme in .streamlit/config.toml ---
st.markdown("""
<style>
    .stApp { background-color: #0A0A0A; }
    h1, h2, h3 { font-weight: 500; letter-spacing: 0.02em; }
    div[data-testid="stMetricValue"] { font-family: monospace; }
    div[data-testid="stMetricLabel"] { color: #999999; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }
    .stButton button { background-color: #FFFFFF; color: #000000; border: none; border-radius: 2px; font-weight: 600; }
    .stButton button:hover { background-color: #CCCCCC; color: #000000; }
    hr { border-color: #2A2A2A; }

    /* Dropdown option lists (selectbox menus) -- make sure option text
       stays readable against the dark menu background. */
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="popover"] li { color: #F2F2F2 !important; }

    /* ------------------------------------------------------------------
       Checkbox contrast fix. Streamlit's dark theme uses primaryColor
       (white, from .streamlit/config.toml) for BOTH the checked box fill
       AND its checkmark glyph -- white-on-white, so the tick is invisible.
       Cover a few possible BaseWeb DOM shapes since the exact structure
       has changed across Streamlit versions; harmless if a rule below
       simply doesn't match anything in your installed version.
       ------------------------------------------------------------------ */
    label[data-baseweb="checkbox"] > span:first-child,
    label[data-baseweb="checkbox"] > div:first-child {
        border-color: #777777 !important;
    }
    label[data-baseweb="checkbox"] input:checked ~ span:first-child,
    label[data-baseweb="checkbox"] input:checked ~ div:first-child {
        background-color: #FFFFFF !important;
        border-color: #FFFFFF !important;
    }
    label[data-baseweb="checkbox"] input:checked ~ span:first-child svg,
    label[data-baseweb="checkbox"] input:checked ~ div:first-child svg,
    label[data-baseweb="checkbox"] svg {
        fill: #000000 !important;
        stroke: #000000 !important;
        color: #000000 !important;
    }

    /* ------------------------------------------------------------------
       st.pills (ARIMA exogenous columns) contrast fix -- selected pill
       renders with a white/primary-colored background; force dark text
       on it specifically instead of the default light theme text color.
       ------------------------------------------------------------------ */
    button[aria-pressed="true"],
    button[kind="pillsButtonActive"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    button[aria-pressed="true"] p,
    button[kind="pillsButtonActive"] p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* ------------------------------------------------------------------
       date_input's calendar popup: the selected-day cell renders with a
       white background AND white text -- invisible. Force both explicitly
       (previously only text color was forced, which wasn't enough since
       the background was the actual problem). Multiple selector shapes
       for robustness across BaseWeb versions.
       ------------------------------------------------------------------ */
    div[data-baseweb="calendar"] [aria-selected="true"],
    div[data-baseweb="calendar"] [aria-selected="true"] > div,
    div[data-baseweb="calendar"] div[role="gridcell"][aria-selected="true"] div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Progress bar + status text while a run is in flight */
    div[data-testid="stProgress"] > div > div > div { background-color: #FFFFFF; }
    .run-status-text { color: #AAAAAA; font-size: 0.85rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.title("SILVER PREDICTION — CONTROL")
st.caption("Real-data run: features → walk-forward → signals → backtest vs buy-and-hold → paper broker replay")

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Configuration")

    # SILVERMIC is the only contract we actually have data for -- not a
    # real choice, so no dropdown; shown as static info instead.
    contract = "SILVERMIC"
    st.text("Contract: SILVERMIC")

    st.subheader("Date range")
    use_full_history = st.checkbox("Use full available history", value=True)
    if use_full_history:
        start_date, end_date = None, None
    else:
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", value=pd.Timestamp("2023-01-01")).isoformat()
        end_date = col2.date_input("End", value=pd.Timestamp.today()).isoformat()

    horizon = st.number_input("Prediction horizon (days ahead)", min_value=1, max_value=20, value=1)

    st.subheader("Model hyperparameters")
    xgb_max_depth = st.slider("XGBoost max_depth", 1, 10, 3)
    xgb_n_estimators = st.slider("XGBoost n_estimators", 10, 300, 80, step=10)
    arima_exog_cols = st.pills(
        "ARIMA exogenous columns", DEFAULT_ARIMA_EXOG_CHOICES, selection_mode="multi",
        default=["ret_mean_5", "comex_mcx_spread_z_10"],
    ) or []
    min_train_size = st.number_input("Walk-forward min_train_size", min_value=10, max_value=1000, value=120, step=10)
    st.caption(
        "Feature warmup (ema_200 needs 200 prior bars) already eats into short date ranges "
        "before this even applies -- e.g. ~250 trading days (1yr) leaves ~50 rows. If that's "
        "less than min_train_size, it auto-shrinks to fit rather than failing (you'll see a "
        "warning); widen the date range for a more reliable run."
    )

    st.subheader("Signal settings")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 2.0, 0.5, step=0.01,
                                      help="Predicted move must be >= this x recent daily vol to trade")
    cooldown_days = st.number_input("Cooldown days", min_value=0, max_value=30, value=3)

    with st.expander("Backtest costs / sizing"):
        fees = st.number_input("Fees (fraction)", value=0.0003, format="%.4f")
        slippage = st.number_input("Slippage (fraction)", value=0.0005, format="%.4f")
        margin_pct = st.number_input("Margin % of notional", value=0.15, format="%.2f")
        init_cash = st.number_input("Initial cash", value=1_000_000.0, step=100_000.0)
        lot_size = st.number_input("Paper-broker lot size", value=5, min_value=1)

    st.markdown("---")
    run_clicked = st.button("RUN", width="stretch")

# ----------------------------------------------------------------- main ---
if "result" not in st.session_state:
    st.session_state.result = None
if "freshness_checked_this_session" not in st.session_state:
    st.session_state.freshness_checked_this_session = False

if run_clicked:
    # --- 4th requested feature: on the first RUN of the session, check
    # whether stored data is behind today's date and auto-fetch the gap
    # (Kite if configured, else the COMEX+USDINR proxy) before running the
    # pipeline, with a live status panel so it's obvious data is being
    # fetched rather than the app just looking stuck. ---
    if not st.session_state.freshness_checked_this_session:
        with st.status("Checking data freshness...", expanded=True) as status:
            def _freshness_log(msg: str) -> None:
                st.write(msg)

            freshness = check_and_fetch_missing_data(commodity=contract, progress_callback=_freshness_log)
            st.write(freshness.message)
            status.update(
                label="Data freshness check complete" if freshness.checked else "Data freshness check failed",
                state="complete" if freshness.checked else "error",
                expanded=False,
            )
        st.session_state.freshness_checked_this_session = True

    cfg = RunConfig(
        contract=contract, start_date=start_date, end_date=end_date, horizon=int(horizon),
        xgb_max_depth=int(xgb_max_depth), xgb_n_estimators=int(xgb_n_estimators),
        arima_exog_cols=arima_exog_cols, min_train_size=int(min_train_size),
        confidence_threshold=float(confidence_threshold), cooldown_days=int(cooldown_days),
        fees=float(fees), slippage=float(slippage), margin_pct=float(margin_pct),
        init_cash=float(init_cash), lot_size=int(lot_size),
    )
    # run_pipeline() now calls progress_callback(message, pct) at each real
    # stage boundary (data load, each walk-forward fold, signals, backtest,
    # broker replay) -- see ui/pipeline_runner.py. We run it on a background
    # thread and poll a shared dict from the main thread, since Streamlit
    # widgets can only be touched from the main thread.
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    progress_state = {"pct": 0, "msg": "Starting..."}
    outcome = {}

    def _progress_callback(msg: str, pct: int) -> None:
        progress_state["msg"] = msg
        progress_state["pct"] = pct

    def _worker():
        try:
            outcome["result"] = run_pipeline(cfg, progress_callback=_progress_callback)
        except Exception as e:
            outcome["error"] = str(e)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    while worker.is_alive():
        progress_bar.progress(progress_state["pct"])
        status_placeholder.markdown(
            f'<div class="run-status-text">{progress_state["msg"]}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.15)

    worker.join()
    progress_bar.progress(100)
    status_placeholder.markdown('<div class="run-status-text">Done.</div>', unsafe_allow_html=True)
    time.sleep(0.3)
    progress_bar.empty()
    status_placeholder.empty()

    if "result" in outcome:
        st.session_state.result = outcome["result"]
        st.session_state.error = None
    else:
        st.session_state.result = None
        st.session_state.error = outcome.get("error", "Unknown error.")

if st.session_state.get("error"):
    st.error(st.session_state.error)

result = st.session_state.result

if result is None:
    st.info("Configure parameters in the sidebar and click RUN.")
else:
    if result.warning:
        st.warning(result.warning)

    st.caption(
        f"{result.raw_row_count:,} raw rows ({result.source_counts}) → "
        f"{result.n_model_ready_rows:,} rows with complete features → "
        f"{result.n_predictions:,} walk-forward predictions, "
        f"{result.date_range[0].date()} to {result.date_range[1].date()}"
    )

    # --- top-line metrics ---
    strat = result.comparison.loc["strategy"]
    bh = result.comparison.loc["buy_and_hold"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sharpe", f"{strat['sharpe_ratio']:.3f}", f"{strat['sharpe_ratio'] - bh['sharpe_ratio']:+.3f} vs B&H")
    m2.metric("Total return", f"{strat['total_return_pct']:.2f}%", f"{strat['total_return_pct'] - bh['total_return_pct']:+.2f}pp vs B&H")
    m3.metric("Max drawdown", f"{strat['max_drawdown_pct']:.2f}%")
    m4.metric("Win rate", f"{strat['win_rate_pct']:.1f}%")
    m5.metric("Profit factor", f"{strat['profit_factor']:.3f}")

    st.markdown("---")

    # --- 1st requested feature: interactive TradingView-style Plotly charts,
    # replacing the old static matplotlib equity curve. ---
    st.subheader("Price chart — trade markers")
    if result.price_series is not None and not result.price_series.empty:
        st.plotly_chart(build_price_chart(result.price_series, result.trade_markers), width="stretch")
    else:
        st.caption("No price series available for this run.")

    st.subheader("Equity curve — strategy vs buy-and-hold")
    st.plotly_chart(
        build_equity_chart(result.strategy_equity, result.buy_hold_equity, result.config.init_cash),
        width="stretch",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Performance report")
        # Transposed (metrics as rows, strategy/buy&hold as columns) --
        # the old layout rendered 9 metrics as 9 COLUMNS, which is what
        # was getting cut off at 100% zoom. A fixed 2-data-column table
        # always fits regardless of screen width. Color-coded green/red
        # per metric for whichever side wins it (see
        # ui/format_utils.metric_winner for the higher/lower-is-better
        # rules per metric).
        display_comparison = build_display_comparison(result.comparison)
        raw_metrics = result.comparison.T.index.tolist()

        def _highlight_winner(row):
            metric = raw_metrics[display_comparison.index.get_loc(row.name)]
            winner = metric_winner(result.comparison, metric)
            styles = ["", ""]
            if winner == "strategy":
                styles = ["background-color: #123B22; color: #6FE39A; font-weight: 600", ""]
            elif winner == "buy_and_hold":
                styles = ["", "background-color: #123B22; color: #6FE39A; font-weight: 600"]
            return styles

        styled = display_comparison.style.apply(_highlight_winner, axis=1)
        st.dataframe(styled, width="stretch")

        st.subheader("Signal counts")
        signal_df = pd.DataFrame.from_dict(result.signal_counts, orient="index", columns=["count"])
        st.bar_chart(signal_df, color="#FFFFFF")

    with col_right:
        st.subheader("Trade-level PnL breakdown")
        pnl = result.trade_pnl
        p1, p2 = st.columns(2)
        p1.metric("Trades", pnl["num_trades"])
        p2.metric("Win / Loss", f"{pnl['num_wins']} / {pnl['num_losses']}")
        p1.metric("Avg win", f"{pnl['avg_win']:,.0f}" if pd.notna(pnl["avg_win"]) else "—")
        p2.metric("Avg loss", f"{pnl['avg_loss']:,.0f}" if pd.notna(pnl["avg_loss"]) else "—")
        p1.metric("Avg win/loss ratio", f"{pnl['avg_win_loss_ratio']:.2f}" if pd.notna(pnl["avg_win_loss_ratio"]) else "—")
        p2.metric("Expectancy/trade", f"{pnl['expectancy_per_trade']:,.0f}" if pd.notna(pnl["expectancy_per_trade"]) else "—")

        st.subheader("Paper broker replay")
        b1, b2, b3 = st.columns(3)
        b1.metric("Orders placed", result.broker_n_orders)
        b2.metric("Realised P&L", f"{result.broker_pnl['realised']:,.0f}")
        b3.metric("Total P&L", f"{result.broker_pnl['total']:,.0f}")

    st.markdown("---")

    # --- 3rd requested feature: export button, shown after the run's data
    # has finished rendering above (not blocking anything above it). ---
    export_col, _ = st.columns([1, 3])
    with export_col:
        if st.button("Export report as PDF", width="stretch"):
            with st.spinner("Building PDF report..."):
                pdf_bytes = build_pdf_report(result)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"silver_prediction_report_{pd.Timestamp.today().date()}.pdf",
                mime="application/pdf",
                width="stretch",
            )
