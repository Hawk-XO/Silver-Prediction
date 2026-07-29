"""
ui/pages/3_market_sim.py

Market Simulator screen (multipage app) -- the piece that didn't exist
before. "Market, not money": runs the exact same real-data pipeline as the
Predictor page (features -> walk-forward -> signals -> vectorbt backtest),
then replays the signals through PaperKiteBroker (already fake-money, see
broker/kite_paper_broker.py) and presents it as a trading-desk screen --
equity curve, order-by-order fill log, running P&L -- rather than the
Predictor page's model-comparison framing. Every screen is clearly labeled
simulated/no real money.

No new modeling logic lives here -- see ui/market_sim_runner.py, which is
a thin wrapper around ui/pipeline_runner.run_pipeline.
"""

from __future__ import annotations

import threading
import time

# Streamlit adds the script's OWN directory to sys.path, not the project
# root -- so `import ui...` fails with "No module named 'ui'" unless the
# project root is inserted first. Inlined (plain sys/pathlib only) because
# importing FROM the ui package is exactly what fails before this runs.
import sys as _sys
from pathlib import Path as _Path
_here = _Path(__file__).resolve()
for _candidate in [_here.parent, *_here.parents]:
    if (_candidate / "ui" / "__init__.py").exists():
        if str(_candidate) not in _sys.path:
            _sys.path.insert(0, str(_candidate))
        break

import pandas as pd
import streamlit as st

from ui.theme import inject_theme
from ui.market_sim_runner import MarketSimConfig, run_market_sim, build_orders_table
from ui.pipeline_runner import check_and_fetch_missing_data, load_preview_price_series
from ui.charts import build_price_chart, build_equity_chart

inject_theme()

if st.button("\u2190 Start", key="back_to_menu_market_sim"):
    st.switch_page("pages/1_menu.py")

st.title("SILVER MARKET SIMULATOR")
st.markdown('<span class="sim-badge">simulated -- no real money</span>', unsafe_allow_html=True)
st.caption("Same real-data pipeline as Predictor, replayed as paper-broker orders: fill-by-fill, running P&L.")

CONTRACT = "SILVERMIC"

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Simulation setup")
    st.text(f"Contract: {CONTRACT}")

    st.subheader("Date range")
    use_full_history = st.checkbox(
        "Use full available history", value=True,
        help="Uses every stored row for this contract.",
        key="sim_full_history",
    )
    if use_full_history:
        start_date, end_date = None, None
    else:
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", value=pd.Timestamp("2023-01-01"), key="sim_start").isoformat()
        end_date = col2.date_input("End", value=pd.Timestamp.today(), key="sim_end").isoformat()

    horizon = st.number_input(
        "Prediction horizon (days ahead)", min_value=1, max_value=20, value=1, key="sim_horizon",
    )

    st.subheader("Signal settings")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 2.0, 0.5, step=0.01, key="sim_conf",
        help="Higher = fewer, higher-conviction paper trades.",
    )
    cooldown_days = st.number_input(
        "Cooldown days", min_value=0, max_value=30, value=3, key="sim_cooldown",
    )

    with st.expander("Paper account settings"):
        init_cash = st.number_input(
            "Starting paper cash", value=1_000_000.0, step=100_000.0, key="sim_init_cash",
        )
        lot_size = st.number_input(
            "Lot size per order", value=5, min_value=1, key="sim_lot_size",
        )
        fees_pct = st.number_input("Fees (% of trade value)", value=0.015, step=0.001, format="%.3f", key="sim_fees")
        slippage_pct = st.number_input("Slippage (% of price)", value=0.030, step=0.005, format="%.3f", key="sim_slip")
        margin_pct_input = st.number_input("Margin (% of notional)", value=12.0, step=0.5, format="%.1f", key="sim_margin")

    st.markdown("---")
    run_clicked = st.button("RUN SIMULATION", width="stretch")

# ----------------------------------------------------------------- main ---
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "sim_error" not in st.session_state:
    st.session_state.sim_error = None
if "sim_freshness_checked" not in st.session_state:
    st.session_state.sim_freshness_checked = False

if st.session_state.sim_result is None:
    st.subheader("Data preview")
    try:
        preview_series = load_preview_price_series(commodity=CONTRACT)
    except Exception as e:
        preview_series = None
        st.caption(f"Couldn't load a preview (DB not reachable yet?): {e}")

    if preview_series is not None and not preview_series.empty:
        st.caption(
            f"{len(preview_series):,} stored trading days, "
            f"{preview_series.index.min().date()} to {preview_series.index.max().date()}. "
            f"Configure the sidebar and click RUN SIMULATION."
        )
    else:
        st.info("No data stored yet for this contract -- go to Start and run the freshness check first.")

    st.markdown("---")

if run_clicked:
    if not st.session_state.sim_freshness_checked:
        with st.status("Checking data freshness...", expanded=True) as status:
            def _freshness_log(msg: str) -> None:
                st.write(msg)
            freshness = check_and_fetch_missing_data(commodity=CONTRACT, progress_callback=_freshness_log)
            st.write(freshness.message)
            status.update(
                label="Data freshness check complete" if freshness.checked else "Data freshness check failed",
                state="complete" if freshness.checked else "error",
                expanded=False,
            )
        st.session_state.sim_freshness_checked = True

    cfg = MarketSimConfig(
        contract=CONTRACT, start_date=start_date, end_date=end_date, horizon=int(horizon),
        confidence_threshold=float(confidence_threshold), cooldown_days=int(cooldown_days),
        fees=float(fees_pct) / 100, slippage=float(slippage_pct) / 100, margin_pct=float(margin_pct_input) / 100,
        init_cash=float(init_cash), lot_size=int(lot_size),
    )

    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    progress_state = {"pct": 0, "msg": "Starting..."}
    outcome = {}

    def _progress_callback(msg: str, pct: int) -> None:
        progress_state["msg"] = msg
        progress_state["pct"] = pct

    def _worker():
        try:
            outcome["result"] = run_market_sim(cfg, progress_callback=_progress_callback)
        except Exception as e:
            outcome["error"] = str(e)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    while worker.is_alive():
        progress_bar.progress(progress_state["pct"])
        status_placeholder.markdown(
            f'<div class="run-status-text">{progress_state["msg"]}</div>', unsafe_allow_html=True,
        )
        time.sleep(0.15)
    worker.join()
    progress_bar.progress(100)
    time.sleep(0.2)
    progress_bar.empty()
    status_placeholder.empty()

    if "result" in outcome:
        st.session_state.sim_result = outcome["result"]
        st.session_state.sim_error = None
    else:
        st.session_state.sim_result = None
        st.session_state.sim_error = outcome.get("error", "Unknown error.")

if st.session_state.sim_error:
    st.error(st.session_state.sim_error)

result = st.session_state.sim_result

if result is None:
    st.info("Configure the simulation in the sidebar and click RUN SIMULATION.")
else:
    if result.warning:
        st.warning(result.warning)

    st.caption(
        f"{result.n_predictions:,} paper signals generated over "
        f"{result.date_range[0].date()} to {result.date_range[1].date()}"
    )

    # --- Paper account summary, front and center on this page ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Starting cash", f"\u20b9{result.config.init_cash:,.0f}")
    m2.metric("Realised P&L", f"\u20b9{result.broker_pnl['realised']:,.0f}")
    m3.metric("Unrealised P&L", f"\u20b9{result.broker_pnl['unrealised']:,.0f}")
    m4.metric("Total P&L (paper)", f"\u20b9{result.broker_pnl['total']:,.0f}",
              f"{100 * result.broker_pnl['total'] / result.config.init_cash:+.2f}%")

    st.markdown("---")

    st.subheader("Price chart -- paper trade markers")
    if result.price_series is not None and not result.price_series.empty:
        st.plotly_chart(
            build_price_chart(result.price_series, result.trade_markers, result.position_series),
            width="stretch",
        )
        st.caption("Green triangle = paper BUY (long) \u00b7 Red triangle = paper SELL (short)")
    else:
        st.caption("No price series available for this run.")

    st.subheader("Equity curve -- paper account vs buy-and-hold")
    st.plotly_chart(
        build_equity_chart(result.strategy_equity, result.buy_hold_equity, result.config.init_cash),
        width="stretch",
    )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Order log (paper broker)")
        orders_df = build_orders_table(result)
        if orders_df.empty:
            st.caption("No orders placed this run.")
        else:
            st.dataframe(orders_df, width="stretch", height=360)
        st.caption(f"{result.broker_n_orders:,} paper orders placed \u00b7 lot size {result.config.lot_size}")

    with col_right:
        st.subheader("Signal mix")
        signal_df = pd.DataFrame.from_dict(result.signal_counts, orient="index", columns=["count"])
        st.bar_chart(signal_df, color="#FFFFFF")

        st.subheader("Trade P&L")
        pnl = result.trade_pnl
        p1, p2 = st.columns(2)
        p1.metric("Trades", pnl["num_trades"])
        p2.metric("Win / Loss", f"{pnl['num_wins']} / {pnl['num_losses']}")
        p1.metric("Avg win", f"{pnl['avg_win']:,.0f}" if pd.notna(pnl["avg_win"]) else "\u2014")
        p2.metric("Avg loss", f"{pnl['avg_loss']:,.0f}" if pd.notna(pnl["avg_loss"]) else "\u2014")

    st.markdown("---")
    st.caption(
        "This screen simulates order placement against historical/synthetic data using "
        "PaperKiteBroker -- no real orders are ever sent, no real money moves. See "
        "broker/kite_paper_broker.py for what would need to change to point this at a real "
        "(sandbox or live) Kite Connect account."
    )
