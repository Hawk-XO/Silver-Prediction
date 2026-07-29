"""
ui/pages/3_forecast.py

Forecast screen (multipage app) -- replaces the old Market Simulator page.
No orders, no fake cash, no P&L: pick a start date, and the model draws its
own line -- a second, amber, dashed line next to the real price -- showing
where it thinks the market is heading over the next N trading days. The
model never gets to peek at real prices after start_date; each day's guess
is built only from its own prior guesses (see ui/forecast_runner.py for the
roll-forward mechanics).
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
from ui.forecast_runner import ForecastConfig, run_forecast, DEFAULT_ARIMA_EXOG_CHOICES
from ui.pipeline_runner import check_and_fetch_missing_data, load_preview_price_series
from ui.charts import build_forecast_chart, build_price_only_chart

inject_theme()

if st.button("\u2190 Start", key="back_to_menu_forecast"):
    st.switch_page("pages/1_menu.py")

st.title("SILVER FORECAST")
st.markdown('<span class="sim-badge">no trades \u00b7 no p&amp;l \u00b7 forecast only</span>', unsafe_allow_html=True)
st.caption(
    "Pick a date. The model trains on everything up to that point, then draws its own "
    "multi-day price line forward -- no peeking at what actually happened next."
)

CONTRACT = "SILVERMIC"

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Forecast setup")
    st.text(f"Contract: {CONTRACT}")

    st.subheader("Start date")
    start_date = st.date_input(
        "Model sees real data up to and including this date",
        value=pd.Timestamp.today() - pd.Timedelta(days=30),
        key="fc_start_date",
    ).isoformat()

    horizon_days = st.slider(
        "Forecast horizon (trading days forward)", min_value=5, max_value=60, value=20, key="fc_horizon",
    )

    with st.expander("Model settings"):
        xgb_max_depth = st.number_input("XGBoost max depth", min_value=1, max_value=10, value=3, key="fc_depth")
        xgb_n_estimators = st.number_input(
            "XGBoost n_estimators", min_value=10, max_value=500, value=80, step=10, key="fc_estimators",
        )
        arima_exog_cols = st.multiselect(
            "ARIMA exogenous features", options=DEFAULT_ARIMA_EXOG_CHOICES,
            default=["ret_mean_5", "comex_mcx_spread_z_10"], key="fc_exog",
        )
        min_train_size = st.number_input(
            "Min training rows", min_value=20, max_value=1000, value=120, step=10, key="fc_min_train",
        )

    st.markdown("---")
    run_clicked = st.button("RUN FORECAST", width="stretch")

# ----------------------------------------------------------------- main ---
if "fc_result" not in st.session_state:
    st.session_state.fc_result = None
if "fc_error" not in st.session_state:
    st.session_state.fc_error = None
if "fc_freshness_checked" not in st.session_state:
    st.session_state.fc_freshness_checked = False

if st.session_state.fc_result is None:
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
            f"Configure the sidebar and click RUN FORECAST."
        )
        st.plotly_chart(build_price_only_chart(preview_series), width="stretch")
    else:
        st.info("No data stored yet for this contract -- go to Start and run the freshness check first.")

    st.markdown("---")

if run_clicked:
    if not st.session_state.fc_freshness_checked:
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
        st.session_state.fc_freshness_checked = True

    cfg = ForecastConfig(
        contract=CONTRACT, start_date=start_date, horizon_days=int(horizon_days),
        xgb_max_depth=int(xgb_max_depth), xgb_n_estimators=int(xgb_n_estimators),
        arima_exog_cols=list(arima_exog_cols), min_train_size=int(min_train_size),
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
            outcome["result"] = run_forecast(cfg, progress_callback=_progress_callback)
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
        st.session_state.fc_result = outcome["result"]
        st.session_state.fc_error = None
    else:
        st.session_state.fc_result = None
        st.session_state.fc_error = outcome.get("error", "Unknown error.")

if st.session_state.fc_error:
    st.error(st.session_state.fc_error)

result = st.session_state.fc_result

if result is None:
    st.info("Configure the forecast in the sidebar and click RUN FORECAST.")
else:
    if result.warning:
        st.warning(result.warning)

    m1, m2, m3 = st.columns(3)
    m1.metric("Start date", result.start_date.date().isoformat())
    m2.metric("Forecast horizon", f"{result.config.horizon_days} trading days")
    m3.metric("Training rows used", f"{result.n_train_rows:,}")

    st.markdown("---")

    st.subheader("Actual vs. model forecast")
    st.plotly_chart(
        build_forecast_chart(result.actual_price, result.predicted_price, result.start_date),
        width="stretch",
    )
    st.caption(
        "White = real price. Amber dashed = the model's own guess, rolled forward day by day "
        "from the start date with no access to real prices after it. Global factors (COMEX "
        "silver, USD/INR, DXY, COMEX gold) are held constant over the forecast window -- only "
        "the technical/price-derived features evolve with the model's own predictions, which is "
        "why the line tends to flatten out the further it goes. Expected behavior, not a bug."
    )

    end_of_forecast = result.predicted_price.index.max()
    actual_after = result.actual_price[
        (result.actual_price.index > result.start_date) & (result.actual_price.index <= end_of_forecast)
    ]
    if not actual_after.empty:
        last_actual = actual_after.iloc[-1]
        last_pred = result.predicted_price.reindex(actual_after.index, method="nearest").iloc[-1]
        st.caption(
            f"As of {actual_after.index[-1].date()}: actual \u20b9{last_actual:,.2f} vs. "
            f"forecast \u20b9{last_pred:,.2f} ({100 * (last_pred / last_actual - 1):+.2f}% off)."
        )

    st.markdown("---")
    st.caption(
        "This screen only draws a predicted price path -- it never places an order, tracks "
        "cash, or computes P&L. See ui/forecast_runner.py for the exact roll-forward mechanics."
    )
