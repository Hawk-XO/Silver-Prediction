"""
ui/pages/1_menu.py

First screen in the multipage flow. Job: make sure stored data is up to
date BEFORE either downstream screen runs anything against it, then let
the user pick where to go next.

    1. Show latest stored date vs today.
    2. "Check & fetch missing data" button -> check_and_fetch_missing_data()
       (same function ui/app.py used to call inline on first RUN -- moved
       here so it happens once, up front, instead of buried in either
       downstream page).
    3. Once checked this session, show two nav cards: Predictor / Market
       Simulator -- st.switch_page() to whichever the user picks.
"""

from __future__ import annotations

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

import streamlit as st

from ui.theme import inject_theme
from ui.pipeline_runner import check_and_fetch_missing_data

inject_theme()

st.title("SILVER PREDICTION")
st.caption("Start here -- confirm data is current, then choose a screen.")

if "freshness_checked_this_session" not in st.session_state:
    st.session_state.freshness_checked_this_session = False
if "freshness_result" not in st.session_state:
    st.session_state.freshness_result = None

CONTRACT = "SILVERMIC"  # only contract with data on this deployment -- see ui/pipeline_runner.AVAILABLE_CONTRACTS

st.subheader("1. Data freshness")

col_status, col_action = st.columns([3, 1])

with col_status:
    fr = st.session_state.freshness_result
    if fr is None:
        st.caption("Not checked yet this session.")
    else:
        if fr.latest_before is not None:
            st.caption(f"Latest stored ({CONTRACT}): **{fr.latest_before.date()}** -- {fr.message}")
        else:
            st.caption(fr.message)

with col_action:
    check_clicked = st.button(
        "Check & fetch" if not st.session_state.freshness_checked_this_session else "Re-check",
        width="stretch",
    )

if check_clicked:
    with st.status("Checking data freshness...", expanded=True) as status:
        def _log(msg: str) -> None:
            st.write(msg)

        result = check_and_fetch_missing_data(commodity=CONTRACT, progress_callback=_log)
        st.write(result.message)
        status.update(
            label="Data freshness check complete" if result.checked else "Data freshness check failed",
            state="complete" if result.checked else "error",
            expanded=False,
        )
    st.session_state.freshness_result = result
    st.session_state.freshness_checked_this_session = True
    st.rerun()

st.markdown("---")
st.subheader("2. Where to?")

if not st.session_state.freshness_checked_this_session:
    st.info("Run the freshness check above first (or proceed anyway -- both screens will run their own check if you skip this).")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        '<div class="nav-card"><h3>\u25c8 Predictor</h3>'
        "<p style='color:#AAAAAA'>Configure model hyperparameters and run a full walk-forward "
        "backtest -- Sharpe, drawdown, win rate, strategy vs buy-and-hold.</p></div>",
        unsafe_allow_html=True,
    )
    if st.button("Go to Predictor", width="stretch", key="go_predictor"):
        st.switch_page("pages/2_predictor.py")

with nav_col2:
    st.markdown(
        '<div class="nav-card"><h3>\u25b3 Forecast <span class="sim-badge">no trades \u00b7 no p&amp;l</span></h3>'
        "<p style='color:#AAAAAA'>Pick a date and see the model's own multi-day price line drawn "
        "next to the real one -- no orders, no fake cash, just the forecast.</p></div>",
        unsafe_allow_html=True,
    )
    if st.button("Go to Forecast", width="stretch", key="go_forecast"):
        st.switch_page("pages/3_forecast.py")
