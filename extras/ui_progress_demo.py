"""
ui/progress_demo.py

Standalone sanity check for two things fixed in app.py:
  1. Real progress bar + status text driven by a progress_callback(msg, pct)
     from a background thread (same pattern as run_pipeline in
     ui/pipeline_runner.py -- just a fake 12-"fold" loop instead of a real
     walk-forward, so this runs in a few seconds with zero dependencies).
  2. The multiselect pill-text contrast fix (white pill, black text).

Doesn't import anything from the rest of the project -- just streamlit,
threading, time. Run it on its own with:

    streamlit run ui/progress_demo.py
"""

from __future__ import annotations

import threading
import time

import streamlit as st

st.set_page_config(page_title="Progress + contrast demo", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0A0A0A; }

    div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] {
        background-color: #FFFFFF !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="tag"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        fill: #000000 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="popover"] li { color: #F2F2F2 !important; }

    div[data-testid="stProgress"] > div > div > div { background-color: #FFFFFF; }
    .run-status-text { color: #AAAAAA; font-size: 0.85rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.title("Progress bar + pill contrast demo")

st.subheader("1. Pill text visibility")
st.caption("Selected items below should show BLACK text on WHITE pills, not blank pills.")
choice = st.multiselect(
    "Pick a few columns",
    ["ret_mean_5", "ret_mean_10", "ret_mean_20", "comex_mcx_spread_z_10",
     "comex_mcx_spread_z_20", "gold_silver_ratio"],
    default=["ret_mean_5", "comex_mcx_spread_z_10"],
)

st.markdown("---")
st.subheader("2. Real progress bar")
st.caption("Simulates 12 'folds' of work with a real progress_callback(msg, pct) -- same wiring as the actual pipeline.")

run_clicked = st.button("RUN DEMO")

if run_clicked:
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    progress_state = {"pct": 0, "msg": "Starting..."}
    outcome = {}

    def fake_pipeline(progress_callback):
        progress_callback("Loading data...", 5)
        time.sleep(0.5)

        total_folds = 12
        for i in range(total_folds + 1):
            pct = 10 + int(70 * i / total_folds)
            progress_callback(f"Walk-forward fold {i}/{total_folds}...", pct)
            time.sleep(0.25)  # stand-in for a real fold's ARIMA+XGBoost fit

        progress_callback("Generating signals...", 85)
        time.sleep(0.4)
        progress_callback("Backtesting...", 92)
        time.sleep(0.4)
        progress_callback("Done.", 100)
        return "ok"

    def _worker():
        outcome["result"] = fake_pipeline(
            lambda msg, pct: progress_state.update(msg=msg, pct=pct)
        )

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    while worker.is_alive():
        progress_bar.progress(progress_state["pct"])
        status_placeholder.markdown(
            f'<div class="run-status-text">{progress_state["msg"]}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.1)

    worker.join()
    progress_bar.progress(100)
    status_placeholder.markdown('<div class="run-status-text">Done.</div>', unsafe_allow_html=True)
    time.sleep(0.3)
    progress_bar.empty()
    status_placeholder.empty()
    st.success(f"Demo finished, selected columns: {choice}")
