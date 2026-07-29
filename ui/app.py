"""
ui/app.py

Entry point / router only. Actual page content lives in ui/pages/ -- this
file's only job is wiring st.navigation() and calling st.set_page_config()
exactly once (Streamlit errors if it's called more than once per session,
so individual pages must NOT call it themselves).

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

# Streamlit adds the script's OWN directory to sys.path, not the project
# root -- so `import ui...` fails with "No module named 'ui'" unless the
# project root is inserted first. This can't be delegated to a helper
# inside ui/ (e.g. ui.theme) because importing FROM the ui package is
# exactly what fails before this runs -- chicken-and-egg. Kept inline and
# dependency-free (plain sys/pathlib only) for that reason.
import sys as _sys
from pathlib import Path as _Path
_here = _Path(__file__).resolve()
for _candidate in [_here.parent, *_here.parents]:
    if (_candidate / "ui" / "__init__.py").exists():
        if str(_candidate) not in _sys.path:
            _sys.path.insert(0, str(_candidate))
        break

import streamlit as st

st.set_page_config(page_title="Silver Prediction", layout="wide", page_icon="\U0001F948")

pg = st.navigation([
    st.Page("pages/1_menu.py", title="Start", icon=":material/home:", default=True),
    st.Page("pages/2_predictor.py", title="Predictor", icon=":material/query_stats:"),
    st.Page("pages/3_forecast.py", title="Forecast", icon=":material/show_chart:"),
])
pg.run()
