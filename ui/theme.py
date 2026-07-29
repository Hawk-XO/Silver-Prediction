"""
ui/theme.py

Shared dark-theme CSS, reused by every page in ui/pages/ so the same
<style> block isn't copy-pasted three times. NOTE: the project-root
sys.path bootstrap (needed because `streamlit run ui/app.py` puts ui/ on
sys.path, not the project root) can't live here as an importable helper --
importing FROM the ui package is exactly what fails before that bootstrap
runs. Each entry-point script (ui/app.py and every file in ui/pages/)
inlines that bootstrap directly using only sys/pathlib, before its first
`ui.*` import; only AFTER that can it safely `from ui.theme import
inject_theme`.
"""

from __future__ import annotations



THEME_CSS = """
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
       Multiselect chip contrast fix. Same root cause as the checkbox fix
       above -- primaryColor (white, from .streamlit/config.toml) is used
       as the selected-tag background AND its text/close-icon color,
       so each selected item renders as a solid unreadable white block.
       ------------------------------------------------------------------ */
    span[data-baseweb="tag"], div[data-baseweb="tag"] {
        background-color: #2A2A2A !important;
        border: 1px solid #444444 !important;
    }
    span[data-baseweb="tag"] span, div[data-baseweb="tag"] span,
    span[data-baseweb="tag"] div, div[data-baseweb="tag"] div {
        color: #F2F2F2 !important;
        -webkit-text-fill-color: #F2F2F2 !important;
    }
    span[data-baseweb="tag"] svg, div[data-baseweb="tag"] svg {
        fill: #F2F2F2 !important;
    }
    span[data-baseweb="tag"] [role="button"]:hover,
    div[data-baseweb="tag"] [role="button"]:hover {
        background-color: #3A3A3A !important;
    }

    /* st.pills selected-state contrast fix */
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

    /* date_input calendar popup selected-day contrast fix */
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

    /* Nav cards on the menu screen */
    .nav-card {
        border: 1px solid #2A2A2A; border-radius: 4px; padding: 1.25rem;
        background-color: #111111;
    }
    .sim-badge {
        display: inline-block; background-color: #1A1A1A; color: #AAAAAA;
        border: 1px solid #333333; border-radius: 3px; padding: 0.15rem 0.5rem;
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
    }
</style>
"""


def inject_theme() -> None:
    import streamlit as st
    st.markdown(THEME_CSS, unsafe_allow_html=True)
