"""
ui/kite_connect.py

Sidebar panel for Kite Connect status + a one-click-ish connect flow.

IMPORTANT — what "automatic" can and can't mean here: Zerodha does not
offer a headless/fully-automated login API. Every Kite Connect app requires
a real login (password + 2FA) through Zerodha's own login page, by design
(this is a regulatory requirement, not a Zerodha API limitation we can code
around) -- so a silent, no-human-involved daily login isn't something any
app is allowed to do, ours included. What *is* automatable, and what this
panel does:

  1. Turns the old 3-step CLI process (run a script, copy a URL, run another
     script with a pasted token) into a 2-click UI flow: click "Open Zerodha
     login", log in normally, paste the one string from the redirect URL
     back in, click "Connect" -- no .env editing, no terminal.
  2. Once connected, EVERYTHING downstream is automatic for the rest of that
     day's session: check_and_fetch_missing_data() (see ui/pipeline_runner.py)
     already runs the freshness check + EOD pull on every first RUN without
     any further clicks, and this panel's "Pull latest EOD now" button lets
     you trigger that same pull on demand too.
  3. Shows live status (checks the token actually works via kite.profile(),
     not just "a token string exists" -- tokens expire daily) so it's always
     obvious whether today's login has been done yet.
"""

from __future__ import annotations

import streamlit as st

from data.kite_auth import get_login_url, exchange_request_token, validate_access_token
from config.settings import settings


def render_kite_connect_panel() -> None:
    st.subheader("Kite Connect")

    if not settings.kite_api_key or not settings.kite_api_secret:
        st.caption(
            "KITE_API_KEY / KITE_API_SECRET not set in .env -- add those from your "
            "Kite Connect app page first. Until then, data fetches fall back to the "
            "COMEX+USDINR proxy automatically."
        )
        return

    if "kite_status_checked" not in st.session_state:
        st.session_state.kite_status_checked = False
        st.session_state.kite_status_ok = False
        st.session_state.kite_status_msg = ""

    check_col, _ = st.columns([1, 1])
    if check_col.button("Check status", width="stretch", key="kite_check_status"):
        ok, msg = validate_access_token()
        st.session_state.kite_status_checked = True
        st.session_state.kite_status_ok = ok
        st.session_state.kite_status_msg = msg

    if not st.session_state.kite_status_checked:
        st.caption("Not checked yet this session -- click \"Check status\".")
    elif st.session_state.kite_status_ok:
        st.success(st.session_state.kite_status_msg)
    else:
        st.warning(st.session_state.kite_status_msg)

    with st.expander("Connect / refresh login", expanded=not st.session_state.kite_status_ok):
        st.markdown(
            "**1.** Open the Zerodha login page and log in as usual.\n\n"
            "**2.** After login you'll land on a broken/unreachable-looking redirect "
            "page -- that's expected. Copy the `request_token=...` value out of that "
            "page's URL.\n\n"
            "**3.** Paste it below and click Connect."
        )
        try:
            login_url = get_login_url()
            st.link_button("Open Zerodha login", login_url, width="stretch")
        except Exception as e:
            st.error(f"Couldn't build login URL: {e}")
            login_url = None

        request_token = st.text_input("Pasted request_token", key="kite_request_token")
        if st.button("Connect", width="stretch", key="kite_connect_btn", disabled=not request_token):
            try:
                exchange_request_token(request_token)
                ok, msg = validate_access_token()
                st.session_state.kite_status_checked = True
                st.session_state.kite_status_ok = ok
                st.session_state.kite_status_msg = msg
                if ok:
                    st.success("Connected. You can close this section now.")
                else:
                    st.error(f"Exchange succeeded but the token didn't validate: {msg}")
            except Exception as e:
                st.error(f"Connect failed: {e}")

    if st.button("Pull latest EOD now", width="stretch", key="kite_pull_eod_btn"):
        from ui.pipeline_runner import check_and_fetch_missing_data

        with st.status("Pulling latest EOD data...", expanded=True) as status:
            def _log(msg: str) -> None:
                st.write(msg)

            result = check_and_fetch_missing_data(commodity=settings.mcx_commodity, progress_callback=_log)
            st.write(result.message)
            status.update(
                label="Done" if result.checked else "Failed",
                state="complete" if result.checked else "error",
                expanded=False,
            )
