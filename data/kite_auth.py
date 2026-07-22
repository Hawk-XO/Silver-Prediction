"""
data/kite_auth.py

Handles Kite Connect's login/token-exchange flow. Kite access tokens expire
every day (Zerodha's standard behavior, not a bug) — so this isn't a one-time
setup step, it's something you'll re-run each morning before the first fetch,
until we wire up full automation in a later phase.

Usage
-----
Step 1 — get the login URL and open it in your browser:

    python -m data.kite_auth login-url

Step 2 — log in with your Zerodha credentials. You'll get redirected to your
Redirect URL (e.g. https://127.0.0.1/?...&request_token=XXXX&...) — the page
itself will look broken/unreachable, that's expected. Copy the request_token
value out of the address bar.

Step 3 — exchange it for an access token (this writes KITE_ACCESS_TOKEN
straight into your .env file, so nothing to copy-paste manually):

    python -m data.kite_auth exchange <request_token>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from kiteconnect import KiteConnect

from config.settings import settings

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def get_login_url() -> str:
    if not settings.kite_api_key:
        raise RuntimeError(
            "KITE_API_KEY is blank in .env — paste in your API Key from the "
            "Kite Connect app page first."
        )
    kite = KiteConnect(api_key=settings.kite_api_key)
    return kite.login_url()


def exchange_request_token(request_token: str) -> str:
    """Exchanges a request_token for an access_token and writes it into .env.
    Returns the access_token (also useful if you want to use it immediately
    in the same process without re-reading .env)."""
    if not settings.kite_api_key or not settings.kite_api_secret:
        raise RuntimeError(
            "KITE_API_KEY and/or KITE_API_SECRET is blank in .env — fill "
            "those in from the Kite Connect app page first."
        )

    kite = KiteConnect(api_key=settings.kite_api_key)
    session_data = kite.generate_session(request_token, api_secret=settings.kite_api_secret)
    access_token = session_data["access_token"]

    _write_access_token_to_env(access_token)
    apply_access_token_to_running_settings(access_token)
    print("Success. KITE_ACCESS_TOKEN has been written to .env.")
    print(f"(Zerodha client: {session_data.get('user_id', '?')} — "
          f"token valid until end of trading day / next login, whichever is first.)")
    return access_token


def _write_access_token_to_env(access_token: str) -> None:
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f".env not found at {ENV_PATH} — copy .env.example to .env first."
        )

    text = ENV_PATH.read_text()
    pattern = re.compile(r"^KITE_ACCESS_TOKEN=.*$", re.MULTILINE)
    new_line = f"KITE_ACCESS_TOKEN={access_token}"

    if pattern.search(text):
        text = pattern.sub(new_line, text)
    else:
        text = text.rstrip("\n") + f"\n{new_line}\n"

    ENV_PATH.write_text(text)


def apply_access_token_to_running_settings(access_token: str) -> None:
    """
    settings.settings is a frozen dataclass built once at import time, so
    writing a new KITE_ACCESS_TOKEN to .env alone doesn't take effect until
    the process restarts. This patches the already-loaded singleton in
    place (object.__setattr__ bypasses the frozen dataclass's normal
    write-protection) so a token exchanged mid-session through the UI's
    connect flow works on the very next fetch, with no restart needed.
    """
    from config import settings as settings_module
    object.__setattr__(settings_module.settings, "kite_access_token", access_token)


def validate_access_token(access_token: str | None = None) -> tuple[bool, str]:
    """
    Checks whether a Kite access token actually works right now (as opposed
    to just being a non-empty string) -- a token can be present in .env but
    expired (Zerodha expires them daily) or simply wrong. Calls kite.profile(),
    the cheapest authenticated endpoint Kite Connect offers, purely as a
    connectivity/validity check.

    Returns (is_valid, message) rather than raising, so UI code can show
    the message directly without a try/except at the call site.
    """
    from config.settings import settings

    token = access_token or settings.kite_access_token
    if not settings.kite_api_key or not token:
        return False, "Not connected — API key or access token missing."

    kite = KiteConnect(api_key=settings.kite_api_key)
    kite.set_access_token(token)
    try:
        profile = kite.profile()
        return True, f"Connected as {profile.get('user_name', profile.get('user_id', 'Zerodha client'))}."
    except Exception as e:
        return False, f"Token rejected ({e}) — likely expired, needs a fresh login."



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login-url", help="Print the URL to open in your browser to log in")
    exchange_parser = sub.add_parser("exchange", help="Exchange a request_token for an access_token")
    exchange_parser.add_argument("request_token", help="The request_token copied from the redirect URL")

    args = parser.parse_args()

    if args.command == "login-url":
        print(get_login_url())
    elif args.command == "exchange":
        exchange_request_token(args.request_token)


if __name__ == "__main__":
    main()
