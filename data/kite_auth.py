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
