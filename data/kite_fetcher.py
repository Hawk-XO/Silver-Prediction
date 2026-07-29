"""
data/kite_fetcher.py

Pulls real MCX Silver EOD data via Zerodha's Kite Connect Historical Data
API and stores it in MySQL (mcx_silver_ohlcv, source='kite_api'). Requires
KITE_API_KEY, KITE_API_SECRET, and KITE_ACCESS_TOKEN to be set in .env — the
latter via data/kite_auth.py, since it expires daily.

Two entry points you'll actually use:

    backfill_history(commodity, years_back=2)
        One-time (or occasional) pull of past years of daily data across
        every contract expiry that existed in that window. Mirrors exactly
        what the manual-CSV route did, just automated.

    fetch_latest_eod(commodity)
        Pulls just the most recent trading day for the currently active
        contract(s) — this is what the daily automation job (a later phase)
        will call every evening after market close.

Design note: Kite's historical API does have a `continuous=1` mode that
stitches expiries together server-side, but that behavior is best-documented
for NFO/equity derivatives — MCX support isn't guaranteed the same way. To
stay consistent with the manual-CSV path (and avoid depending on an
unverified feature), this fetches each contract separately and reuses the
same build_continuous_series() ratio-adjustment already tested against your
real CSV data.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import settings
from data.contract_roll import build_continuous_series
from data.db import upsert_ohlcv, get_engine
from data.retry import retry_with_backoff

MCX_EXCHANGE = "MCX"


@retry_with_backoff(attempts=3, initial_delay=1.0, backoff_factor=2.0)
def _kite_instruments(kite: KiteConnect, exchange: str):
    """Retry-wrapped kite.instruments() -- transient network blips shouldn't
    take down a contract lookup that would otherwise succeed on retry."""
    return kite.instruments(exchange)


@retry_with_backoff(attempts=3, initial_delay=1.0, backoff_factor=2.0)
def _kite_historical_data(kite: KiteConnect, **kwargs):
    """Retry-wrapped kite.historical_data() -- same rationale as
    _kite_instruments() above; this is the call made once per contract per
    fetch, so it's the one most exposed to a flaky connection."""
    return kite.historical_data(**kwargs)


def _get_client() -> KiteConnect:
    if not settings.kite_configured:
        raise RuntimeError(
            "Kite Connect isn't fully configured yet (need KITE_API_KEY, "
            "KITE_API_SECRET, and KITE_ACCESS_TOKEN all set in .env — the "
            "access token comes from running data/kite_auth.py). "
            "Use the mcx_proxy or manual CSV route until then."
        )
    kite = KiteConnect(api_key=settings.kite_api_key)
    kite.set_access_token(settings.kite_access_token)
    return kite


def list_contracts(commodity: str, kite: KiteConnect | None = None) -> pd.DataFrame:
    """
    Returns every MCX futures contract for a given commodity (e.g.
    'SILVERMIC'), one row per expiry, with the instrument_token needed to
    fetch historical data for it.
    """
    kite = kite or _get_client()
    instruments = pd.DataFrame(_kite_instruments(kite, MCX_EXCHANGE))

    if instruments.empty:
        raise RuntimeError("Kite returned no MCX instruments — check API access / market hours.")

    matches = instruments[
        (instruments["name"].str.upper() == commodity.upper())
        & (instruments["segment"].str.upper().str.contains("FUT"))
    ].copy()

    if matches.empty:
        available = sorted(instruments["name"].dropna().unique().tolist())
        raise ValueError(
            f"No MCX futures contracts found for commodity='{commodity}'. "
            f"Available commodity names on MCX right now include: {available[:20]}..."
        )

    matches["expiry"] = pd.to_datetime(matches["expiry"])
    return matches.sort_values("expiry")[
        ["instrument_token", "tradingsymbol", "name", "expiry", "lot_size"]
    ].reset_index(drop=True)


def fetch_contract_history(
    instrument_token: int,
    from_date: dt.date,
    to_date: dt.date,
    kite: KiteConnect | None = None,
) -> pd.DataFrame:
    """
    Fetches daily OHLCV+OI for a single contract over a date range.
    Kite's historical API caps how much can be requested per call for daily
    interval (documented limit is generous for 'day' interval — years at a
    time — but if you hit a "date range too big" error, this is where a
    chunking loop would go; not implemented here since daily-interval limits
    are large enough for our multi-year use case in one shot).
    """
    kite = kite or _get_client()
    raw = _kite_historical_data(
        kite,
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval="day",
        continuous=False,
        oi=True,
    )
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns={"oi": "open_interest"})
    return df[["open", "high", "low", "close", "volume", "open_interest"]]


def _fetch_all_contracts_in_window(
    commodity: str,
    window_start: dt.date,
    window_end: dt.date,
    kite: KiteConnect | None = None,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Shared per-contract fetch loop used by both backfill_history() (wide,
    years-back window) and fetch_missing_range() (narrow, just-the-gap
    window). Walks every contract expiry for `commodity`, pulls whatever
    Kite has for the overlap between [window_start, window_end] and that
    contract's lifetime, and returns the concatenated raw per-contract rows
    (NOT yet upserted -- callers own that, since backfill_history() also
    needs the continuous series before it upserts).

    log: optional callable(str) for progress messages -- defaults to
    print() when not provided, so this stays usable from plain scripts too.
    """
    log = log or print
    kite = kite or _get_client()
    contracts = list_contracts(commodity, kite=kite)

    all_contract_frames = []
    for _, row in contracts.iterrows():
        expiry = row["expiry"].date()
        if expiry < window_start:
            continue  # contract expired before our window even starts

        # Request the full window and let Kite return whatever actually
        # exists — if the contract wasn't listed that far back, Kite simply
        # returns data starting from its real listing date, so there's no
        # need (and no benefit) to guess a cutoff ourselves.
        contract_start = window_start
        contract_end = min(window_end, expiry)
        if contract_start > contract_end:
            continue

        log(f"Fetching {row['tradingsymbol']} (expiry {expiry}) "
            f"from {contract_start} to {contract_end} ...")
        hist = fetch_contract_history(row["instrument_token"], contract_start, contract_end, kite=kite)
        if hist.empty:
            log(f"  -> no data returned (may be too old for Kite's history retention).")
            continue

        hist["contract"] = f"{commodity}_{row['tradingsymbol']}"
        all_contract_frames.append(hist)

    if not all_contract_frames:
        return pd.DataFrame()
    return pd.concat(all_contract_frames)


def backfill_history(commodity: str, years_back: int = 2) -> pd.DataFrame:
    """
    Pulls `years_back` years of daily history across every contract expiry
    for `commodity`, stitches them into a continuous series, and upserts the
    result into MySQL (source='kite_api'). Returns the continuous DataFrame.
    """
    kite = _get_client()
    today = dt.date.today()
    window_start = today - dt.timedelta(days=365 * years_back)

    multi_contract_df = _fetch_all_contracts_in_window(commodity, window_start, today, kite=kite)

    if multi_contract_df.empty:
        raise RuntimeError(
            f"No historical data returned for any {commodity} contract in the "
            f"last {years_back} years — check KITE_ACCESS_TOKEN is fresh (it "
            f"expires daily) and that {commodity} is the right commodity name."
        )

    continuous = build_continuous_series(multi_contract_df)

    engine = get_engine()
    n = upsert_ohlcv(multi_contract_df, source="kite_api", engine=engine)
    print(f"Upserted {n} raw per-contract rows into mcx_silver_ohlcv (source=kite_api).")

    return continuous


def fetch_missing_range(
    commodity: str,
    from_date: dt.date,
    to_date: dt.date,
    log: Callable[[str], None] | None = None,
) -> int:
    """
    Fetches and upserts just the [from_date, to_date] gap for `commodity` --
    the narrow-window counterpart to backfill_history()'s years-back pull.
    This is what the UI's startup freshness check calls when the latest
    stored date is behind today, rather than re-pulling years of history
    every time.

    Falls back to data.mcx_proxy's COMEX+USDINR proxy (source='proxy') if
    Kite isn't configured or the call fails for any reason (expired token,
    no market data yet, etc.) -- so a missing/stale KITE_ACCESS_TOKEN
    degrades to the existing proxy path instead of blocking the whole UI
    run. Returns the number of rows upserted (0 if nothing new was found on
    either path).

    log: optional callable(str) for progress messages, e.g. wired to a
    Streamlit status box by the caller -- defaults to print().
    """
    log = log or print

    if from_date > to_date:
        return 0

    if settings.kite_configured:
        try:
            multi_contract_df = _fetch_all_contracts_in_window(commodity, from_date, to_date, log=log)
            if not multi_contract_df.empty:
                n = upsert_ohlcv(multi_contract_df, source="kite_api")
                log(f"Upserted {n} row(s) into mcx_silver_ohlcv (source=kite_api) "
                    f"for {from_date} to {to_date}.")
                return n
            log(f"Kite returned no rows for {from_date} to {to_date} "
                f"(holiday/weekend-only gap, or too recent) -- falling back to proxy.")
        except Exception as e:
            log(f"Kite fetch failed ({e}) -- falling back to the COMEX+USDINR proxy.")
    else:
        log("Kite Connect not configured (missing KITE_API_KEY/SECRET/ACCESS_TOKEN) "
            "-- using the COMEX+USDINR proxy instead.")

    from data.mcx_proxy import fetch_mcx_silver_proxy

    proxy = fetch_mcx_silver_proxy(start=from_date.isoformat(), end=to_date.isoformat())
    if proxy.empty:
        log(f"Proxy fetch also returned nothing for {from_date} to {to_date}.")
        return 0

    proxy_ohlcv = proxy[["mcx_proxy_open", "mcx_proxy_high", "mcx_proxy_low", "mcx_proxy_close"]].rename(
        columns={
            "mcx_proxy_open": "open", "mcx_proxy_high": "high",
            "mcx_proxy_low": "low", "mcx_proxy_close": "close",
        }
    )
    proxy_ohlcv["contract"] = f"{commodity}_PROXY"
    n = upsert_ohlcv(proxy_ohlcv, source="proxy")
    log(f"Upserted {n} row(s) into mcx_silver_ohlcv (source=proxy) for {from_date} to {to_date}.")
    return n


def fetch_latest_eod(commodity: str) -> pd.DataFrame:
    """
    Pulls just the most recent trading day for the current front-month
    contract and upserts it. This is the function a daily scheduled job
    (built in a later phase) will call every evening after MCX close.
    """
    kite = _get_client()
    contracts = list_contracts(commodity, kite=kite)

    today = dt.date.today()
    active = contracts[contracts["expiry"] >= pd.Timestamp(today)]
    if active.empty:
        raise RuntimeError(f"No active (non-expired) {commodity} contracts found.")

    front_month = active.iloc[0]  # nearest expiry = front month
    yesterday = today - dt.timedelta(days=1)

    hist = fetch_contract_history(front_month["instrument_token"], yesterday, today)
    if hist.empty:
        print(f"No new EOD data for {front_month['tradingsymbol']} yet "
              f"(market may not have closed, or it's a holiday).")
        return hist

    hist["contract"] = f"{commodity}_{front_month['tradingsymbol']}"
    n = upsert_ohlcv(hist, source="kite_api")
    print(f"Upserted {n} row(s) for {front_month['tradingsymbol']} "
          f"({hist.index.max().date()}).")
    return hist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commodity", default=settings.mcx_commodity)
    parser.add_argument("--mode", choices=["backfill", "latest"], default="latest")
    parser.add_argument("--years", type=int, default=2, help="Only used with --mode backfill")
    args = parser.parse_args()

    if args.mode == "backfill":
        backfill_history(args.commodity, years_back=args.years)
    else:
        fetch_latest_eod(args.commodity)
