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

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import settings
from data.contract_roll import build_continuous_series
from data.db import upsert_ohlcv, get_engine

MCX_EXCHANGE = "MCX"


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
    instruments = pd.DataFrame(kite.instruments(MCX_EXCHANGE))

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
    raw = kite.historical_data(
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


def backfill_history(commodity: str, years_back: int = 2) -> pd.DataFrame:
    """
    Pulls `years_back` years of daily history across every contract expiry
    for `commodity`, stitches them into a continuous series, and upserts the
    result into MySQL (source='kite_api'). Returns the continuous DataFrame.
    """
    kite = _get_client()
    contracts = list_contracts(commodity, kite=kite)

    today = dt.date.today()
    window_start = today - dt.timedelta(days=365 * years_back)

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
        contract_end = min(today, expiry)

        print(f"Fetching {row['tradingsymbol']} (expiry {expiry}) "
              f"from {contract_start} to {contract_end} ...")
        hist = fetch_contract_history(row["instrument_token"], contract_start, contract_end, kite=kite)
        if hist.empty:
            print(f"  -> no data returned (may be too old for Kite's history retention).")
            continue

        hist["contract"] = f"{commodity}_{row['tradingsymbol']}"
        all_contract_frames.append(hist)

    if not all_contract_frames:
        raise RuntimeError(
            f"No historical data returned for any {commodity} contract in the "
            f"last {years_back} years — check KITE_ACCESS_TOKEN is fresh (it "
            f"expires daily) and that {commodity} is the right commodity name."
        )

    multi_contract_df = pd.concat(all_contract_frames)
    continuous = build_continuous_series(multi_contract_df)

    engine = get_engine()
    n = upsert_ohlcv(multi_contract_df, source="kite_api", engine=engine)
    print(f"Upserted {n} raw per-contract rows into mcx_silver_ohlcv (source=kite_api).")

    return continuous


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
