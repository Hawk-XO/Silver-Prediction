"""
data/build_merged_history.py

Fills the "before Kite data existed" gap in your MCX Silver history using
the COMEX+USDINR proxy (data/mcx_proxy.py), calibrated against the real
prices Kite already gave us so the proxy's price *level* lines up with
reality rather than just approximating it.

Why this exists: Kite Connect's historical API can only return data for
contracts that are still in MCX's live instrument list — already-expired,
delisted contracts (and therefore anything before roughly when your oldest
currently-listed contract started trading) aren't reachable that way at
all. The proxy has no such limitation (COMEX Silver history via yfinance
goes back decades), so it's the practical way to extend history earlier.

What it does
------------
1. Loads the real Kite rows already in MySQL (source='kite_api') to find
   (a) the earliest real date we have, and (b) reference close prices to
   calibrate against.
2. Builds the raw (uncalibrated) COMEX+USDINR proxy for the full deep-history
   window, then calibrates it using calibrate_premium() against the overlap
   period where both proxy and real Kite data exist.
3. Upserts ONLY the pre-Kite-history portion (dates strictly before the
   earliest real Kite date) into MySQL as contract='{COMMODITY}_PROXY',
   source='proxy' — we never overwrite real data with the approximation.

After this runs, build_continuous_series() (from contract_roll.py) will
naturally stitch the proxy segment and the real Kite segments together via
its usual ratio-adjustment at the boundary, since both live in the same
long-format schema.

Usage
-----
    python -m data.build_merged_history --commodity SILVERMIC --deep-start 2015-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.settings import settings
from data.db import get_engine, load_ohlcv, upsert_ohlcv
from data.mcx_proxy import fetch_mcx_silver_proxy, calibrate_premium


def build_and_store(commodity: str, deep_start: str) -> pd.DataFrame:
    engine = get_engine()

    real = load_ohlcv(source="kite_api", engine=engine)
    if real.empty:
        raise RuntimeError(
            "No source='kite_api' rows found in MySQL yet — run "
            "`python -m data.kite_fetcher --mode backfill` first, since "
            "calibration needs at least some real MCX prices to anchor to."
        )
    earliest_real_date = real.index.min()
    print(f"Earliest real (Kite) data on file: {earliest_real_date.date()}. "
          f"Building proxy history before that date.")

    # Build the raw, uncalibrated proxy over the FULL window (deep_start to
    # today) — we need the overlap with real data to calibrate, even though
    # we'll only ultimately store the pre-real-data portion.
    raw_proxy = fetch_mcx_silver_proxy(start=deep_start, end=None, premium_pct=0.0)
    if raw_proxy.empty:
        raise RuntimeError("fetch_mcx_silver_proxy returned no data — check network/yfinance access.")

    premium_pct = calibrate_premium(raw_proxy, real["close"])
    print(f"Calibrated premium: {premium_pct:+.4%} vs. raw import-parity price.")

    calibrated_proxy = fetch_mcx_silver_proxy(start=deep_start, end=None, premium_pct=premium_pct)

    # Only keep the gap BEFORE real data starts — never let the proxy
    # overwrite or compete with real prices for dates we actually have.
    calibrated_proxy.index = calibrated_proxy.index.tz_localize(None) if calibrated_proxy.index.tz else calibrated_proxy.index
    earliest_real_naive = earliest_real_date.tz_localize(None) if earliest_real_date.tz else earliest_real_date
    gap = calibrated_proxy[calibrated_proxy.index < earliest_real_naive].copy()

    if gap.empty:
        print("No gap to fill — proxy history doesn't extend earlier than real data "
              "(or deep_start is later than the earliest real date). Nothing stored.")
        return gap

    out = gap.rename(columns={
        "mcx_proxy_open": "open", "mcx_proxy_high": "high",
        "mcx_proxy_low": "low", "mcx_proxy_close": "close",
    })[["open", "high", "low", "close"]]
    out["contract"] = f"{commodity}_PROXY"
    out["volume"] = None
    out["open_interest"] = None

    n = upsert_ohlcv(out, source="proxy", engine=engine)
    print(f"Upserted {n} calibrated proxy rows (source=proxy) covering "
          f"{out.index.min().date()} to {out.index.max().date()}.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commodity", default=settings.mcx_commodity)
    parser.add_argument("--deep-start", default="2015-01-01",
                         help="How far back to attempt building proxy history (default: 2015-01-01)")
    args = parser.parse_args()
    build_and_store(args.commodity, args.deep_start)
