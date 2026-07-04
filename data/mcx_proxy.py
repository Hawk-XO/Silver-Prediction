"""
data/mcx_proxy.py

Fully automated alternative to manual MCX bhavcopy downloads. Instead of
scraping/downloading MCX's own site (which disallows bot access and
requires manual steps either way), this module reconstructs an MCX
Silver-equivalent price series from two internationally-sourced, freely
and automatically fetchable series: COMEX Silver futures and USD/INR.

Why this works: MCX Silver is priced against the international silver
market. India imports the vast majority of its silver, so MCX Silver
(INR/kg) tracks:

    mcx_proxy = comex_silver_usd_per_oz * usdinr * OZ_TO_KG + local_premium

`local_premium` captures import duty, GST, and local demand/supply premium
(India often trades at a premium or discount to the pure import-parity
price — this varies over time and isn't something we can compute without
real MCX data). We do NOT try to fabricate this precisely; instead we:

  1. Compute the raw parity price (comex * usdinr * oz_to_kg) with zero
     assumed premium as the baseline proxy.
  2. Expose a `calibrate_premium()` function that, IF you later obtain even
     a small sample of real MCX prices (a handful of days is enough), fits
     a premium/scaling correction so the proxy matches real MCX levels.
     Until calibrated, the proxy's absolute price level will be
     approximately right but not exact — its RETURNS (which is what our
     models actually predict, per PROJECT_NOTES.md) are far more reliable
     than its absolute levels, since USDINR and COMEX moves dominate
     day-to-day INR silver price changes regardless of the fixed premium.

This is the practical trade-off: zero manual effort, fully automated,
always fresh — at the cost of the proxy's price *level* needing later
calibration against even a small amount of real MCX data for anything
level-sensitive (e.g. absolute position sizing in INR). Return-based
modeling and signal direction are usable immediately.
"""

from __future__ import annotations

import pandas as pd

from data.global_factors import fetch_comex_silver, fetch_usdinr

TROY_OZ_TO_KG = 32.1507  # 1 kg = 32.1507 troy ounces

DEFAULT_PREMIUM_PCT = 0.0  # additive premium as a fraction of parity price; 0 until calibrated


def fetch_mcx_silver_proxy(
    start: str | None = None,
    end: str | None = None,
    premium_pct: float = DEFAULT_PREMIUM_PCT,
) -> pd.DataFrame:
    """
    Build a fully-automated MCX Silver-equivalent price series from COMEX
    Silver + USD/INR, with zero manual download steps.

    Parameters
    ----------
    start, end : str | None
        Date range, passed through to the underlying yfinance fetchers.
    premium_pct : float
        Additive premium/discount vs. pure import parity, as a fraction
        (e.g. 0.015 for a 1.5% premium). Default 0.0 (uncalibrated). Use
        `calibrate_premium()` once you have real MCX reference prices.

    Returns
    -------
    pd.DataFrame
        Indexed by date (IST), columns:
            comex_close, usdinr_close  (raw inputs, for transparency/debugging)
            mcx_proxy_open, mcx_proxy_high, mcx_proxy_low, mcx_proxy_close
        The proxy OHLC is derived from COMEX OHLC scaled by the SAME day's
        USDINR close (we don't have intraday FX granularity to match COMEX's
        own intraday range, so all four proxy OHLC points use one day's
        USDINR conversion rate — this is a simplification worth knowing
        about if you later need intraday-accurate proxy ranges).
    """
    comex = fetch_comex_silver(start, end)
    usdinr = fetch_usdinr(start, end)

    comex.index = comex.index.normalize()
    usdinr.index = usdinr.index.normalize()

    merged = comex[["open", "high", "low", "close"]].add_prefix("comex_").join(
        usdinr[["close"]].rename(columns={"close": "usdinr_close"}),
        how="inner",  # both are US-hours-driven series on similar calendars; inner join is fine here
    )

    scale = merged["usdinr_close"] * TROY_OZ_TO_KG * (1 + premium_pct)

    for col in ["open", "high", "low", "close"]:
        merged[f"mcx_proxy_{col}"] = merged[f"comex_{col}"] * scale

    return merged


def calibrate_premium(proxy_df: pd.DataFrame, real_mcx_close: pd.Series) -> float:
    """
    Given a small sample of REAL MCX Silver close prices (even 5-10 days is
    enough), compute the average premium/discount vs. the uncalibrated
    proxy, so future proxy calls can be scaled to match real MCX price
    levels.

    Parameters
    ----------
    proxy_df : pd.DataFrame
        Output of fetch_mcx_silver_proxy() with premium_pct=0.0, covering
        at least the same dates as real_mcx_close.
    real_mcx_close : pd.Series
        Real MCX Silver close prices, indexed by date, for the same period.

    Returns
    -------
    float
        Estimated premium_pct to pass into future fetch_mcx_silver_proxy()
        calls so proxy levels align with real MCX prices.
    """
    proxy_dates = proxy_df.index
    real_dates = real_mcx_close.index
    if getattr(proxy_dates, "tz", None) is not None:
        proxy_dates = proxy_dates.tz_localize(None)
    if getattr(real_dates, "tz", None) is not None:
        real_dates = real_dates.tz_localize(None)

    proxy_normalized = proxy_df[["mcx_proxy_close"]].copy()
    proxy_normalized.index = proxy_dates.normalize()
    real_normalized = real_mcx_close.rename("real_close").copy()
    real_normalized.index = real_dates.normalize()

    aligned = proxy_normalized.join(real_normalized, how="inner")
    if aligned.empty:
        raise ValueError(
            "calibrate_premium: no overlapping dates between proxy_df and "
            "real_mcx_close — check the date ranges/index alignment."
        )
    ratio = aligned["real_close"] / aligned["mcx_proxy_close"]
    premium_pct = float(ratio.mean() - 1.0)
    return premium_pct
