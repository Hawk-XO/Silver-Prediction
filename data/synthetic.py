"""
data/synthetic.py

Generates a plausible synthetic MCX Silver multi-contract dataset plus
matching COMEX/USDINR/DXY series, purely so the ingestion pipeline
(loader -> roll-adjustment -> global fetch -> merge) can be exercised
end-to-end before real data is available.

This is NOT for modeling — synthetic random-walk data has no genuine
predictive structure. Its only purpose is to prove the pipeline runs
without errors and produces the expected schema. Replace with real data
before Phase 4 (modeling) onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42


def _random_walk(n: int, start: float, daily_vol: float, drift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=daily_vol, size=n)
    log_prices = np.log(start) + np.cumsum(returns)
    return np.exp(log_prices)


def generate_synthetic_mcx_contracts(
    start_date: str = "2024-01-01",
    n_days: int = 250,
    start_price: float = 75000.0,  # roughly INR/kg ballpark, illustrative only
) -> pd.DataFrame:
    """
    Generate a long-format multi-contract MCX Silver dataset with three
    overlapping monthly contracts rolling every ~21 trading days, mimicking
    the shape build_continuous_series() expects.
    """
    dates = pd.bdate_range(start=start_date, periods=n_days, tz="Asia/Kolkata")
    close = _random_walk(n_days, start_price, daily_vol=0.012, drift=0.0002, seed=RNG_SEED)

    rows = []
    contract_names = ["SILVER_M1", "SILVER_M2", "SILVER_M3"]
    roll_period = 21  # trading days between rolls, illustrative

    for i, date in enumerate(dates):
        # Determine which contract is "front month" this block.
        block = (i // roll_period) % len(contract_names)
        front = contract_names[block]
        nxt = contract_names[(block + 1) % len(contract_names)]

        base_close = close[i]
        daily_range = base_close * 0.008

        for j, contract in enumerate([front, nxt]):
            # Next-month contract trades at a small illustrative premium (contango).
            premium = 1.0 if contract == front else 1.003
            c = base_close * premium
            o = c * (1 + np.random.default_rng(RNG_SEED + i + j).normal(0, 0.002))
            h = max(o, c) + abs(np.random.default_rng(RNG_SEED + i + j + 100).normal(0, daily_range * 0.3))
            l = min(o, c) - abs(np.random.default_rng(RNG_SEED + i + j + 200).normal(0, daily_range * 0.3))
            vol = int(np.random.default_rng(RNG_SEED + i + j + 300).integers(5000, 50000))
            if contract != front:
                vol = int(vol * 0.2)  # next-month contract trades much thinner
            oi = int(vol * np.random.default_rng(RNG_SEED + i + j + 400).uniform(3, 8))

            rows.append(
                {
                    "date": date,
                    "contract": contract,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "volume": vol,
                    "open_interest": oi,
                }
            )

    df = pd.DataFrame(rows).set_index("date")
    return df


def generate_synthetic_global_factor(
    start_date: str = "2024-01-01",
    n_days: int = 250,
    start_price: float = 24.0,
    daily_vol: float = 0.014,
    seed_offset: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV series matching the schema global_factors fetchers return."""
    dates = pd.bdate_range(start=start_date, periods=n_days, tz="Asia/Kolkata")
    close = _random_walk(n_days, start_price, daily_vol=daily_vol, drift=0.0001, seed=RNG_SEED + seed_offset)
    df = pd.DataFrame(index=dates)
    df["close"] = close
    df["open"] = df["close"].shift(1).fillna(start_price) * (1 + np.random.default_rng(seed_offset).normal(0, 0.002, n_days))
    df["high"] = df[["open", "close"]].max(axis=1) * 1.004
    df["low"] = df[["open", "close"]].min(axis=1) * 0.996
    df["volume"] = np.random.default_rng(seed_offset + 1).integers(10000, 100000, n_days)
    return df[["open", "high", "low", "close", "volume"]]


def generate_full_synthetic_dataset(start_date: str = "2024-01-01", n_days: int = 250) -> dict:
    """
    Build a complete synthetic dataset covering every stage of the ingestion
    pipeline: raw multi-contract MCX data, plus synthetic COMEX/USDINR/DXY,
    matching the shapes the real loaders/fetchers would produce.

    Returns
    -------
    dict with keys: 'mcx_multi_contract', 'comex_silver', 'usdinr', 'dxy'
    """
    return {
        "mcx_multi_contract": generate_synthetic_mcx_contracts(start_date, n_days),
        "comex_silver": generate_synthetic_global_factor(start_date, n_days, start_price=24.0, daily_vol=0.014, seed_offset=1),
        "usdinr": generate_synthetic_global_factor(start_date, n_days, start_price=83.0, daily_vol=0.003, seed_offset=2),
        "dxy": generate_synthetic_global_factor(start_date, n_days, start_price=104.0, daily_vol=0.004, seed_offset=3),
    }
