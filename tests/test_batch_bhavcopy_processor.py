"""
tests/test_batch_bhavcopy_processor.py

Tests the batch processor against realistic simulated MCX bhavcopy files
(multi-commodity, multi-expiry, real-world column names like TradDt/
OpenPrice/ClosePrice), proving it correctly:
  - filters down to only the target commodity
  - exact-matches SILVER without pulling in SILVERM/SILVERMIC
  - builds distinct per-expiry contract identifiers
  - feeds cleanly into build_continuous_series()
"""

import os

import numpy as np
import pandas as pd
import pytest

from data.batch_bhavcopy_processor import process_bhavcopy_folder
from data.contract_roll import build_continuous_series


@pytest.fixture(scope="module")
def fake_bhavcopy_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fake_bhavcopy")
    commodities = ["GOLD", "GOLDM", "SILVER", "SILVERM", "SILVERMIC", "CRUDEOIL"]
    expiries = ["30-Jul-2024", "29-Aug-2024"]
    dates = pd.bdate_range("2024-06-01", periods=10)

    base_prices = {
        "GOLD": 72000, "GOLDM": 72000, "SILVER": 91000,
        "SILVERM": 91000, "SILVERMIC": 91000, "CRUDEOIL": 6500,
    }

    for i, date in enumerate(dates):
        rows = []
        for comm in commodities:
            comm_expiries = expiries if comm == "SILVER" else [expiries[0]]
            for exp in comm_expiries:
                premium = 1.002 if exp == expiries[1] else 1.0
                price = base_prices[comm] * premium * (1 + np.random.default_rng(i).normal(0, 0.01))
                rows.append({
                    "TradDt": date.strftime("%d-%b-%Y"),
                    "InstrumentName": comm,
                    "ExpiryDate": exp,
                    "OpenPrice": round(price * 0.998, 2),
                    "HighPrice": round(price * 1.005, 2),
                    "LowPrice": round(price * 0.995, 2),
                    "ClosePrice": round(price, 2),
                    "TotalTradedQty": int(np.random.default_rng(i + 50).integers(1000, 50000)),
                    "OpenInterest": int(np.random.default_rng(i + 99).integers(5000, 200000)),
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(str(folder), f"bhavcopy_{date.strftime('%Y%m%d')}.csv"), index=False
        )
    return str(folder)


def test_exact_match_excludes_mini_contracts(fake_bhavcopy_folder):
    df = process_bhavcopy_folder(fake_bhavcopy_folder, commodity="SILVER", exact_match=True)
    contracts = df["contract"].unique()
    assert all("SILVERM" not in c and "SILVERMIC" not in c for c in contracts), (
        "exact_match=True must not pull in SILVERM/SILVERMIC as substring matches of SILVER"
    )
    assert len(contracts) == 2  # two expiry months of pure SILVER


def test_substring_match_includes_mini_contracts(fake_bhavcopy_folder):
    df = process_bhavcopy_folder(fake_bhavcopy_folder, commodity="SILVER", exact_match=False)
    contracts = df["contract"].unique()
    assert any("SILVERM" in c for c in contracts)
    assert any("SILVERMIC" in c for c in contracts)


def test_contract_identifier_includes_expiry(fake_bhavcopy_folder):
    df = process_bhavcopy_folder(fake_bhavcopy_folder, commodity="SILVER", exact_match=True)
    for contract in df["contract"].unique():
        assert "_" in contract, "contract id should combine commodity + expiry, e.g. SILVER_30-Jul-2024"


def test_output_feeds_into_continuous_series(fake_bhavcopy_folder):
    df = process_bhavcopy_folder(fake_bhavcopy_folder, commodity="SILVER", exact_match=True)
    continuous = build_continuous_series(df)
    assert not continuous.empty
    assert continuous[["open", "high", "low", "close"]].isna().sum().sum() == 0


def test_missing_commodity_raises_helpful_error(fake_bhavcopy_folder):
    with pytest.raises(ValueError, match="zero"):
        process_bhavcopy_folder(fake_bhavcopy_folder, commodity="PLATINUM", exact_match=True)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError, match="no files matching"):
        process_bhavcopy_folder(str(tmp_path), commodity="SILVER")
