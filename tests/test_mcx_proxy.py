"""
tests/test_mcx_proxy.py

Tests the MCX Silver proxy calculation logic (COMEX * USDINR * oz-to-kg
conversion) using mocked fetch functions, since this sandbox has no network
access to Yahoo Finance. The math itself — unit conversion, premium
scaling, calibration — is fully verifiable without a live network call;
only the actual HTTP fetch (fetch_comex_silver/fetch_usdinr, already used
successfully in Phase 2's global_factors module) needs a real network to
exercise, which you should do in your own environment before trusting
live output.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.mcx_proxy import fetch_mcx_silver_proxy, calibrate_premium, TROY_OZ_TO_KG


@pytest.fixture
def mock_comex_usdinr():
    dates = pd.bdate_range("2024-06-01", periods=10, tz="Asia/Kolkata")
    comex = pd.DataFrame({
        "open": np.full(10, 24.0),
        "high": np.full(10, 24.5),
        "low": np.full(10, 23.5),
        "close": np.linspace(24.0, 25.0, 10),  # gently rising COMEX silver, USD/oz
        "volume": np.full(10, 50000),
    }, index=dates)

    usdinr = pd.DataFrame({
        "open": np.full(10, 83.0),
        "high": np.full(10, 83.2),
        "low": np.full(10, 82.8),
        "close": np.full(10, 83.0),  # flat USDINR for a clean isolated test of COMEX scaling
        "volume": np.full(10, 0),
    }, index=dates)

    return comex, usdinr


def test_proxy_calculation_matches_manual_formula(mock_comex_usdinr):
    comex, usdinr = mock_comex_usdinr
    with patch("data.mcx_proxy.fetch_comex_silver", return_value=comex), \
         patch("data.mcx_proxy.fetch_usdinr", return_value=usdinr):
        proxy = fetch_mcx_silver_proxy(premium_pct=0.0)

    # Manually compute expected proxy close for first row and verify exactly.
    expected_close_row0 = comex["close"].iloc[0] * usdinr["close"].iloc[0] * TROY_OZ_TO_KG
    assert np.isclose(proxy["mcx_proxy_close"].iloc[0], expected_close_row0)


def test_premium_scales_output_linearly(mock_comex_usdinr):
    comex, usdinr = mock_comex_usdinr
    with patch("data.mcx_proxy.fetch_comex_silver", return_value=comex), \
         patch("data.mcx_proxy.fetch_usdinr", return_value=usdinr):
        proxy_no_premium = fetch_mcx_silver_proxy(premium_pct=0.0)
        proxy_with_premium = fetch_mcx_silver_proxy(premium_pct=0.05)

    ratio = proxy_with_premium["mcx_proxy_close"] / proxy_no_premium["mcx_proxy_close"]
    assert np.allclose(ratio, 1.05)


def test_calibrate_premium_recovers_known_offset(mock_comex_usdinr):
    comex, usdinr = mock_comex_usdinr
    with patch("data.mcx_proxy.fetch_comex_silver", return_value=comex), \
         patch("data.mcx_proxy.fetch_usdinr", return_value=usdinr):
        proxy = fetch_mcx_silver_proxy(premium_pct=0.0)

    # Simulate "real" MCX prices as the proxy scaled up by exactly 3% —
    # calibrate_premium should recover ~0.03.
    real_mcx = proxy["mcx_proxy_close"] * 1.03
    real_mcx.index = real_mcx.index.normalize()

    recovered = calibrate_premium(proxy, real_mcx)
    assert np.isclose(recovered, 0.03, atol=1e-6)


def test_calibrate_premium_raises_on_no_overlap(mock_comex_usdinr):
    comex, usdinr = mock_comex_usdinr
    with patch("data.mcx_proxy.fetch_comex_silver", return_value=comex), \
         patch("data.mcx_proxy.fetch_usdinr", return_value=usdinr):
        proxy = fetch_mcx_silver_proxy(premium_pct=0.0)

    far_future_index = pd.bdate_range("2099-01-01", periods=5)
    fake_real = pd.Series([100.0] * 5, index=far_future_index)

    with pytest.raises(ValueError, match="no overlapping dates"):
        calibrate_premium(proxy, fake_real)
