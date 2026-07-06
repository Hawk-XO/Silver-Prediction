"""
tests/test_kite_paper_broker.py

Phase 7 tests for broker/kite_paper_broker.py:
  1. A simple BUY then SELL of the same quantity at a higher price realizes
     the correct profit and flattens the position.
  2. Partial closes keep the average entry price of the remaining position
     unchanged.
  3. A fill that exceeds the existing opposite position correctly flips
     the position (closes old + opens new at the fill price).
  4. get_positions()/get_pnl() reflect unrealised P&L via
     update_market_price() even with no new orders.
  5. MARKET orders without a known price raise a clear error; LIMIT-style
     orders with an explicit price don't need update_market_price() first.
  6. Order log records exist with expected fields.
"""

from __future__ import annotations

import pytest

from broker.kite_paper_broker import PaperKiteBroker


def test_buy_then_sell_realizes_correct_profit():
    broker = PaperKiteBroker(initial_cash=1_000_000)
    broker.place_order("SILVERFUT", "BUY", 10, price=75000)
    broker.place_order("SILVERFUT", "SELL", 10, price=75500)

    positions = broker.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos["quantity"] == 0
    assert pos["realised"] == pytest.approx(10 * (75500 - 75000))
    assert pos["unrealised"] == pytest.approx(0.0)

    pnl = broker.get_pnl()
    assert pnl["realised"] == pytest.approx(5000)
    assert pnl["total"] == pytest.approx(5000)


def test_short_then_cover_realizes_correct_profit():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "SELL", 5, price=75000)
    broker.place_order("SILVERFUT", "BUY", 5, price=74000)  # price dropped -> short profits

    pos = broker.get_positions()[0]
    assert pos["quantity"] == 0
    assert pos["realised"] == pytest.approx(5 * (75000 - 74000))


def test_partial_close_keeps_average_price_of_remainder():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 10, price=75000)
    broker.place_order("SILVERFUT", "SELL", 4, price=76000)  # partial close, 4 of 10

    pos = broker.get_positions()[0]
    assert pos["quantity"] == 6
    assert pos["average_price"] == pytest.approx(75000)  # unchanged for the remaining 6
    assert pos["realised"] == pytest.approx(4 * (76000 - 75000))


def test_fill_larger_than_position_flips_direction():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 5, price=75000)
    broker.place_order("SILVERFUT", "SELL", 12, price=75200)  # closes 5 long, opens 7 short

    pos = broker.get_positions()[0]
    assert pos["quantity"] == -7
    assert pos["average_price"] == pytest.approx(75200)  # new short leg's entry price
    assert pos["realised"] == pytest.approx(5 * (75200 - 75000))  # profit on the closed long leg


def test_averaging_up_on_same_direction_fills():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 10, price=100.0)
    broker.place_order("SILVERFUT", "BUY", 10, price=110.0)

    pos = broker.get_positions()[0]
    assert pos["quantity"] == 20
    assert pos["average_price"] == pytest.approx(105.0)  # simple average of two equal-size legs


def test_unrealised_pnl_updates_with_market_price_and_no_new_orders():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 10, price=75000)
    broker.update_market_price("SILVERFUT", 75300)  # mark-to-market, no order placed

    pos = broker.get_positions()[0]
    assert pos["quantity"] == 10
    assert pos["realised"] == pytest.approx(0.0)
    assert pos["unrealised"] == pytest.approx(10 * (75300 - 75000))
    assert broker.get_pnl()["total"] == pytest.approx(3000)


def test_market_order_without_price_raises_clear_error():
    broker = PaperKiteBroker()
    with pytest.raises(RuntimeError, match="update_market_price"):
        broker.place_order("SILVERFUT", "BUY", 10)  # no price, no prior update_market_price call


def test_market_order_uses_last_updated_price():
    broker = PaperKiteBroker()
    broker.update_market_price("SILVERFUT", 74800)
    order_id = broker.place_order("SILVERFUT", "BUY", 3)  # MARKET, should use 74800
    pos = broker.get_positions()[0]
    assert pos["average_price"] == pytest.approx(74800)
    assert order_id.startswith("PAPER")


def test_invalid_transaction_type_rejected():
    broker = PaperKiteBroker()
    with pytest.raises(ValueError):
        broker.place_order("SILVERFUT", "HOLD", 10, price=75000)


def test_orders_are_logged():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 10, price=75000, tag="test-entry")
    broker.place_order("SILVERFUT", "SELL", 10, price=75500, tag="test-exit")

    orders = broker.get_orders()
    assert len(orders) == 2
    assert orders[0]["transaction_type"] == "BUY"
    assert orders[0]["tag"] == "test-entry"
    assert orders[1]["status"] == "COMPLETE"
    # Order ids should be unique and monotonically distinguishable.
    assert orders[0]["order_id"] != orders[1]["order_id"]


def test_multiple_symbols_tracked_independently():
    broker = PaperKiteBroker()
    broker.place_order("SILVERFUT", "BUY", 10, price=75000)
    broker.place_order("GOLDFUT", "SELL", 2, price=60000)

    positions = {p["tradingsymbol"]: p for p in broker.get_positions()}
    assert positions["SILVERFUT"]["quantity"] == 10
    assert positions["GOLDFUT"]["quantity"] == -2
