"""
broker/kite_paper_broker.py

Paper-trading wrapper matching the shape of Zerodha Kite Connect's order
API — `place_order`, `positions` (exposed here as `get_positions`), and a
convenience `get_pnl` on top. The goal: strategy/execution code is written
once against this interface, and swapping in a real `kiteconnect.KiteConnect`
client later (or a sandbox/paper account through Kite) shouldn't require
touching the strategy logic — only the broker object passed in changes.

What's simulated vs. real Kite Connect
-----------------------------------------
Real Kite Connect fills orders against live market prices; there's no
concept of "feed me today's price" in its API. A PAPER broker has no real
market to fill against, so `PaperKiteBroker` needs one extra thing real
Kite doesn't: `update_market_price(tradingsymbol, price)`, called once per
bar/tick from your backtest or live-feed loop before `place_order()` for
that symbol. Everything else (`place_order`, `get_positions`, `get_pnl`)
is shaped to match what a real broker wrapper would expose.

`get_pnl()` is a convenience addition, not a literal Kite Connect method —
real Kite Connect only exposes PnL via the `pnl`/`unrealised`/`realised`
fields inside each position dict returned by `positions()`. A future real
wrapper can implement `get_pnl()` by just summing those fields, so
strategy code calling `broker.get_pnl()` doesn't need to change either way.

Swapping in the real thing later (sketch, not implemented — needs a live
Kite Connect API key/secret and a running access-token flow):

    from kiteconnect import KiteConnect

    class LiveKiteBroker:
        def __init__(self, api_key, access_token):
            self._kite = KiteConnect(api_key=api_key)
            self._kite.set_access_token(access_token)

        def place_order(self, **kwargs):
            return self._kite.place_order(variety="regular", **kwargs)

        def get_positions(self):
            return self._kite.positions()["net"]

        def get_pnl(self):
            positions = self.get_positions()
            realised = sum(p["realised"] for p in positions)
            unrealised = sum(p["unrealised"] for p in positions)
            return {"realised": realised, "unrealised": unrealised, "total": realised + unrealised}

    # Strategy code stays identical either way:
    #   broker.place_order(exchange="MCX", tradingsymbol="SILVER25AUGFUT", ...)
    #   broker.get_positions()
    #   broker.get_pnl()
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


VALID_TRANSACTION_TYPES = {"BUY", "SELL"}


@dataclass
class _Position:
    quantity: int = 0            # signed: positive = long, negative = short
    average_price: float = 0.0   # average entry price of the OPEN position
    realised_pnl: float = 0.0
    last_price: float = 0.0


class PaperKiteBroker:
    """
    In-memory paper-trading broker. No network calls, no real orders — just
    enough position/PnL accounting to test a strategy's execution logic
    before pointing it at a real (sandbox or live) Kite Connect account.
    """

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.cash = initial_cash
        self._positions: dict[str, _Position] = {}
        self._market_prices: dict[str, float] = {}
        self._orders: list[dict] = []
        self._order_id_counter = itertools.count(1)

    # -- market data feed (paper-trading-only addition, see module docstring) --

    def update_market_price(self, tradingsymbol: str, price: float) -> None:
        """Mark a symbol's current price — call this once per bar/tick
        before place_order() for MARKET orders, and periodically anyway to
        keep unrealised PnL current on open positions."""
        self._market_prices[tradingsymbol] = price
        if tradingsymbol in self._positions:
            self._positions[tradingsymbol].last_price = price

    # -- Kite Connect-shaped API --

    def place_order(
        self,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        exchange: str = "MCX",
        product: str = "NRML",
        order_type: str = "MARKET",
        price: float | None = None,
        variety: str = "regular",
        tag: str | None = None,
        **kwargs,
    ) -> str:
        """
        Signature mirrors kite.place_order(...). Paper orders fill
        IMMEDIATELY and COMPLETELY (no partial fills, no rejections) at:
          - `price`, if given (LIMIT-style), else
          - the last price set via update_market_price(tradingsymbol, ...).

        Returns an order_id string, like the real API.
        """
        if transaction_type not in VALID_TRANSACTION_TYPES:
            raise ValueError(f"transaction_type must be one of {VALID_TRANSACTION_TYPES}, got {transaction_type!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        fill_price = price if price is not None else self._market_prices.get(tradingsymbol)
        if fill_price is None:
            raise RuntimeError(
                f"No price available to fill MARKET order for {tradingsymbol!r} — "
                f"call update_market_price() first, or pass price= for a LIMIT-style fill."
            )

        signed_qty = quantity if transaction_type == "BUY" else -quantity
        self._apply_fill(tradingsymbol, signed_qty, fill_price)

        order_id = f"PAPER{next(self._order_id_counter):08d}"
        self._orders.append(
            {
                "order_id": order_id,
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "price": fill_price,
                "product": product,
                "order_type": order_type,
                "variety": variety,
                "status": "COMPLETE",
                "tag": tag,
            }
        )
        return order_id

    def get_positions(self) -> list[dict]:
        """Shaped like kite.positions()["net"]: one dict per symbol with an
        open or previously-open position."""
        result = []
        for symbol, pos in self._positions.items():
            unrealised = pos.quantity * (pos.last_price - pos.average_price)
            result.append(
                {
                    "tradingsymbol": symbol,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "last_price": pos.last_price,
                    "realised": pos.realised_pnl,
                    "unrealised": unrealised,
                    "pnl": pos.realised_pnl + unrealised,
                }
            )
        return result

    def get_pnl(self) -> dict:
        """Convenience aggregate across all symbols (see module docstring —
        not a literal Kite Connect method, but trivially implementable
        against one)."""
        positions = self.get_positions()
        realised = sum(p["realised"] for p in positions)
        unrealised = sum(p["unrealised"] for p in positions)
        return {"realised": realised, "unrealised": unrealised, "total": realised + unrealised}

    def get_orders(self) -> list[dict]:
        """Shaped like kite.orders() — full order history, most recent last."""
        return list(self._orders)

    # -- internal position accounting --

    def _apply_fill(self, symbol: str, signed_qty: int, fill_price: float) -> None:
        pos = self._positions.setdefault(symbol, _Position())
        old_qty, old_avg = pos.quantity, pos.average_price

        same_direction = old_qty == 0 or (old_qty > 0) == (signed_qty > 0)

        if same_direction:
            new_qty = old_qty + signed_qty
            if new_qty != 0:
                pos.average_price = (old_qty * old_avg + signed_qty * fill_price) / new_qty
            pos.quantity = new_qty
        else:
            # Opposite direction: this fill closes some or all of the
            # existing position (and may flip it to the other side).
            closing_qty = min(abs(signed_qty), abs(old_qty))
            # Profit per unit closed = (fill - entry) if old position was
            # long, or (entry - fill) if old position was short.
            direction_sign = 1 if old_qty > 0 else -1
            pos.realised_pnl += closing_qty * (fill_price - old_avg) * direction_sign

            new_qty = old_qty + signed_qty
            if abs(signed_qty) > abs(old_qty):
                # Flipped: leftover quantity opens a fresh position at fill_price.
                pos.quantity = new_qty
                pos.average_price = fill_price
            elif new_qty == 0:
                pos.quantity = 0
                pos.average_price = 0.0
            else:
                # Partial close: quantity shrinks, average price of the
                # remaining (still-open) position is unchanged.
                pos.quantity = new_qty

        pos.last_price = fill_price
