"""
backtest/vectorbt_backtest.py

Wires signals/signal_engine.py output (Phase 6) into a vectorbt Portfolio
simulation with approximate MCX transaction costs, slippage, and margin
utilization, then reports Sharpe / max drawdown / win rate / profit factor
against a buy-and-hold baseline.

MCX cost/margin approximations (documented, not exact)
---------------------------------------------------------
Real MCX Silver costs (brokerage + exchange transaction charges + GST +
stamp duty + SEBI turnover fees) vary by broker and change over time —
the `BacktestConfig` defaults below are a round-number approximation for a
retail discount broker, not pulled from a live fee schedule. Update them
with your actual broker's current rates before trusting the P&L numbers.

MCX Silver's exchange-mandated initial margin (SPAN + exposure margin) is
typically in the ~10-20% of notional range and is revised periodically by
MCX. Rather than bending vectorbt's position sizing into an artificial
leverage hack, we run the simulation at 100%-of-available-capital sizing
(the position always uses whatever capital the portfolio has) and
separately report the IMPLIED margin utilization at `margin_pct` — how
much of your capital would actually need to be posted for a position of
that notional size. This keeps the P&L simulation simple/correct while
still surfacing the margin dimension.

HOLD = flat (documented design choice)
------------------------------------------
Whether a HOLD (from low confidence, a high-vol regime, OR a cooldown
block) should mean "go flat" or "keep the previous position open" wasn't
pinned down in Phase 6. We treat every HOLD as flat here — the more
conservative reading ("not confident enough to carry directional risk
today"), and consistent with the Phase 5 naive-backtest convention. If you
want cooldown-blocked days to silently continue holding the prior trade
instead, that's a one-line change in `signals_to_position()` below (you'd
need to track "last real direction" separately from the flattened
`signal` column).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class BacktestConfig:
    fees: float = 0.0003        # ~3 bps: approx brokerage + exchange txn charges per trade value — tune to your broker
    slippage: float = 0.0005    # ~5 bps: approx adverse fill vs. reference close
    margin_pct: float = 0.15    # approx MCX Silver initial margin, % of notional — verify against current MCX circulars
    init_cash: float = 1_000_000.0
    freq: str = "1D"


def signals_to_position(signals: pd.Series) -> pd.Series:
    """Map BUY/SELL/HOLD -> +1/-1/0."""
    return signals.map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0).astype(int)


def _position_to_entries_exits(position: pd.Series):
    """Retained for inspection/testing purposes — shows which days the
    desired position actually changes. The backtest itself (below) uses
    this same change-detection logic but drives vectorbt via
    Portfolio.from_orders + targetpercent sizing rather than from_signals,
    because from_signals + percent-sizing cannot handle a same-bar
    reversal (BUY directly to SELL with no HOLD day between them) — see
    run_strategy_backtest()'s docstring."""
    prev = position.shift(1).fillna(0)
    long_entries = (position == 1) & (prev != 1)
    long_exits = (prev == 1) & (position != 1)
    short_entries = (position == -1) & (prev != -1)
    short_exits = (prev == -1) & (position != -1)
    return long_entries, long_exits, short_entries, short_exits


def run_strategy_backtest(
    signals_df: pd.DataFrame,
    price_col: str = "entry_price",
    signal_col: str = "signal",
    config: BacktestConfig | None = None,
) -> vbt.Portfolio:
    """
    Simulate the signal-engine strategy (Phase 6 output) with vectorbt.

    Implementation note: uses Portfolio.from_orders with target-percent
    sizing, submitting an order ONLY on days the desired position changes
    (all other days get NaN size, which vectorbt treats as "no order").
    This was chosen over the more common Portfolio.from_signals precisely
    because our signal engine can flip directly from BUY to SELL on
    consecutive days (no HOLD day forcing a flat close in between) —
    from_signals combined with percent-of-capital sizing raises
    "SizeType.Percent does not support position reversal using signals" in
    that case; from_orders + targetpercent handles a direct reversal (long
    100% -> short 100%) as a single correctly-costed order, which is both
    more realistic and avoids that restriction.
    """
    cfg = config or BacktestConfig()
    price = signals_df[price_col]
    position = signals_to_position(signals_df[signal_col]).astype(float)
    changed = position != position.shift(1).fillna(0)

    size = pd.Series(np.nan, index=position.index)
    size[changed] = position[changed]

    return vbt.Portfolio.from_orders(
        close=price,
        size=size,
        size_type="targetpercent",
        fees=cfg.fees,
        slippage=cfg.slippage,
        init_cash=cfg.init_cash,
        freq=cfg.freq,
    )


def run_buy_and_hold_backtest(price: pd.Series, config: BacktestConfig | None = None) -> vbt.Portfolio:
    """Baseline: go fully long on day 1, hold through the end of the period."""
    cfg = config or BacktestConfig()
    entries = pd.Series(False, index=price.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=price.index)

    return vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        size=1.0,
        size_type="percent",
        fees=cfg.fees,
        slippage=cfg.slippage,
        init_cash=cfg.init_cash,
        freq=cfg.freq,
    )


def estimate_margin_utilization(pf: vbt.Portfolio, config: BacktestConfig | None = None) -> dict:
    """Approximate margin utilization implied by the simulated position
    sizes (see module docstring)."""
    cfg = config or BacktestConfig()
    notional = pf.asset_value().abs()
    margin_required = notional * cfg.margin_pct
    pct_of_capital = margin_required / cfg.init_cash
    return {
        "avg_margin_pct_of_capital": float(pct_of_capital.mean()),
        "max_margin_pct_of_capital": float(pct_of_capital.max()),
    }


def performance_report(pf: vbt.Portfolio, config: BacktestConfig | None = None) -> dict:
    """Sharpe, max drawdown, win rate, profit factor, total return + the
    margin-utilization estimate, pulled from vectorbt's own stats."""
    stats = pf.stats()
    margin = estimate_margin_utilization(pf, config)
    return {
        "total_return_pct": float(stats.get("Total Return [%]", np.nan)),
        "sharpe_ratio": float(stats.get("Sharpe Ratio", np.nan)),
        "max_drawdown_pct": float(stats.get("Max Drawdown [%]", np.nan)),
        "win_rate_pct": float(stats.get("Win Rate [%]", np.nan)),
        "profit_factor": float(stats.get("Profit Factor", np.nan)),
        "total_trades": int(stats.get("Total Trades", 0)),
        "total_fees_paid": float(stats.get("Total Fees Paid", np.nan)),
        **margin,
    }


def trade_pnl_breakdown(pf: vbt.Portfolio) -> dict:
    """
    Per-trade PnL breakdown, separate from performance_report()'s aggregate
    stats. Answers "are wins small and losses large (or the reverse)?" --
    the question aggregate win-rate/Sharpe can't answer on its own. A
    strategy can have win_rate > 50% and still lose money overall if
    average losers are bigger than average winners (or the reverse: a low
    win rate can still be profitable if winners run and losers are cut
    short). Pulled from vectorbt's per-trade records, not its summary stats.
    """
    trades = pf.trades.records_readable
    if len(trades) == 0:
        return {
            "num_trades": 0, "num_wins": 0, "num_losses": 0,
            "avg_win": np.nan, "avg_loss": np.nan,
            "median_win": np.nan, "median_loss": np.nan,
            "largest_win": np.nan, "largest_loss": np.nan,
            "avg_win_loss_ratio": np.nan,
            "avg_holding_days": np.nan,
            "expectancy_per_trade": np.nan,
        }

    pnl = trades["PnL"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    holding_days = (
        pd.to_datetime(trades["Exit Timestamp"]) - pd.to_datetime(trades["Entry Timestamp"])
    ).dt.days

    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan

    return {
        "num_trades": int(len(trades)),
        "num_wins": int(len(wins)),
        "num_losses": int(len(losses)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "median_win": float(wins.median()) if len(wins) else np.nan,
        "median_loss": float(losses.median()) if len(losses) else np.nan,
        "largest_win": float(wins.max()) if len(wins) else np.nan,
        "largest_loss": float(losses.min()) if len(losses) else np.nan,
        "avg_win_loss_ratio": float(abs(avg_win / avg_loss)) if avg_loss not in (0, np.nan) and not np.isnan(avg_loss) else np.nan,
        "avg_holding_days": float(holding_days.mean()),
        "expectancy_per_trade": float(pnl.mean()),
    }


def compare_to_buy_and_hold(
    signals_df: pd.DataFrame,
    price_col: str = "entry_price",
    signal_col: str = "signal",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run both backtests and return a side-by-side performance table."""
    cfg = config or BacktestConfig()
    strategy_pf = run_strategy_backtest(signals_df, price_col, signal_col, cfg)
    bh_pf = run_buy_and_hold_backtest(signals_df[price_col], cfg)

    return pd.DataFrame(
        {
            "strategy": performance_report(strategy_pf, cfg),
            "buy_and_hold": performance_report(bh_pf, cfg),
        }
    ).T
