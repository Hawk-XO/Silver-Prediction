"""
backtest/atr_exit_backtest.py

run_strategy_backtest() in vectorbt_backtest.py exits a trade ONLY when the
signal changes (BUY -> HOLD/SELL, etc). It never looks at the `stop_loss`
and `target` columns that signal_engine.generate_signals() computes for
every trade. This module fixes that: a trade closes at whichever of these
happens FIRST, checked in this order each day:
    1. price touches the ATR-based stop_loss level  -> exit_reason="stop_hit"
    2. price touches the ATR-based target level      -> exit_reason="target_hit"
    3. the signal itself changes (flips or goes HOLD) -> exit_reason="signal_exit"
    4. end of the data window (open position forced closed) -> "end_of_data"

Why this matters
-----------------
The trade breakdown on the pre-COVID window showed avg_loss ~1.5x avg_win
despite a >50% win rate -- a "cut winners short, let losers run" pattern.
That's the SIGNATURE of stop-loss/target levels being computed but not
enforced: a bad trade rides all the way to the next signal flip, however
far that is, instead of being cut at a pre-defined stop. This backtest
tests whether actually enforcing those levels fixes the win/loss ratio.

Known approximation -- read before trusting the numbers
----------------------------------------------------------
This checks each day's CLOSE price against the stop/target level, not
intraday high/low, because reliable intraday OHLC isn't joined into the
signal pipeline yet. A real stop-loss usually triggers intraday, so this
likely UNDERSTATES how much adverse movement happens before a stop would
really fire in live trading (a real broker-side stop would often cut a
losing trade sooner, at a worse average price than this simulation
assumes, since gaps and intraday spikes aren't visible here). Treat this
as a best-case estimate of what enforcing stops would do, not a precise
live-trading simulation. Position sizing is 100%-of-equity per trade
(compounding), matching run_strategy_backtest()'s targetpercent sizing, so
the two are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AtrExitConfig:
    fees: float = 0.0003        # per leg (entry and exit charged separately), matches BacktestConfig
    slippage: float = 0.0005    # per leg
    init_cash: float = 1_000_000.0


def run_atr_exit_backtest(
    signals_df: pd.DataFrame,
    config: AtrExitConfig | None = None,
) -> dict:
    """
    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of signals.signal_engine.generate_signals() -- must contain
        entry_price (used as each day's close), stop_loss, target, signal.

    Returns
    -------
    dict with:
        equity_curve: pd.Series indexed like signals_df
        trades: pd.DataFrame, one row per closed trade, with exit_reason
        stats: dict of aggregate performance stats (same keys as
            backtest.vectorbt_backtest.performance_report where applicable,
            so the two are directly comparable side by side)
        exit_reason_counts: dict, how trades actually closed
    """
    cfg = config or AtrExitConfig()
    close = signals_df["entry_price"]
    signal = signals_df["signal"]
    stop_col = signals_df["stop_loss"]
    target_col = signals_df["target"]
    idx = signals_df.index

    equity = cfg.init_cash
    equity_curve = []
    position = 0          # +1 long, -1 short, 0 flat
    entry_price = np.nan
    entry_equity = np.nan
    entry_date = None
    stop_level = np.nan
    target_level = np.nan
    prev_close = None

    trades = []

    for i in range(len(idx)):
        date = idx[i]
        px = close.iloc[i]
        sig = signal.iloc[i]

        # Mark existing position to market on this bar's move before
        # checking exits, so PnL reflects the move that triggered the exit.
        if position != 0 and prev_close is not None and not np.isnan(prev_close):
            day_ret = (px - prev_close) / prev_close * position
            equity *= (1 + day_ret)

        if position != 0:
            exit_reason = None
            exit_price = px
            if position == 1:
                if px <= stop_level:
                    exit_reason, exit_price = "stop_hit", stop_level
                elif px >= target_level:
                    exit_reason, exit_price = "target_hit", target_level
            else:
                if px >= stop_level:
                    exit_reason, exit_price = "stop_hit", stop_level
                elif px <= target_level:
                    exit_reason, exit_price = "target_hit", target_level

            if exit_reason is None and (
                (position == 1 and sig != "BUY") or (position == -1 and sig != "SELL")
            ):
                exit_reason, exit_price = "signal_exit", px

            if exit_reason is not None:
                equity *= (1 - cfg.fees - cfg.slippage)  # exit cost
                pnl = equity - entry_equity
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "direction": "long" if position == 1 else "short",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "holding_days": (date - entry_date).days if hasattr(date - entry_date, "days") else np.nan,
                    "pnl": pnl, "return_pct": (equity / entry_equity - 1) * 100,
                })
                position = 0
                entry_price = entry_equity = np.nan
                entry_date = None

        # Open a new position only if flat AND today's (post-exit) signal calls for one.
        if position == 0 and sig in ("BUY", "SELL"):
            equity *= (1 - cfg.fees - cfg.slippage)  # entry cost
            position = 1 if sig == "BUY" else -1
            entry_price = px
            entry_equity = equity
            entry_date = date
            stop_level = stop_col.iloc[i]
            target_level = target_col.iloc[i]

        equity_curve.append(equity)
        prev_close = px

    # Force-close anything still open at the end of the window.
    if position != 0:
        equity *= (1 - cfg.fees - cfg.slippage)
        pnl = equity - entry_equity
        trades.append({
            "entry_date": entry_date, "exit_date": idx[-1],
            "direction": "long" if position == 1 else "short",
            "entry_price": entry_price, "exit_price": close.iloc[-1],
            "exit_reason": "end_of_data",
            "holding_days": (idx[-1] - entry_date).days if hasattr(idx[-1] - entry_date, "days") else np.nan,
            "pnl": pnl, "return_pct": (equity / entry_equity - 1) * 100,
        })
        equity_curve[-1] = equity

    equity_series = pd.Series(equity_curve, index=idx, name="equity")
    trades_df = pd.DataFrame(trades)

    daily_ret = equity_series.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd_pct = float(-drawdown.min() * 100)
    total_return_pct = float((equity_series.iloc[-1] / cfg.init_cash - 1) * 100)

    if len(trades_df) > 0:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] < 0, "pnl"]
        win_rate_pct = float(len(wins) / len(trades_df) * 100)
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        avg_win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss not in (0,) and not np.isnan(avg_loss) else np.nan
        exit_reason_counts = trades_df["exit_reason"].value_counts().to_dict()
    else:
        win_rate_pct = profit_factor = avg_win = avg_loss = avg_win_loss_ratio = np.nan
        exit_reason_counts = {}

    stats = {
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "total_trades": int(len(trades_df)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "expectancy_per_trade": float(trades_df["pnl"].mean()) if len(trades_df) else np.nan,
    }

    return {
        "equity_curve": equity_series,
        "trades": trades_df,
        "stats": stats,
        "exit_reason_counts": exit_reason_counts,
    }
