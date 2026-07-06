"""
signals/signal_engine.py

Phase 6: turns model predictions into BUY/SELL/HOLD signals.

Pipeline for each row:
  1. CONFIDENCE — how big is the predicted move relative to typical recent
     noise? confidence = |prediction| / recent_volatility. Below
     `confidence_threshold` -> HOLD regardless of anything else.
  2. VOLATILITY REGIME FILTER — if ATR (normalized by price) is unusually
     high relative to its own recent history, suppress trading entirely
     (HOLD), even if confidence was high — high-ATR regimes are exactly
     when a return prediction is least trustworthy and slippage is worst.
  3. RAW DIRECTION — sign of the prediction (BUY if positive, SELL if
     negative), only reached if steps 1-2 didn't already force HOLD.
  4. COOLDOWN — if a signal in the OPPOSITE direction fired within the
     last `cooldown_days` days, suppress the flip (HOLD) rather than
     immediately reversing a position that was just opened. Same-direction
     signals and HOLDs are never blocked by cooldown (they can't cause a
     flip).
  5. ATR-based stop-loss / target levels are attached to every BUY/SELL,
     computed off the row's own ATR and close price.

Every row (not just BUY/SELL) is logged, since a HOLD is a decision too —
the log records the reason a signal was or wasn't taken, plus the full
feature snapshot that produced it, so signal quality can be audited later.

Anti-leakage note
------------------
Per PROJECT_NOTES.md Section 3, the rolling ATR-percentile threshold used
for the volatility regime filter must not use today's own ATR reading in
the window it's compared against — so, consistent with the rest of the
project, we `.shift(1)` the normalized-ATR series before computing its
rolling quantile. The current row's own ATR value (already itself a
lagged, leakage-safe feature from features/indicators.py) is what gets
compared against that threshold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    pred_col: str = "meta_pred"
    volatility_col: str = "ret_std_20"     # rolling return-vol feature from Phase 3
    atr_col: str = "atr_14"
    price_col: str = "mcx_close"
    confidence_threshold: float = 0.5      # predicted move must be >= 0.5x recent daily vol
    atr_regime_window: int = 60
    atr_regime_percentile: float = 0.80    # suppress trading above this trailing ATR percentile
    stop_loss_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    cooldown_days: int = 3


def compute_confidence(df: pd.DataFrame, pred_col: str, volatility_col: str) -> pd.Series:
    """|prediction| / recent realized volatility — a signal-to-noise ratio.
    A prediction of 0.5% with typical daily vol of 1% has confidence 0.5
    (a fairly ordinary-sized move); a 2% prediction against 1% vol has
    confidence 2.0 (an unusually large, higher-conviction call)."""
    vol = df[volatility_col].replace(0, np.nan)
    return (df[pred_col].abs() / vol).replace([np.inf, -np.inf], np.nan)


def compute_volatility_regime_flag(
    df: pd.DataFrame,
    atr_col: str,
    price_col: str,
    window: int,
    percentile: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (normalized_atr, rolling_threshold, high_vol_regime_flag).

    normalized_atr = ATR / price, so it's comparable across different
    price levels over time. The rolling threshold is computed on the
    LAGGED normalized-ATR series (shift(1) before rolling — see module
    docstring), so today's own ATR reading isn't part of the bar it's
    being judged against.
    """
    normalized_atr = df[atr_col] / df[price_col]
    lagged = normalized_atr.shift(1)
    threshold = lagged.rolling(window).quantile(percentile)
    high_vol_regime = normalized_atr > threshold
    return normalized_atr, threshold, high_vol_regime


def _apply_cooldown(raw_directions: pd.Series, cooldown_days: int) -> pd.Series:
    """
    Sequential pass: suppress a signal that would FLIP the position
    (BUY -> SELL or SELL -> BUY) within `cooldown_days` of the last
    non-HOLD signal. HOLD is never suppressed by cooldown (there's nothing
    to flip away from), and re-affirming the SAME direction is never
    suppressed either (that's not a flip-flop).
    """
    final = []
    last_direction = None
    last_signal_pos = None

    for pos, direction in enumerate(raw_directions):
        if direction == "HOLD":
            final.append("HOLD")
            continue

        is_flip = last_direction is not None and direction != last_direction
        within_cooldown = (
            last_signal_pos is not None and (pos - last_signal_pos) <= cooldown_days
        )

        if is_flip and within_cooldown:
            final.append("HOLD")
            # Position/direction state doesn't change — we're still
            # holding whatever the last real signal was.
            continue

        final.append(direction)
        last_direction = direction
        last_signal_pos = pos

    return pd.Series(final, index=raw_directions.index)


def generate_signals(df: pd.DataFrame, config: SignalConfig | None = None) -> pd.DataFrame:
    """
    Build BUY/SELL/HOLD signals for every row of `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `config.pred_col`, `config.volatility_col`,
        `config.atr_col`, `config.price_col`. Typically this is the
        feature matrix joined with walk-forward predictions (see
        run_phase6_demo.py for how to build that join).
    config : SignalConfig | None

    Returns
    -------
    pd.DataFrame indexed like `df`, with columns:
        prediction, confidence, normalized_atr, atr_threshold,
        high_vol_regime, raw_direction, signal, cooldown_blocked,
        entry_price, stop_loss, target
    """
    cfg = config or SignalConfig()
    out = pd.DataFrame(index=df.index)

    out["prediction"] = df[cfg.pred_col]
    out["confidence"] = compute_confidence(df, cfg.pred_col, cfg.volatility_col)

    normalized_atr, atr_threshold, high_vol_regime = compute_volatility_regime_flag(
        df, cfg.atr_col, cfg.price_col, cfg.atr_regime_window, cfg.atr_regime_percentile
    )
    out["normalized_atr"] = normalized_atr
    out["atr_threshold"] = atr_threshold
    out["high_vol_regime"] = high_vol_regime.fillna(False)

    confident_enough = out["confidence"] >= cfg.confidence_threshold
    tradable_regime = ~out["high_vol_regime"]

    direction = np.where(out["prediction"] > 0, "BUY", "SELL")
    raw_direction = np.where(confident_enough & tradable_regime, direction, "HOLD")
    out["raw_direction"] = raw_direction

    out["signal"] = _apply_cooldown(out["raw_direction"], cfg.cooldown_days)
    out["cooldown_blocked"] = (out["raw_direction"] != "HOLD") & (out["signal"] == "HOLD")

    out["entry_price"] = df[cfg.price_col]
    atr_value = df[cfg.atr_col]
    is_buy = out["signal"] == "BUY"
    is_sell = out["signal"] == "SELL"

    out["stop_loss"] = np.nan
    out["target"] = np.nan
    out.loc[is_buy, "stop_loss"] = out.loc[is_buy, "entry_price"] - cfg.stop_loss_atr_mult * atr_value[is_buy]
    out.loc[is_buy, "target"] = out.loc[is_buy, "entry_price"] + cfg.target_atr_mult * atr_value[is_buy]
    out.loc[is_sell, "stop_loss"] = out.loc[is_sell, "entry_price"] + cfg.stop_loss_atr_mult * atr_value[is_sell]
    out.loc[is_sell, "target"] = out.loc[is_sell, "entry_price"] - cfg.target_atr_mult * atr_value[is_sell]

    return out


def log_signals_to_csv(
    signals_df: pd.DataFrame,
    feature_snapshot_df: pd.DataFrame,
    path: str = "signals/signal_log.csv",
) -> str:
    """
    Write every signal (BUY/SELL/HOLD, one row per date) to CSV, joined
    with the feature snapshot that produced it — a full audit trail of
    "what did the model see when it made this call".

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of generate_signals().
    feature_snapshot_df : pd.DataFrame
        The feature matrix row(s) (e.g. features.pipeline output, restricted
        to feature columns) aligned to the same index as `signals_df`.
    path : str
        Output CSV path. Parent directory is created if it doesn't exist.

    Returns
    -------
    str: the path written to.
    """
    combined = signals_df.join(feature_snapshot_df, how="left")
    combined = combined.reset_index().rename(columns={"index": "date"})

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    combined.to_csv(path, index=False)
    return path
