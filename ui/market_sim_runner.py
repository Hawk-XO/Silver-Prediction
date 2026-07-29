"""
ui/market_sim_runner.py

Backend for the "Market Simulator" page (ui/pages/3_market_sim.py).

Deliberately NOT a reimplementation: the Predictor page and the Market
Simulator page are the SAME underlying pipeline (features -> walk-forward
-> signals -> vectorbt backtest -> paper-broker replay, see
ui/pipeline_runner.run_pipeline). The only thing that differs is framing --
the Predictor page is about "does this model/parameter combo predict well",
the Market Simulator page is about "what would placing these signals as
paper orders actually have looked like, order by order, no real money" --
so this module just calls run_pipeline() and adds one presentation helper
(build_orders_table) for the order-log view the market-sim page shows and
the Predictor page doesn't. No Streamlit imports here, same as
pipeline_runner, so it's independently testable.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from ui.pipeline_runner import RunConfig, RunResult, run_pipeline

# Re-exported so ui/pages/3_market_sim.py only needs to import from this
# one module instead of reaching into pipeline_runner directly.
__all__ = ["MarketSimConfig", "run_market_sim", "build_orders_table"]

MarketSimConfig = RunConfig  # same knobs; alias just makes intent clear on the market-sim page


def run_market_sim(
    cfg: MarketSimConfig,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> RunResult:
    """Runs the real pipeline end-to-end, including the paper-broker
    replay. Returns the same RunResult the Predictor page uses -- the
    market-sim page just chooses to render `.orders` / `.broker_pnl`
    front-and-center instead of the model-comparison table."""
    return run_pipeline(cfg, progress_callback=progress_callback)


def build_orders_table(result: RunResult) -> pd.DataFrame:
    """
    Flattens RunResult.orders (list of dicts shaped like Kite Connect's
    order objects, see broker/kite_paper_broker.py) into a display-ready
    DataFrame for st.dataframe -- most recent order first, only the
    columns that matter for a paper-trading log.
    """
    if not result.orders:
        return pd.DataFrame(
            columns=["tag", "tradingsymbol", "transaction_type", "quantity", "price", "status"]
        )
    df = pd.DataFrame(result.orders)
    cols = [c for c in ["tag", "tradingsymbol", "transaction_type", "quantity", "price", "status"] if c in df.columns]
    df = df[cols].rename(columns={
        "tag": "Date",
        "tradingsymbol": "Symbol",
        "transaction_type": "Side",
        "quantity": "Qty",
        "price": "Fill price",
        "status": "Status",
    })
    return df.iloc[::-1].reset_index(drop=True)
