# NSE/MCX Silver Futures Algorithmic Trading Framework

A Python framework for forecasting MCX Silver futures returns and generating
buy/sell signals, using an ensemble of ARIMA, XGBoost, and LSTM models with
strict walk-forward validation.

See [`PROJECT_NOTES.md`](./PROJECT_NOTES.md) for binding conventions
(target definition, anti-leakage rules, validation methodology) before
modifying any code.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Streamlit UI

```bash
streamlit run ui/app.py
```

A three-screen multipage app:

- **Start** (`ui/pages/1_menu.py`) — checks stored data freshness (fetches
  any missing days via Kite, falling back to the COMEX+USDINR proxy) before
  either downstream screen runs, then routes you onward.
- **Predictor** (`ui/pages/2_predictor.py`) — configure model
  hyperparameters and run a full walk-forward backtest: Sharpe, drawdown,
  win rate, strategy vs buy-and-hold, PDF export.
- **Market Simulator** (`ui/pages/3_market_sim.py`) — the same real-data
  pipeline, replayed as a paper-trading order log (`PaperKiteBroker`, see
  `broker/kite_paper_broker.py`): equity curve, fill-by-fill order table,
  running paper P&L. Simulated only — no real orders are ever sent.

`ui/pipeline_runner.py` and `ui/market_sim_runner.py` hold all the actual
pipeline logic; the pages themselves are just widgets wired to those two
modules, so the pipeline can be smoke-tested from plain Python/pytest
without starting Streamlit.

## Project Structure

```
data/       # ingestion: MCX loader, continuous contract roll, cross-asset fetch
features/   # indicators, rolling stats, leakage-safe target construction
models/     # ARIMA, XGBoost, LSTM, stacking meta-learner
backtest/   # walk-forward harness, vectorbt/backtrader wiring
signals/    # BUY/SELL/HOLD signal engine
ui/         # Streamlit multipage app (see "Streamlit UI" above)
tests/      # pytest suite (leakage tests are mandatory, see PROJECT_NOTES.md)
config/     # config + .env loading
```

## Build Phases

1. Scaffolding (this commit)
2. Data ingestion
3. Feature engineering
4. Modeling layer
5. Walk-forward validation
6. Signal execution engine
7. Backtest + paper trading (Kite Connect sandbox)

## Status

`Phase 2 complete: data ingestion layer implemented and tested.`

### Phase 2 additions

- `data/mcx_loader.py` — loads MCX Silver futures OHLCV CSVs, tolerant of
  column-naming differences across brokers/vendors (via alias matching).
- `data/contract_roll.py` — builds a ratio-adjusted continuous price series
  across monthly MCX contract expiries (`build_continuous_series`).
- `data/global_factors.py` — fetches COMEX Silver, USD/INR, and DXY via
  yfinance.
- `data/merge.py` — aligns MCX continuous series with global factors on
  the MCX trading calendar, forward-filling global factors on days the US
  market was closed (with a `_stale` flag per source and a fill-limit to
  avoid propagating indefinitely).
- `data/synthetic.py` — generates a synthetic multi-contract MCX dataset
  plus matching global factors, purely so the pipeline can be tested
  end-to-end offline before real data is available. **Not for modeling.**
- `tests/test_data_pipeline.py` — end-to-end pytest suite covering the
  full ingestion pipeline, including a check that ratio-adjustment removes
  artificial roll-date price jumps. All 5 tests currently pass.

### Addendum: batch bhavcopy processor

MCX's website disallows automated/bot access (robots.txt), so fully
automated fetching of MCX Silver data isn't something this project scrapes
around — that's their stated policy on their own data. Instead:

- `data/batch_bhavcopy_processor.py` — automates everything AFTER a manual
  download. Drop any number of manually-downloaded MCX bhavcopy or
  historical-data CSVs into a folder (any naming, any date range), and
  `process_bhavcopy_folder(folder, commodity="SILVER")` filters to the
  target commodity, builds a proper per-expiry contract identifier
  (commodity + expiry date, so different expiry months aren't collapsed
  into one contract), and outputs data in the exact schema
  `build_continuous_series()` expects.
- `exact_match=True` (default) ensures "SILVER" doesn't also pull in
  "SILVERM"/"SILVERMIC" as substring matches — these are distinct products
  and mixing them would corrupt the continuous series.
- `tests/test_batch_bhavcopy_processor.py` — 6 tests covering exact-match
  filtering, contract/expiry identification, and end-to-end feed into
  `build_continuous_series()`. All passing.
- `data/mcx_loader.py`'s column aliases were expanded to cover real MCX
  bhavcopy column names (`TradDt`, `OpenPrice`, `ClosePrice`,
  `TotalTradedQty`, etc.) discovered while building this test.

### Addendum 2: fully-automated MCX proxy (recommended path going forward)

Manual bhavcopy downloads (Addendum 1) work but are tedious. `data/mcx_proxy.py`
gives a **zero-manual-step, fully automated** alternative: it reconstructs
an MCX Silver-equivalent price series from COMEX Silver + USD/INR (both
already fetched automatically via `global_factors.py`), using the same
import-parity relationship that drives MCX Silver's actual pricing:

```
mcx_proxy_price = comex_silver_usd_per_oz * usdinr * 32.1507 (oz->kg) * (1 + premium_pct)
```

- `fetch_mcx_silver_proxy(start, end, premium_pct)` — fully automated fetch, no downloads.
- `calibrate_premium(proxy_df, real_mcx_close)` — once you have even a small
  sample of real MCX prices (5-10 days is enough — from a manual bhavcopy
  pull, or eyeballing a chart), this fits the constant premium/discount so
  the proxy's absolute price level matches real MCX, not just its shape.
- Tested via mocked COMEX/USDINR inputs (4 tests, all passing) — the
  calculation logic, premium scaling, and calibration math are verified.
  **The live yfinance fetch itself has NOT been tested end-to-end**, because
  this build sandbox's network doesn't reach Yahoo Finance — run
  `fetch_mcx_silver_proxy()` once yourself to confirm it returns real data
  before relying on it.

Trade-off to know about: the proxy's absolute price *level* is
approximate until calibrated (import duty/local premium isn't something
COMEX+USDINR alone can capture). Its day-to-day *returns* — which is what
we actually model, per PROJECT_NOTES.md's target definition — are much
more reliable immediately, since COMEX and USDINR moves dominate MCX
Silver's daily return regardless of the fixed premium.

Recommended going forward: use `mcx_proxy.py` as the primary, always-fresh
data source for Phases 3-7, and treat the manual `batch_bhavcopy_processor.py`
route as an optional calibration/validation source when you want to
sanity-check the proxy against real prices occasionally.

### Known limitation

`global_factors.py` has not been tested against a live network call in
this environment (sandboxed, no internet access to Yahoo Finance). The
yfinance ticker symbols (`SI=F`, `USDINR=X`, `DX-Y.NYB`) are current as of
this writing but should be verified the first time you run it for real —
if a fetch returns empty, search for the current correct ticker rather
than assuming these are permanent.
