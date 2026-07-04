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

## Project Structure

```
data/       # ingestion: MCX loader, continuous contract roll, cross-asset fetch
features/   # indicators, rolling stats, leakage-safe target construction
models/     # ARIMA, XGBoost, LSTM, stacking meta-learner
backtest/   # walk-forward harness, vectorbt/backtrader wiring
signals/    # BUY/SELL/HOLD signal engine
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

### Known limitation

`global_factors.py` has not been tested against a live network call in
this environment (sandboxed, no internet access to Yahoo Finance). The
yfinance ticker symbols (`SI=F`, `USDINR=X`, `DX-Y.NYB`) are current as of
this writing but should be verified the first time you run it for real —
if a fetch returns empty, search for the current correct ticker rather
than assuming these are permanent.
