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

