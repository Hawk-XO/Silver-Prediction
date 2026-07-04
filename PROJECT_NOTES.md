# PROJECT_NOTES.md

This file is the source of truth for conventions in this project. Read it before
writing or modifying any code in this repo. If you are an AI assistant working
on a later phase, treat every rule below as binding unless the user explicitly
overrides it in the prompt.

---

## 1. Project Goal

Build a forecasting + signal-generation framework for **MCX Silver futures**
(India), using an ensemble of ARIMA, XGBoost, and LSTM models, validated with a
strict walk-forward methodology, and eventually wired into a paper-trading
broker sandbox (Zerodha Kite Connect).

Note: NSE does not list silver derivatives — silver futures trade on **MCX**
(Multi Commodity Exchange of India). All data ingestion should target MCX
contracts (SILVER, SILVERM, SILVERMIC), not NSE.

---

## 2. Target Variable Definition

The prediction target is the **forward log return**:

```
target[t] = log(close[t + h]) - log(close[t])
```

where `h` is the forecast horizon (default: `h = 1`, i.e. next-day close-to-close
log return, for daily bars). If the horizon is later changed to `h > 1`, every
walk-forward fold MUST purge training samples whose label window
`[t, t+h]` overlaps the test period (see Section 4).

We predict **returns, not price levels**. Raw price is non-stationary; do not
fit models directly on price without differencing/log-returning first (confirm
with an ADF test if unsure).

---

## 3. Anti-Leakage Rule for Features (non-negotiable)

**Every rolling, lagged, or windowed feature must be computed using only
information available strictly before time t.**

In practice this means:

```python
# WRONG — includes today's close in today's feature
df['roll_mean_5'] = df['close'].rolling(5).mean()

# RIGHT — shift first, so row t only sees data through t-1
df['roll_mean_5'] = df['close'].shift(1).rolling(5).mean()
```

This applies to: rolling means/std/skew/kurtosis, technical indicators (EMA,
RSI, MACD, ATR, Bollinger Bands, ADX), rolling correlations, and any lag
feature. If a feature construction function does not visibly show a
`.shift(1)` (or equivalent) before a rolling/window operation, it is
considered a bug.

Every feature-engineering module must ship with a pytest test asserting that
no feature value at row `t` depends on data at row `>= t`.

---

## 4. Walk-Forward Validation (non-negotiable)

- Use **expanding-window walk-forward validation**. Never use k-fold
  cross-validation with shuffling on time-series data.
- Pattern: train on all rows up to day `N`, test exclusively on day `N+1`,
  then roll forward one day and repeat.
- **Purging**: if the target label for a training row's window overlaps the
  test period, that row must be dropped from the training fold.
- **Embargo**: after each test point, leave a buffer of `h` days before
  resuming eligibility of subsequent rows for training, to prevent
  information bleed through overlapping label windows.
- **Scalers/encoders must be refit per fold** on training data only, then
  applied to the test fold. Never fit a scaler on the full dataset upfront.
- Execution logic must respect real-world timing: a signal generated from a
  given day's close can only be executed at the next available open, never at
  that same day's close.

---

## 5. Repository Structure

```
nse-silver-algo/
├── data/         # data ingestion: loaders, roll-adjustment, cross-asset fetchers
├── features/     # feature engineering: indicators, rolling stats, target construction
├── models/       # ARIMA/XGBoost/LSTM wrapper classes + stacking meta-learner
├── backtest/     # walk-forward harness, vectorbt/backtrader wiring
├── signals/      # signal execution engine: BUY/SELL/HOLD logic, position sizing
├── tests/        # pytest suite — especially leakage tests (Sections 3 & 4)
├── config/       # config files, .env loading, contract specs
├── requirements.txt
├── PROJECT_NOTES.md   # <- this file
└── README.md
```

Each folder is a Python package (`__init__.py` present) so modules can import
across folders, e.g. `from features.indicators import add_rsi`.

---

## 6. Phase Plan (for continuity across chats)

1. Project scaffolding (this phase)
2. Data ingestion (MCX loader, continuous contract roll, COMEX/USDINR/DXY fetch)
3. Feature engineering (indicators, rolling stats, leakage-safe target)
4. Modeling layer (ARIMA, XGBoost, LSTM, stacking meta-learner)
5. Walk-forward validation harness + leakage pytest suite
6. Signal execution engine
7. Backtest (vectorbt) + paper trading wrapper (Kite Connect sandbox)

When starting a new phase in a new chat, attach the zip from the previous
phase so this file and all prior code are present as context.

---

## 7. Secrets

Never commit API keys, tokens, or credentials. All secrets load from a local
`.env` file (excluded via `.gitignore`) using `python-dotenv`. Broker
credentials (Kite Connect) are only introduced in Phase 7, as a sandbox
wrapper — no live trading logic belongs in this repo without explicit,
separate confirmation.
