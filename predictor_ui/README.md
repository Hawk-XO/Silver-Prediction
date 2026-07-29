# predictor_ui/

A single-file React dashboard (`silver_options_predictor.jsx`) for the
project's daily BUY / SELL / HOLD signal, built on **real pipeline output**
rather than in-browser fake math.

## What's in here

- **`silver_options_predictor.jsx`** — the app. Three pages (Home,
  Predictor, Market Prediction). The pipeline output below is embedded
  directly in the file as a JS constant (`PIPELINE_DATA`) so it's a
  drop-in single-file artifact with no fetch/build step needed to view it.
- **`predictor_data.json`** — the raw export from `export_predictor_data.py`
  (kept alongside the `.jsx` for inspection/diffing; the `.jsx` has its own
  embedded copy and does not read this file at runtime).

## How the data was produced

`export_predictor_data.py` (at the project root) runs the project's real
pipeline end-to-end:

```
data/synthetic.py (synthetic MCX/COMEX/USDINR/DXY prices)
  -> data/contract_roll.py + data/merge.py
  -> features/pipeline.py (real feature engineering)
  -> backtest/walk_forward.py (real ARIMA + XGBoost + meta-learner,
     purge/embargo-safe walk-forward fitting)
  -> signals/signal_engine.py (real signal generation)
  -> backtest/vectorbt_backtest.py (real vectorbt backtest vs. buy-and-hold)
```

Nothing about the modeling or backtest math is simplified for the UI — it's
the identical code path used by `run_phase7_demo.py` and the test suite.

**Why synthetic data, not real MCX prices:** there's no live market feed
reachable from the sandbox this was built in (no Kite session, no outbound
access to Zerodha's API). `data/synthetic.py` is the project's own built-in
data generator, used elsewhere in the project for exactly this purpose —
exercising the real pipeline end-to-end without a live feed. Its own
docstring is explicit that it has *no genuine predictive structure* (pure
random walk), so the resulting ~51% directional accuracy is the **correct,
expected** result — it demonstrates the walk-forward harness isn't leaking
future information into training, not a weakness of the model. A
much-better-than-50% number on this particular dataset would actually be
the red flag.

## Refreshing the data

Once real MCX history is flowing through `data/db.py` (via `run_eod_job.py`
or a manual backfill), swap `export_predictor_data.py`'s data source from
`data.synthetic.generate_full_synthetic_dataset(...)` to
`data.pipeline_common.load_real_features(...)` (the same loader
`signals/live_predict.py` uses) and re-run:

```bash
python export_predictor_data.py
```

Then re-embed the resulting `predictor_ui/predictor_data.json` into
`silver_options_predictor.jsx`'s `PIPELINE_DATA` constant (replace the
object literal directly — it's plain JSON, no build step required).
