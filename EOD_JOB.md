# EOD job (Track C)

Automates the "pull today's close, store it, predict on it" loop that
`ui/app.py` otherwise requires a person to trigger by hand.

## What it does, in order

1. **Pull** — `data/kite_fetcher.py::fetch_latest_eod(commodity)` fetches the
   front-month contract's latest daily bar from Kite Connect and upserts it
   into MySQL (`data/db.py`, table `mcx_silver_ohlcv`). This already existed;
   Track C's job was to actually *call* it on a schedule.
2. **Rebuild** — the next step (predict) re-runs feature engineering over the
   full stored history so the new row gets its indicators. This is a full
   rebuild, not an incremental update — see the caveat below.
3. **Predict** — `signals/live_predict.py::generate_live_signal()` refits
   ARIMA + XGBoost + the Ridge meta-learner on all history with a resolved
   label, and predicts the newest (label-less) row via
   `run_walk_forward(..., include_live_row=True)` — the same purge/embargo-safe
   fold logic the backtest uses, just for one extra "live" fold.
4. **Alert on failure** — any exception, or a returned row that's older than
   `MAX_STALE_CALENDAR_DAYS` (4) without a holiday explaining it, is logged
   at `CRITICAL` and optionally POSTed to `ALERT_WEBHOOK_URL` (a Slack
   incoming-webhook-compatible URL) if that's set in `.env`.

## Persistent store: MySQL, not parquet/SQLite

The original spec suggested parquet or SQLite since "everything is
generated fresh in-memory each run." That's true of the *feature matrix*,
but the raw OHLCV data was already durably stored in MySQL from earlier
phases. Adding a second on-disk store alongside it would just be a second
place the two could drift out of sync — so this job upserts into the
existing table instead of introducing a new one.

## "Rebuild", not "incremental update"

The original spec asked for an incremental feature-engineering update.
What's actually implemented re-runs `build_feature_matrix()` over the full
stored history every time. At this project's data volume (thousands of
rows, not millions) a full rebuild + refit finishes in under two minutes
(see the timing note in `run_eod_job.py`'s module docstring) — incremental
feature computation would have added real complexity (tracking which
rolling windows are still valid, cache invalidation on backfills/corrections)
for a speed win that doesn't matter yet. Revisit if the stored history grows
enough that this stops being true.

## Running it

```bash
# One-off, for an external cron entry:
python run_eod_job.py --once

# Example crontab line (23:45 IST, weekdays):
45 23 * * 1-5  cd /path/to/main && venv/bin/python run_eod_job.py --once >> logs/cron.log 2>&1

# OR: leave one process running with its own built-in scheduler
python run_eod_job.py
```

Logs go to `logs/eod_job.log` (rotating, 5 x 2MB) and stdout.

## Alerting setup

Set `ALERT_WEBHOOK_URL` in `.env` to a Slack incoming-webhook URL (or any
endpoint that accepts `{"text": "..."}` JSON) to get a message posted there
on failure, in addition to the CRITICAL log line that always gets written
regardless of whether the webhook is configured or reachable.

## Known gaps / things not yet built

- No dead-man's-switch check (e.g. "alert if the job hasn't run successfully
  in 3 days") — currently relies on cron's own failure visibility (non-zero
  exit code) or on someone noticing `logs/eod_job.log` stopped updating.
- `is_stale()`'s threshold (4 calendar days) is a starting guess, not tuned
  against MCX's actual holiday calendar — a run of 2+ consecutive holidays
  would need a slightly higher threshold to avoid a false alarm.
- The live signal is logged, not persisted anywhere queryable (no
  `live_signals` table) — today's prediction only lives in the log file.
  Worth adding a small table if you want to track live-signal history over
  time rather than just the most recent run's log line.
