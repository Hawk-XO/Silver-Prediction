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
4. **Persist** — today's live signal is upserted into the `live_signals`
   table (`data/db.py`) so it's queryable after the fact, not just visible
   in the log line at the moment the job ran.
5. **Heartbeat** — a small JSON status file (`EOD_HEARTBEAT_PATH` in `.env`,
   default `logs/heartbeat.json`) is written at every exit path (success,
   no-data/holiday, or error), for `check_heartbeat.py`'s independent
   dead-man's-switch check to read on its own separate cron schedule.
6. **Alert on failure** — any exception, or a returned row that's older than
   `EOD_MAX_STALE_CALENDAR_DAYS` (default 4) without a holiday explaining it,
   is logged at `CRITICAL` and optionally POSTed to `ALERT_WEBHOOK_URL` (a
   Slack incoming-webhook-compatible URL) if that's set in `.env`.

## Startup guard: refuses to run on a placeholder MySQL password

Before doing anything else, `run_once()` checks
`settings.mysql_password_is_placeholder` and refuses to run (logs
`CRITICAL`, alerts, raises) if `.env`'s `MYSQL_PASSWORD` is still one of the
known example/placeholder values (`silver_pass`, `pick_your_own_password`,
`changeme`, empty, etc). This is deliberately loud and fatal rather than a
warning — a job silently failing every MySQL call because of an unfilled
placeholder is a much worse failure mode than refusing to start.

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

Run `check_heartbeat.py` on its own **separate** cron schedule (it deliberately
does not run inside `run_eod_job.py` itself — see "Dead-man's-switch" below):

```bash
# 09:00 IST daily, independent of the job's own 23:45 IST schedule:
0 9 * * *  cd /path/to/main && venv/bin/python check_heartbeat.py >> logs/heartbeat_check.log 2>&1
```

## Alerting setup

Set `ALERT_WEBHOOK_URL` in `.env` to a Slack incoming-webhook URL (or any
endpoint that accepts `{"text": "..."}` JSON) to get a message posted there
on failure, in addition to the CRITICAL log line that always gets written
regardless of whether the webhook is configured or reachable.

## Dead-man's-switch: check_heartbeat.py

Closes the gap noted below — a check *inside* `run_eod_job.py` can never
fire if the job's process died, crashed the interpreter, or was never
scheduled at all. `check_heartbeat.py` is a separate script, meant to run
on its own independent cron schedule, that:

1. Alerts if the heartbeat file is missing entirely (job has apparently
   never run, or its heartbeat file was deleted).
2. Alerts if the heartbeat file's timestamp is older than
   `EOD_HEARTBEAT_MAX_AGE_HOURS` (default 96h / 4 days — generous enough to
   cross a weekend + one holiday without a false alarm).
3. Alerts if the last recorded status was `"error"`, even if the timestamp
   itself is fresh — a job that's running on schedule but failing every
   time is exactly what this exists to catch, not just a job that stopped
   running entirely.

It reuses `run_eod_job.py`'s `send_alert()` (same `CRITICAL` log + optional
Slack webhook) rather than a second alerting implementation.

## Live signal history: the `live_signals` table

Resolved — every run now upserts today's signal into `live_signals`
(`data/db.py`: `LiveSignalRecord`, `upsert_live_signal()`,
`load_live_signals()`), keyed on `(date, commodity)`. This means the
day's prediction is queryable after the fact (`load_live_signals()`),
not just visible in whatever ran in the log file at the time.

## Known gaps / things not yet built

- `EOD_MAX_STALE_CALENDAR_DAYS` (configurable via `.env`, default 4) is
  still a starting guess, not tuned against MCX's actual holiday calendar —
  a run of 2+ consecutive holidays would need a slightly higher threshold
  to avoid a false alarm.
- `check_heartbeat.py` and `run_eod_job.py` share `send_alert()`'s webhook
  delivery but have no shared alert *de-duplication* — if the job fails
  every night for a week, that's a week of separate webhook posts rather
  than one "still broken" summary. Not a correctness problem, just noisy
  in a sustained-outage scenario.
