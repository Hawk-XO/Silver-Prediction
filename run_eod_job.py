"""
run_eod_job.py

Track C — automated EOD updater. See EOD_JOB.md for the full writeup;
short version:

Each run does 4 things:
  1. PULL — data.kite_fetcher.fetch_latest_eod(commodity) grabs the newest
     trading day for the front-month contract and upserts it into MySQL
     (data/db.py). MySQL *is* the persistent store here (not a separate
     parquet/SQLite file) — it's already durable, already what every other
     part of this project reads from, and adding a second store alongside
     it would just be a second place things could get out of sync.
  2. REBUILD — re-runs feature engineering over the full stored history so
     the new row gets its indicators. Not a true incremental update (see
     signals/live_predict.py's docstring) — a full rebuild is cheap enough
     at this data volume that it wasn't worth the complexity.
  3. PREDICT — refits ARIMA + XGBoost + the meta-learner on all resolved
     history and produces today's live BUY/SELL/HOLD signal
     (signals/live_predict.generate_live_signal), logged at INFO.
  4. ALERT ON FAILURE — any exception, or data that's suspiciously stale
     (see is_stale() below), is logged at CRITICAL and, if
     ALERT_WEBHOOK_URL is set in .env, POSTed to that Slack-compatible
     incoming-webhook URL. Never fails silently: a bad run always either
     raises (visible to cron's exit code / your terminal) or is logged
     loud enough to find, and it never upserts partial/garbled rows since
     fetch_latest_eod() itself either returns a clean row or raises.

Run modes
---------
    python run_eod_job.py --once
        Run a single update right now and exit. This is what you want if
        you're driving scheduling from an external cron entry, e.g.:

            45 23 * * 1-5  cd /path/to/main && venv/bin/python run_eod_job.py --once >> logs/cron.log 2>&1

    python run_eod_job.py
        Start an in-process APScheduler loop (BlockingScheduler) that
        fires the same job at 23:45 IST, Monday-Friday, and keeps running
        until killed. Use this if you'd rather leave one process running
        (e.g. in a screen/tmux session or as a systemd service) than rely
        on system cron.

Both modes log to logs/eod_job.log (rotating, 5 files x 2MB) as well as
stdout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import logging.handlers
import os
import sys
from pathlib import Path

import pandas as pd

from config.settings import settings
from data.kite_fetcher import fetch_latest_eod
from signals.live_predict import generate_live_signal

LOG_DIR = Path(__file__).parent / "logs"
MAX_STALE_CALENDAR_DAYS = 4  # weekend + 1 holiday, generous on purpose


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("eod_job")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "eod_job.log", maxBytes=2_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def send_alert(message: str) -> None:
    """
    Always logs CRITICAL (so it's in eod_job.log / cron.log regardless of
    anything else). Additionally POSTs to ALERT_WEBHOOK_URL if that env var
    is set (works as-is with a Slack incoming webhook; any endpoint that
    accepts {"text": "..."} JSON works). Silently skipped (just a WARNING
    log line) if the webhook isn't configured or the POST itself fails --
    an alerting failure must never crash the job or mask the original
    failure it was trying to report.
    """
    logger.critical(message)
    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return
    try:
        import requests
        resp = requests.post(webhook_url, json={"text": f"[MCX Silver EOD job] {message}"}, timeout=10)
        if resp.status_code >= 300:
            logger.warning(f"Alert webhook returned HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Couldn't deliver alert to webhook (job failure above is still logged): {e}")


def is_stale(latest_date: pd.Timestamp, today: dt.date) -> bool:
    """True if the newest row we have is suspiciously old given `today` --
    more than MAX_STALE_CALENDAR_DAYS calendar days back. A couple of days
    is normal (weekends, single holidays); more than that suggests the
    fetch is silently returning old data rather than actually failing."""
    age_days = (today - latest_date.date()).days
    return age_days > MAX_STALE_CALENDAR_DAYS


def run_once(commodity: str | None = None) -> None:
    commodity = commodity or settings.mcx_commodity
    logger.info(f"=== EOD job start ({commodity}) ===")

    try:
        hist = fetch_latest_eod(commodity)
    except Exception as e:
        send_alert(f"fetch_latest_eod({commodity}) raised: {e!r}")
        raise

    today = dt.date.today()
    if hist.empty:
        # fetch_latest_eod() already returns empty (not an exception) for
        # the ordinary "market hasn't closed yet / holiday" case -- not
        # itself an alert-worthy failure. It only becomes suspicious if
        # this keeps happening for MAX_STALE_CALENDAR_DAYS+ running --
        # cheap check: look at what's already in MySQL for this commodity.
        logger.info("No new row today (holiday or not yet closed) -- checking staleness of stored data...")
        from data.db import load_ohlcv
        stored = load_ohlcv()
        if "contract" in stored.columns:
            stored = stored[stored["contract"].str.startswith(commodity)]
        if stored.empty:
            send_alert(f"No stored {commodity} data at all AND today's fetch was empty -- "
                       f"nothing to fall back on, needs investigation.")
            return
        latest_stored = stored.index.max()
        if is_stale(latest_stored, today):
            send_alert(
                f"No new {commodity} row today AND the newest stored row is from "
                f"{latest_stored.date()} ({(today - latest_stored.date()).days} calendar days "
                f"ago) -- looks stale, not just a holiday. Check Kite auth / API status."
            )
        else:
            logger.info(f"Newest stored row is {latest_stored.date()} -- within normal range, no alert.")
        return

    latest_date = hist.index.max()
    if is_stale(latest_date, today):
        send_alert(
            f"fetch_latest_eod({commodity}) returned a row but it's dated {latest_date.date()}, "
            f"{(today - latest_date.date()).days} calendar days old -- looks stale rather than "
            f"today's actual close. Not treating this as a clean update."
        )
        return

    logger.info(f"New row upserted for {latest_date.date()}.")

    try:
        live = generate_live_signal(commodity=commodity)
        logger.info(
            f"Live signal for {live.date.date()}: {live.signal} "
            f"(predicted_return={live.predicted_return:+.4%}, confidence={live.confidence:.2f}, "
            f"entry={live.entry_price:.2f}, stop_loss={live.stop_loss}, target={live.target}, "
            f"trained on {live.n_train_rows:,}/{live.n_total_rows_available:,} rows)"
        )
    except Exception as e:
        # A prediction failure after a successful data pull is a real
        # problem (something about the new row broke feature engineering
        # or model fitting) but NOT a reason to have silently corrupted
        # the series -- the row is already safely upserted above. Alert,
        # don't re-raise, so cron doesn't treat "today's data is fine but
        # prediction broke" as a Kite/data outage.
        send_alert(f"New {commodity} row upserted OK but generate_live_signal() failed: {e!r}")
        return

    logger.info(f"=== EOD job done ({commodity}) ===")


def run_scheduler(commodity: str | None = None) -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        run_once, kwargs={"commodity": commodity},
        trigger=CronTrigger(day_of_week="mon-fri", hour=23, minute=45, timezone="Asia/Kolkata"),
        id="eod_update", misfire_grace_time=3600,
    )
    logger.info("Scheduler started -- will run weekdays at 23:45 IST. Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Run a single update now and exit (for external cron).")
    parser.add_argument("--commodity", default=None, help=f"Defaults to settings.mcx_commodity ({settings.mcx_commodity}).")
    args = parser.parse_args()

    if args.once:
        run_once(commodity=args.commodity)
    else:
        run_scheduler(commodity=args.commodity)


if __name__ == "__main__":
    main()
