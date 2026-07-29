"""
check_heartbeat.py

Dead-man's-switch check for run_eod_job.py -- closes the "no dead-man's-
switch check" gap flagged in EOD_JOB.md. This is deliberately a SEPARATE
script from run_eod_job.py, meant to run on its OWN cron schedule (e.g.
once a day, mid-morning IST -- after the EOD job should already have run
the previous night). A check for "did the job run" that lives inside the
job itself can never fire if the job's process died, crashed the Python
interpreter, or was never scheduled at all -- an external, independently
scheduled watcher is the only thing that can actually catch that class of
failure.

What it does
------------
1. Reads the heartbeat file written by run_eod_job.py::write_heartbeat()
   after every run (success, no-data/holiday, or error) -- path configurable
   via EOD_HEARTBEAT_PATH in .env, default logs/heartbeat.json.
2. If the file is missing entirely: alerts "EOD job has apparently never
   run" (or its heartbeat file was deleted).
3. If the file exists but its timestamp is older than
   EOD_HEARTBEAT_MAX_AGE_HOURS (default 96h / 4 days -- generous enough to
   cross a weekend + one holiday without a false alarm): alerts that the
   job appears to have stopped running.
4. If the last recorded status was "error": alerts, even if the timestamp
   itself is fresh -- a job that's running on schedule but failing every
   time is exactly what this check exists to catch, not just a job that
   stopped running entirely.

This reuses run_eod_job.py's send_alert() (same CRITICAL log + optional
Slack webhook via ALERT_WEBHOOK_URL) rather than a second alerting
implementation.

Suggested crontab line (09:00 IST daily, independent of the job's own
23:45 IST schedule):

    0 9 * * *  cd /path/to/main && venv/bin/python check_heartbeat.py >> logs/heartbeat_check.log 2>&1
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from config.settings import settings
from run_eod_job import HEARTBEAT_PATH, logger, send_alert


def check_heartbeat() -> bool:
    """Returns True if everything looks healthy, False if an alert was
    raised. Exposed as a function (rather than only a __main__ block) so
    tests can call it directly against a temp heartbeat path."""
    if not HEARTBEAT_PATH.exists():
        send_alert(
            f"check_heartbeat: no heartbeat file found at {HEARTBEAT_PATH} -- "
            f"the EOD job has apparently never run successfully (or its "
            f"heartbeat file was deleted). Check cron/scheduler setup."
        )
        return False

    try:
        payload = json.loads(HEARTBEAT_PATH.read_text())
        status = payload["status"]
        message = payload["message"]
        timestamp = dt.datetime.fromisoformat(payload["timestamp"])
    except Exception as e:
        send_alert(f"check_heartbeat: heartbeat file at {HEARTBEAT_PATH} is unreadable/malformed: {e!r}")
        return False

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - timestamp
    max_age = dt.timedelta(hours=settings.eod_heartbeat_max_age_hours)

    if age > max_age:
        send_alert(
            f"check_heartbeat: last heartbeat is {age} old (max allowed "
            f"{max_age}) -- EOD job appears to have stopped running. "
            f"Last recorded status: {status!r} ({message})."
        )
        return False

    if status == "error":
        send_alert(
            f"check_heartbeat: EOD job's most recent run ({timestamp.isoformat()}) "
            f"ended in error: {message}"
        )
        return False

    logger.info(f"check_heartbeat: OK -- last run {timestamp.isoformat()} ({status}): {message}")
    return True


if __name__ == "__main__":
    healthy = check_heartbeat()
    sys.exit(0 if healthy else 1)
