"""
config/settings.py

Single place every module reads secrets/config from. Loads .env once via
python-dotenv, exposes a Settings object, and treats missing Kite credentials
as "Kite integration disabled" rather than raising — this project must keep
working (MySQL storage, manual CSV ingestion, mcx_proxy) even before the
Zerodha app is verified and KITE_* is filled in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # no-op if .env doesn't exist; real deployments always have one


@dataclass(frozen=True)
class Settings:
    # MySQL
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str

    # Kite Connect — any of these being blank means "not configured yet"
    kite_api_key: str
    kite_api_secret: str
    kite_access_token: str

    # Defaults
    mcx_commodity: str

    # EOD job (run_eod_job.py / check_heartbeat.py)
    eod_max_stale_calendar_days: int
    eod_heartbeat_path: str
    eod_heartbeat_max_age_hours: int

    @property
    def kite_configured(self) -> bool:
        """True once API key + secret + access token are all present.
        Modules should check this before attempting a live Kite call, and
        fall back (e.g. to mcx_proxy) or raise a clear error otherwise."""
        return bool(self.kite_api_key and self.kite_api_secret and self.kite_access_token)

    @property
    def mysql_password_is_placeholder(self) -> bool:
        """True if MYSQL_PASSWORD still looks like an unfilled example value
        rather than a real password. Checked at EOD job startup so a fresh
        clone with an unedited .env fails loudly instead of silently trying
        (and failing) to connect to MySQL. Matches this repo's own example
        defaults plus a few common generic placeholders."""
        known_placeholders = {
            "", "silver_pass", "pick_your_own_password", "changeme",
            "change_me", "password", "your_password_here",
        }
        return self.mysql_password.strip().lower() in known_placeholders

    @property
    def mysql_url(self) -> str:
        """SQLAlchemy connection URL for the configured MySQL database."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


def load_settings() -> Settings:
    return Settings(
        mysql_host=os.getenv("MYSQL_HOST", "localhost"),
        mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
        mysql_user=os.getenv("MYSQL_USER", "root"),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),
        mysql_database=os.getenv("MYSQL_DATABASE", "silver_algo"),
        kite_api_key=os.getenv("KITE_API_KEY", "").strip(),
        kite_api_secret=os.getenv("KITE_API_SECRET", "").strip(),
        kite_access_token=os.getenv("KITE_ACCESS_TOKEN", "").strip(),
        mcx_commodity=os.getenv("MCX_COMMODITY", "SILVERMIC").strip(),
        eod_max_stale_calendar_days=int(os.getenv("EOD_MAX_STALE_CALENDAR_DAYS", "4")),
        eod_heartbeat_path=os.getenv("EOD_HEARTBEAT_PATH", "logs/heartbeat.json").strip(),
        eod_heartbeat_max_age_hours=int(os.getenv("EOD_HEARTBEAT_MAX_AGE_HOURS", "96")),
    )


# Module-level singleton — `from config.settings import settings`
settings = load_settings()
