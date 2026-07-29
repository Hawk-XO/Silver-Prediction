"""
data/db.py

MySQL storage layer for MCX Silver OHLCV data. One table, one schema,
regardless of where the data came from (manual bhavcopy CSV, Kite Connect
API, or the COMEX+USDINR proxy) — the `source` column records provenance so
you can always tell which rows are exact real prices vs. an approximation.

Schema
------
mcx_silver_ohlcv(
    date            DATE,           -- trading date (IST)
    contract        VARCHAR(64),    -- e.g. 'SILVERMIC_26FEB2027'
    open, high, low, close  DOUBLE,
    volume          BIGINT NULL,
    open_interest   BIGINT NULL,
    source          VARCHAR(32),    -- 'manual_csv' | 'kite_api' | 'proxy'
    ingested_at     TIMESTAMP,      -- when this row was last written
    PRIMARY KEY (date, contract)
)

The primary key on (date, contract) is what makes ingestion idempotent —
running the same folder twice, or the daily EOD job on a day already stored,
overwrites rather than duplicates.

live_signals(
    date                    DATE,          -- signal date (IST)
    commodity               VARCHAR(32),   -- e.g. 'SILVERMIC'
    signal                  VARCHAR(8),    -- 'BUY' | 'SELL' | 'HOLD'
    predicted_return        DOUBLE,
    confidence              DOUBLE NULL,   -- NULL when the model returned NaN
    entry_price              DOUBLE,
    stop_loss               DOUBLE NULL,
    target                  DOUBLE NULL,
    n_train_rows            INTEGER,
    n_total_rows_available  INTEGER,
    generated_at            TIMESTAMP,     -- when this row was last written
    PRIMARY KEY (date, commodity)
)

One row per (date, commodity) -- the daily EOD job upserts today's signal
here so the UI (and anything else) can read "what did we actually say on
day X" after the fact, instead of only ever seeing the live in-memory
signal from the moment run_eod_job.py ran.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import (
    Column, Date, Double, BigInteger, Integer, String, TIMESTAMP, create_engine, text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.mysql import insert as mysql_insert

from config.settings import settings

Base = declarative_base()


class SilverOHLCV(Base):
    __tablename__ = "mcx_silver_ohlcv"

    date = Column(Date, primary_key=True)
    contract = Column(String(64), primary_key=True)
    open = Column(Double, nullable=False)
    high = Column(Double, nullable=False)
    low = Column(Double, nullable=False)
    close = Column(Double, nullable=False)
    volume = Column(BigInteger, nullable=True)
    open_interest = Column(BigInteger, nullable=True)
    source = Column(String(32), nullable=False)
    ingested_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"),
                          onupdate=text("CURRENT_TIMESTAMP"))


class LiveSignalRecord(Base):
    __tablename__ = "live_signals"

    date = Column(Date, primary_key=True)
    commodity = Column(String(32), primary_key=True)
    signal = Column(String(8), nullable=False)
    predicted_return = Column(Double, nullable=False)
    confidence = Column(Double, nullable=True)
    entry_price = Column(Double, nullable=False)
    stop_loss = Column(Double, nullable=True)
    target = Column(Double, nullable=True)
    n_train_rows = Column(Integer, nullable=False)
    n_total_rows_available = Column(Integer, nullable=False)
    generated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"),
                           onupdate=text("CURRENT_TIMESTAMP"))


@dataclass(frozen=True)
class LiveSignalRow:
    """Plain-data shape that upsert_live_signal() writes -- kept separate
    from signals.live_predict.LiveSignal so this module doesn't need to
    import that module's runtime dependencies just to describe a row."""
    date: dt.date
    commodity: str
    signal: str
    predicted_return: float
    confidence: float | None
    entry_price: float
    stop_loss: float | None
    target: float | None
    n_train_rows: int
    n_total_rows_available: int


def live_signal_to_row(live, commodity: str) -> LiveSignalRow:
    """Pure transform from signals.live_predict.LiveSignal to LiveSignalRow.
    No DB/network access -- kept separate and pure so it's cheaply testable
    (see tests/test_live_signals.py). Two things it normalizes:
      - pandas.Timestamp -> plain datetime.date (MySQL DATE column).
      - NaN confidence -> None (a model can return NaN confidence in some
        edge cases; storing NaN in a DOUBLE column round-trips inconsistently
        across drivers, so None is the honest, portable choice).
    """
    date_val = live.date
    if isinstance(date_val, pd.Timestamp):
        date_val = date_val.date()

    confidence = live.confidence
    if confidence is not None and isinstance(confidence, float) and math.isnan(confidence):
        confidence = None

    return LiveSignalRow(
        date=date_val,
        commodity=commodity,
        signal=live.signal,
        predicted_return=live.predicted_return,
        confidence=confidence,
        entry_price=live.entry_price,
        stop_loss=live.stop_loss,
        target=live.target,
        n_train_rows=live.n_train_rows,
        n_total_rows_available=live.n_total_rows_available,
    )


def get_engine(echo: bool = False):
    """Create a SQLAlchemy engine from config/settings.py.
    pool_pre_ping avoids stale-connection errors after the DB has been idle
    (common for a script that only runs once a day)."""
    return create_engine(settings.mysql_url, echo=echo, pool_pre_ping=True)


def init_db(engine=None) -> None:
    """Create the table if it doesn't exist yet. Safe to call every run."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def upsert_ohlcv(df: pd.DataFrame, source: str, engine=None) -> int:
    """
    Upsert a DataFrame of OHLCV rows into mcx_silver_ohlcv.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by date, must have columns: contract, open, high, low, close,
        and optionally volume, open_interest. This is exactly the schema
        produced by data.batch_bhavcopy_processor.process_bhavcopy_folder()
        and data.contract_roll.build_continuous_series().
    source : str
        'manual_csv' | 'kite_api' | 'proxy' — provenance tag.
    engine : sqlalchemy.Engine, optional
        Reuses a shared engine if provided (e.g. from a long-running job);
        otherwise creates one for this call.

    Returns
    -------
    int
        Number of rows upserted.
    """
    if df.empty:
        return 0

    engine = engine or get_engine()
    init_db(engine)

    records = df.reset_index().rename(columns={df.index.name or "index": "date"})
    records["date"] = pd.to_datetime(records["date"]).dt.tz_localize(None).dt.date
    records["source"] = source

    required = ["date", "contract", "open", "high", "low", "close"]
    missing = [c for c in required if c not in records.columns]
    if missing:
        raise ValueError(f"upsert_ohlcv: input is missing required columns: {missing}")

    for optional_col in ("volume", "open_interest"):
        if optional_col not in records.columns:
            records[optional_col] = None

    cols = required + ["volume", "open_interest", "source"]
    rows = records[cols].to_dict(orient="records")

    with engine.begin() as conn:
        for row in rows:
            stmt = mysql_insert(SilverOHLCV).values(**row)
            stmt = stmt.on_duplicate_key_update(
                open=stmt.inserted.open,
                high=stmt.inserted.high,
                low=stmt.inserted.low,
                close=stmt.inserted.close,
                volume=stmt.inserted.volume,
                open_interest=stmt.inserted.open_interest,
                source=stmt.inserted.source,
            )
            conn.execute(stmt)

    return len(rows)


def load_ohlcv(contract: str | None = None, source: str | None = None, engine=None) -> pd.DataFrame:
    """
    Read stored OHLCV data back out as a DataFrame, indexed by date.
    Pass `contract` to filter to one contract (e.g. for calibration/checks);
    pass `source` to filter to one provenance ('manual_csv' | 'kite_api' |
    'proxy'), e.g. to pull only real kite_api rows for calibrating the proxy.
    Omit both to get everything (what build_continuous_series() consumes).
    """
    engine = engine or get_engine()
    query = "SELECT date, contract, open, high, low, close, volume, open_interest, source FROM mcx_silver_ohlcv"
    conditions = []
    params = {}
    if contract:
        conditions.append("contract = :contract")
        params["contract"] = contract
    if source:
        conditions.append("source = :source")
        params["source"] = source
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date"

    df = pd.read_sql(text(query), engine, params=params, parse_dates=["date"])
    return df.set_index("date")


def get_latest_date(commodity: str | None = None, engine=None):
    """
    Return the most recent `date` stored for a commodity (matched by prefix
    before the first underscore in `contract`, same convention as
    data.pipeline_common.load_real_features), or None if the table is empty
    / doesn't exist yet.

    Deliberately a single MAX(date) query rather than loading the whole
    table (load_ohlcv) just to check freshness -- this is meant to be cheap
    enough to call once on every UI page load.
    """
    engine = engine or get_engine()
    query = "SELECT MAX(date) AS latest FROM mcx_silver_ohlcv"
    params = {}
    if commodity:
        query += " WHERE SUBSTRING_INDEX(contract, '_', 1) = :commodity"
        params["commodity"] = commodity

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params).scalar()
    except Exception:
        # Table doesn't exist yet (fresh DB) -- treat as "no data stored".
        return None

    if result is None:
        return None
    return pd.Timestamp(result)


def upsert_live_signal(row: LiveSignalRow, engine=None) -> None:
    """Upsert a single day's live signal into `live_signals`, keyed on
    (date, commodity) -- same idempotent-rerun pattern as upsert_ohlcv()."""
    engine = engine or get_engine()
    init_db(engine)

    values = dict(
        date=row.date,
        commodity=row.commodity,
        signal=row.signal,
        predicted_return=row.predicted_return,
        confidence=row.confidence,
        entry_price=row.entry_price,
        stop_loss=row.stop_loss,
        target=row.target,
        n_train_rows=row.n_train_rows,
        n_total_rows_available=row.n_total_rows_available,
    )

    with engine.begin() as conn:
        stmt = mysql_insert(LiveSignalRecord).values(**values)
        stmt = stmt.on_duplicate_key_update(
            signal=stmt.inserted.signal,
            predicted_return=stmt.inserted.predicted_return,
            confidence=stmt.inserted.confidence,
            entry_price=stmt.inserted.entry_price,
            stop_loss=stmt.inserted.stop_loss,
            target=stmt.inserted.target,
            n_train_rows=stmt.inserted.n_train_rows,
            n_total_rows_available=stmt.inserted.n_total_rows_available,
        )
        conn.execute(stmt)


def load_live_signals(commodity: str | None = None, engine=None) -> pd.DataFrame:
    """Read stored live signals back out as a DataFrame, indexed by date,
    most recent first. Pass `commodity` to filter to one commodity."""
    engine = engine or get_engine()
    query = (
        "SELECT date, commodity, signal, predicted_return, confidence, "
        "entry_price, stop_loss, target, n_train_rows, n_total_rows_available, "
        "generated_at FROM live_signals"
    )
    params = {}
    if commodity:
        query += " WHERE commodity = :commodity"
        params["commodity"] = commodity
    query += " ORDER BY date DESC"

    df = pd.read_sql(text(query), engine, params=params, parse_dates=["date", "generated_at"])
    return df.set_index("date")
