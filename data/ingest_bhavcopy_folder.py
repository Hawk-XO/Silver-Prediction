"""
data/ingest_bhavcopy_folder.py

Reads every CSV in a folder (manually downloaded from MCX's Historical Data
page — one file per expiry, per PROJECT_NOTES.md's data source), and upserts
the result into MySQL (mcx_silver_ohlcv, source='manual_csv').

Usage
-----
    python -m data.ingest_bhavcopy_folder /path/to/csv/folder
    python -m data.ingest_bhavcopy_folder /path/to/csv/folder --commodity SILVER

Safe to re-run: the (date, contract) primary key means re-ingesting the same
folder (or a folder with overlapping dates) just overwrites, never duplicates.
"""

from __future__ import annotations

import argparse
import sys

from data.batch_bhavcopy_processor import process_bhavcopy_folder
from data.db import upsert_ohlcv, get_engine
from config.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder containing MCX bhavcopy/historical-data CSVs")
    parser.add_argument(
        "--commodity", default=settings.mcx_commodity,
        help=f"Commodity to filter to, exact match (default: {settings.mcx_commodity})",
    )
    args = parser.parse_args()

    print(f"Reading '{args.commodity}' rows from CSVs in {args.folder} ...")
    df = process_bhavcopy_folder(args.folder, commodity=args.commodity)

    if df.empty:
        print(f"No '{args.commodity}' rows found in {args.folder} — nothing to ingest.")
        sys.exit(1)

    engine = get_engine()
    n = upsert_ohlcv(df, source="manual_csv", engine=engine)
    print(f"Upserted {n} rows into mcx_silver_ohlcv (source=manual_csv).")

    n_contracts = df["contract"].nunique()
    print(f"Covers {n_contracts} distinct contract(s), "
          f"date range {df.index.min().date()} to {df.index.max().date()}.")


if __name__ == "__main__":
    main()
