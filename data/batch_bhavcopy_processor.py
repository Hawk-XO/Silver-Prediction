"""
data/batch_bhavcopy_processor.py

Automates the ONLY manual step left in the pipeline: turning a folder of
manually-downloaded MCX bhavcopy/historical-data CSV files into a single,
clean, multi-contract Silver dataset ready for build_continuous_series().

Why this step is manual: MCX's website disallows automated/bot access
(robots.txt), so files must be downloaded by hand from either:
  - https://www.mcxindia.com/market-data/bhavcopy  (one file per day, all commodities)
  - https://www.mcxindia.com/market-data/historical-data  (contract-specific, date range)
  - Investing.com's Silver Futures India historical data export

Everything downstream of "files sitting in a folder" is fully automated here.

Usage
-----
1. Download bhavcopy/historical CSVs from MCX or Investing.com, however many
   days/months you've collected. Naming doesn't matter.
2. Put them all in one folder, e.g. /home/claude/raw_bhavcopy/
3. Run:
       from data.batch_bhavcopy_processor import process_bhavcopy_folder
       df = process_bhavcopy_folder("/home/claude/raw_bhavcopy/", commodity="SILVER")
4. `df` is now in the same long-format schema build_continuous_series() expects
   (columns: contract, open, high, low, close, volume, open_interest, indexed by date).
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from data.mcx_loader import _build_rename_map, REQUIRED_COLUMNS, OPTIONAL_COLUMNS

# MCX bhavcopy files cover every commodity traded that day. This is how we
# filter down to just Silver contracts regardless of which vendor format the
# file came in (MCX official bhavcopy vs. Investing.com export have
# different column names for the "which commodity/contract" field).
_COMMODITY_COLUMN_ALIASES = [
    "commodity", "symbol", "instrument", "instrumentname", "commodityhead", "contract",
]

# A commodity name alone (e.g. "SILVER") is NOT a unique contract identifier —
# multiple expiry months trade simultaneously. We combine commodity + expiry
# to build a true per-contract identifier, which is what contract_roll.py
# needs to correctly detect roll dates.
_EXPIRY_COLUMN_ALIASES = ["expirydate", "expiry_date", "expiry", "duedate", "expiry date"]


def _find_commodity_column(columns: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for alias in _COMMODITY_COLUMN_ALIASES:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _find_expiry_column(columns: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}
    for alias in _EXPIRY_COLUMN_ALIASES:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _load_one_bhavcopy_file(path: str, commodity_filter: str, exact_match: bool = True) -> pd.DataFrame:
    """Load a single raw bhavcopy/historical CSV, filter to the target
    commodity (e.g. 'SILVER'), and rename columns to our canonical schema.
    Returns an empty DataFrame (not an error) if the commodity isn't present
    in this particular file — bhavcopy files cover all commodities, so most
    rows in any given file won't be Silver."""
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"[batch_bhavcopy_processor] Skipping unreadable file {path}: {e}")
        return pd.DataFrame()

    commodity_col = _find_commodity_column(list(raw.columns))
    if commodity_col is None:
        print(
            f"[batch_bhavcopy_processor] Skipping {path}: could not find a "
            f"commodity/symbol column to filter on. Columns present: {list(raw.columns)}"
        )
        return pd.DataFrame()

    commodity_values = raw[commodity_col].astype(str).str.upper().str.strip()
    if exact_match:
        mask = commodity_values == commodity_filter.upper().strip()
    else:
        mask = commodity_values.str.contains(commodity_filter.upper(), na=False)
    filtered = raw[mask].copy()
    if filtered.empty:
        return pd.DataFrame()

    # Build the true contract identifier BEFORE renaming: commodity name +
    # expiry (if an expiry column exists) — e.g. "SILVER_30-Jul-2024" — so
    # different expiry months of the same commodity aren't collapsed into
    # one contract, which would silently break roll-date detection later.
    expiry_col = _find_expiry_column(list(filtered.columns))
    if expiry_col is not None:
        filtered["contract"] = (
            filtered[commodity_col].astype(str).str.strip() + "_" + filtered[expiry_col].astype(str).str.strip()
        )
    else:
        # No separate expiry column — assume the commodity/symbol column
        # already encodes the contract uniquely (e.g. "SILVER 05DEC24").
        filtered["contract"] = filtered[commodity_col].astype(str).str.strip()

    rename_map = _build_rename_map(list(filtered.columns))
    # Don't let the generic rename map clobber the contract column we just
    # built deliberately above.
    rename_map = {k: v for k, v in rename_map.items() if v != "contract"}
    filtered = filtered.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in filtered.columns]
    if missing:
        print(
            f"[batch_bhavcopy_processor] Skipping {path}: missing required "
            f"column(s) {missing} after filtering to {commodity_filter}. "
            f"Available: {list(filtered.columns)}"
        )
        return pd.DataFrame()

    keep_cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in filtered.columns]
    return filtered[keep_cols]


def process_bhavcopy_folder(
    folder_path: str,
    commodity: str = "SILVER",
    file_pattern: str = "*.csv",
    tz: str = "Asia/Kolkata",
    exact_match: bool = True,
) -> pd.DataFrame:
    """
    Consolidate a folder of manually-downloaded MCX bhavcopy/historical CSVs
    into a single multi-contract Silver dataset.

    Parameters
    ----------
    folder_path : str
        Folder containing the raw downloaded CSV files (any naming).
    commodity : str
        Commodity to filter for (default 'SILVER').
    file_pattern : str
        Glob pattern for files to process (default all .csv files).
    tz : str
        Timezone to localize dates to.
    exact_match : bool
        If True (default), only rows where the commodity column EXACTLY
        equals `commodity` are kept — SILVER will NOT also pull in
        SILVERM or SILVERMIC, since those are distinct products with their
        own contract specs and mixing them into one "continuous series"
        would be incorrect. Set False if you deliberately want substring
        matching (e.g. to inspect what Silver-related products exist in a
        file at all).

    Returns
    -------
    pd.DataFrame
        Long-format, indexed by date, columns: contract, open, high, low,
        close, volume (if available), open_interest (if available).
        Ready to pass directly into contract_roll.build_continuous_series().

    Raises
    ------
    ValueError
        If no files are found, or no Silver rows could be extracted from
        any file in the folder (likely means the commodity filter or file
        format doesn't match what's actually in the folder — inspect one
        file manually to check column names/values).
    """
    paths = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    if not paths:
        raise ValueError(
            f"process_bhavcopy_folder: no files matching '{file_pattern}' found in {folder_path}"
        )

    frames = []
    n_skipped = 0
    for path in paths:
        result = _load_one_bhavcopy_file(path, commodity, exact_match)
        if result.empty:
            n_skipped += 1
            continue
        frames.append(result)

    if not frames:
        raise ValueError(
            f"process_bhavcopy_folder: found {len(paths)} file(s) in {folder_path} "
            f"but extracted zero '{commodity}' rows from any of them. Open one file "
            f"manually and check: (1) does a commodity/symbol column exist, "
            f"(2) does it actually contain '{commodity}' as a value, "
            f"(3) is the file a bhavcopy/historical export at all."
        )

    combined = pd.concat(frames, ignore_index=False)

    combined["date"] = pd.to_datetime(combined["date"])
    if combined["date"].dt.tz is None:
        combined["date"] = combined["date"].dt.tz_localize(tz)
    else:
        combined["date"] = combined["date"].dt.tz_convert(tz)

    combined = combined.sort_values("date").drop_duplicates(subset=["date", "contract"], keep="last")
    combined = combined.set_index("date")

    for col in ["open", "high", "low", "close"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna(subset=["open", "high", "low", "close"])

    print(
        f"[batch_bhavcopy_processor] Processed {len(paths)} file(s): "
        f"{len(paths) - n_skipped} used, {n_skipped} skipped (no matching rows). "
        f"Extracted {len(combined)} '{commodity}' rows across "
        f"{combined['contract'].nunique()} distinct contract(s), "
        f"date range {combined.index.min().date()} to {combined.index.max().date()}."
    )

    return combined
