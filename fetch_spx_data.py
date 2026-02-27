"""
SPX Data Fetcher - Yahoo Finance (yfinance)
===========================================
Pulls daily OHLCV + VIX + cross-asset closes for the S&P 500 and writes
spx_data.csv.  Supports full refresh or incremental update.

Cross-asset tickers fetched alongside SPX:
  ^VIX3M   -- CBOE 3-Month Volatility (VIX term structure; starts ~2007-12)
  HYG      -- iShares High Yield Bond ETF (credit stress; starts 2007-04)
  ^TNX     -- 10-Year Treasury Yield (%)
  ^IRX     -- 3-Month T-Bill Rate (%)
  DX-Y.NYB -- US Dollar Index
  GLD      -- SPDR Gold ETF

Missing dates or tickers with short history are stored as empty strings and
handled as NaN in calculate_indicator.py.

If the existing CSV is missing any cross-asset columns, they are automatically
backfilled on the first incremental run (one-time migration).

Usage:
    python fetch_spx_data.py            # incremental update (fast)
    python fetch_spx_data.py --full     # full 1990-to-today refresh (9103 rows)
"""

import csv
import sys
import subprocess
import datetime
from pathlib import Path

import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "spx_data.csv"
INDICATOR_SCRIPT = BASE_DIR / "calculate_indicator.py"
BACKTEST_SCRIPT  = BASE_DIR / "backtest_analog.py"

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"
START_DATE = "1990-01-01"

# Cross-asset tickers: {csv_column_name: yfinance_ticker}
CROSS_ASSET = {
    "VIX3M_Close": "^VIX3M",    # CBOE 3-Month Volatility (VIX term structure)
    "HYG_Close":   "HYG",       # High Yield Bond ETF (credit stress proxy)
    "TNX_Close":   "^TNX",      # 10-Year Treasury Yield (%)
    "IRX_Close":   "^IRX",      # 3-Month T-Bill Rate (%)
    "DXY_Close":   "DX-Y.NYB",  # US Dollar Index
    "Gold_Close":  "GLD",       # SPDR Gold ETF
    "SKEW_Close":  "^SKEW",     # CBOE SKEW Index (tail risk; 100-165 range)
}

FIELDNAMES = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "VIX_Close",
] + list(CROSS_ASSET.keys())


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_spx_rows(from_date: str, to_date: str) -> list:
    """Fetch daily OHLCV bars for ^GSPC. Returns list of row dicts."""
    print(f"  Fetching SPX {from_date} to {to_date}...")
    df = yf.Ticker(SPX_TICKER).history(
        start=from_date, end=to_date, interval="1d", auto_adjust=True
    )
    if df.empty:
        print("  No SPX data returned.")
        return []
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "Date":   dt.strftime("%Y-%m-%d"),
            "Open":   round(float(row["Open"]),  2),
            "High":   round(float(row["High"]),  2),
            "Low":    round(float(row["Low"]),   2),
            "Close":  round(float(row["Close"]), 2),
            "Volume": int(row["Volume"]),
        })
    rows.sort(key=lambda r: r["Date"])
    print(f"  {len(rows)} SPX bars retrieved.")
    return rows


def fetch_closes(ticker: str, from_date: str, to_date: str) -> dict:
    """Fetch daily closes for any ticker. Returns {date_str: float}."""
    df = yf.Ticker(ticker).history(
        start=from_date, end=to_date, interval="1d", auto_adjust=True
    )
    if df.empty:
        return {}
    return {
        dt.strftime("%Y-%m-%d"): round(float(row["Close"]), 4)
        for dt, row in df.iterrows()
    }


def fetch_vix_map(from_date: str, to_date: str) -> dict:
    print(f"  Fetching VIX {from_date} to {to_date}...")
    m = fetch_closes(VIX_TICKER, from_date, to_date)
    print(f"  {len(m)} VIX bars retrieved.")
    return m


def fetch_cross_asset_map(from_date: str, to_date: str) -> dict:
    """
    Fetch all cross-asset closes.
    Returns {date_str: {col_name: value}}.
    Silently skips any ticker that fails or has no data.
    """
    combined = {}
    for col_name, ticker in CROSS_ASSET.items():
        print(f"  Fetching {col_name:<14} ({ticker})...", end=" ", flush=True)
        try:
            closes = fetch_closes(ticker, from_date, to_date)
            for date, val in closes.items():
                combined.setdefault(date, {})[col_name] = val
            print(f"{len(closes)} bars")
        except Exception as e:
            print(f"FAILED ({e})")
    return combined


# ── Merge helpers ─────────────────────────────────────────────────────────────

def merge_all(rows: list, vix_map: dict, cross_map: dict) -> list:
    """Attach VIX and all cross-asset closes to each row."""
    for row in rows:
        row["VIX_Close"] = vix_map.get(row["Date"], "")
        day = cross_map.get(row["Date"], {})
        for col in CROSS_ASSET:
            row[col] = day.get(col, "")
    return rows


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def read_existing_csv() -> tuple:
    """Read existing CSV. Returns (rows, last_date, existing_columns_set)."""
    if not CSV_FILE.exists():
        return [], None, set()
    rows = []
    fieldnames = []
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    if not rows:
        return [], None, set()
    return rows, rows[-1]["Date"], set(fieldnames)


def write_csv(rows: list):
    """Write rows to CSV using the full FIELDNAMES set."""
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill any missing columns with empty string
            for col in FIELDNAMES:
                if col not in row:
                    row[col] = ""
            writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    full_refresh = "--full" in sys.argv
    today    = datetime.date.today().isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    if full_refresh:
        print(f"Full refresh: {START_DATE} to {today}")
        spx_rows  = fetch_spx_rows(START_DATE, tomorrow)
        vix_map   = fetch_vix_map(START_DATE, tomorrow)
        cross_map = fetch_cross_asset_map(START_DATE, tomorrow)
        rows      = merge_all(spx_rows, vix_map, cross_map)
        write_csv(rows)
        print(f"Wrote {len(rows)} rows to {CSV_FILE}")

    else:
        existing_rows, last_date, existing_cols = read_existing_csv()

        # ── One-time migration: backfill missing columns ──────────────────────
        missing_cols = set(FIELDNAMES) - existing_cols - {"Date"}
        if existing_rows and missing_cols:
            print(f"Missing columns detected: {sorted(missing_cols)}")
            print("Backfilling full history...")
            vix_map   = fetch_vix_map(START_DATE, tomorrow)
            cross_map = fetch_cross_asset_map(START_DATE, tomorrow)
            existing_rows = merge_all(existing_rows, vix_map, cross_map)
            write_csv(existing_rows)
            print(f"Backfilled {len(existing_rows)} rows.")
            last_date = existing_rows[-1]["Date"]

        # ── Incremental update ────────────────────────────────────────────────
        if last_date:
            next_day = (
                datetime.date.fromisoformat(last_date) +
                datetime.timedelta(days=1)
            ).isoformat()
            print(f"Incremental update: {next_day} to {today}")
            if next_day > today:
                print("  Already up to date!")
            else:
                spx_rows = fetch_spx_rows(next_day, tomorrow)
                if spx_rows:
                    vix_map   = fetch_vix_map(next_day, tomorrow)
                    cross_map = fetch_cross_asset_map(next_day, tomorrow)
                    new_rows  = merge_all(spx_rows, vix_map, cross_map)

                    existing_dates = {r["Date"] for r in existing_rows}
                    appended = 0
                    for row in new_rows:
                        if row["Date"] not in existing_dates:
                            existing_rows.append(row)
                            appended += 1
                    existing_rows.sort(key=lambda r: r["Date"])
                    write_csv(existing_rows)
                    print(f"Appended {appended} new rows (total: {len(existing_rows)})")
                else:
                    print("  No new trading days found.")
        else:
            print(f"No existing CSV — full fetch: {START_DATE} to {today}")
            spx_rows  = fetch_spx_rows(START_DATE, tomorrow)
            vix_map   = fetch_vix_map(START_DATE, tomorrow)
            cross_map = fetch_cross_asset_map(START_DATE, tomorrow)
            rows      = merge_all(spx_rows, vix_map, cross_map)
            write_csv(rows)
            print(f"Wrote {len(rows)} rows to {CSV_FILE}")

    # Update calibration and regenerate indicator
    # Skip in CI mode (--no-compute): CI runs v3/calculate_indicator.py separately
    if "--no-compute" not in sys.argv:
        print("\nRunning backtest_analog.py (updating calibration)...")
        subprocess.run([sys.executable, str(BACKTEST_SCRIPT)], cwd=str(BASE_DIR))

        print("\nRunning calculate_indicator.py...")
        subprocess.run([sys.executable, str(INDICATOR_SCRIPT)], cwd=str(BASE_DIR))
    print("\nDone! Refresh the dashboard.")


if __name__ == "__main__":
    main()
