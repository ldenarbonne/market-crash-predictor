"""Refresh the market-data snapshots stored in data/.

Run locally:        python scripts/update_data.py
Run for one index:  python scripts/update_data.py --tickers ^GSPC
Used weekly by .github/workflows/update-data.yml so the Streamlit app
can read data from the repository instead of calling Yahoo Finance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crashlab.data import DATA_DIR, INDICES, fetch_yahoo, snapshot_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["^GSPC", "^NDX", "^STOXX50E", "^FCHI"])
    parser.add_argument("--start", default="1985-01-01")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    failures = 0
    for tic in args.tickers:
        name = next((k for k, v in INDICES.items() if v == tic), tic)
        try:
            df = fetch_yahoo(tic, args.start)
            df.to_csv(snapshot_path(tic), float_format="%.4f")
            print(f"OK   {name:<30} {len(df):>6} rows  {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as exc:  # keep going for the other indices
            failures += 1
            print(f"FAIL {name:<30} {exc}")
        time.sleep(2)  # be gentle with Yahoo
    return 1 if failures == len(args.tickers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
