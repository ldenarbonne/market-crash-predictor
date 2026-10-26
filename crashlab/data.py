"""Market data loading.

The model needs one equity index close series and, optionally, three
"macro / sentiment" series:

* ``close`` : equity index level (required)
* ``vix``   : CBOE VIX index level (implied volatility, in vol points)
* ``tnx``   : US 10-year Treasury yield, in percent (Yahoo ``^TNX``)
* ``irx``   : US 13-week T-bill yield, in percent (Yahoo ``^IRX``)

Data sources, in order of preference in the app:

1. a CSV snapshot committed in ``data/`` (refreshed by a GitHub Action,
   which avoids hitting Yahoo from Streamlit Community Cloud);
2. live download from Yahoo Finance through ``yfinance``;
3. a user-uploaded CSV;
4. a synthetic regime-switching market (demo / offline).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

INDICES: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones Industrial Average": "^DJI",
    "Euro Stoxx 50": "^STOXX50E",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
}

MACRO_TICKERS: dict[str, str] = {"vix": "^VIX", "tnx": "^TNX", "irx": "^IRX"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLUMNS = ["close", "vix", "tnx", "irx"]


def snapshot_path(index_ticker: str) -> Path:
    """File name used for the committed snapshot of an index."""
    safe = index_ticker.replace("^", "").replace("/", "_")
    return DATA_DIR / f"market_{safe}.csv"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a market DataFrame: DatetimeIndex, known columns, sorted."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNS].astype(float)
    # Macro series trade on slightly different calendars: forward-fill a few days.
    df[["vix", "tnx", "irx"]] = df[["vix", "tnx", "irx"]].ffill(limit=5)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def load_snapshot(index_ticker: str) -> pd.DataFrame | None:
    """Load the committed CSV snapshot, if it exists."""
    path = snapshot_path(index_ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _clean(df)


def fetch_yahoo(index_ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Download index closes and macro series from Yahoo Finance.

    Raises ``RuntimeError`` when Yahoo returns no data for the index
    (rate limit, network error, wrong ticker).
    """
    import yfinance as yf  # imported lazily: optional at test time

    tickers = [index_ticker, *MACRO_TICKERS.values()]
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (rate limit or network issue).")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in level0 else "Adj Close"
        closes = raw[field]
    else:  # single ticker fallback
        closes = raw[["Close"]].rename(columns={"Close": index_ticker})

    out = pd.DataFrame(index=closes.index)
    out["close"] = closes.get(index_ticker)
    for name, tic in MACRO_TICKERS.items():
        out[name] = closes.get(tic)
    if out["close"].dropna().empty:
        raise RuntimeError(f"No price data returned for {index_ticker}.")
    return _clean(out)


def parse_csv(file_like) -> pd.DataFrame:
    """Parse a user CSV.

    Expected columns: a date column (first column or named ``date``),
    ``close`` (required) and optionally ``vix``, ``tnx``, ``irx``.
    Column names are case-insensitive; ``Adj Close`` / ``price`` are
    accepted as aliases of ``close``.
    """
    if isinstance(file_like, (bytes, bytearray)):
        file_like = io.BytesIO(file_like)
    df = pd.read_csv(file_like)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.set_index(date_col)
    aliases = {"adj close": "close", "adj_close": "close", "price": "close", "px_last": "close"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and "close" not in df.columns})
    if "close" not in df.columns:
        raise ValueError("The CSV must contain a 'close' column.")
    return _clean(df)


def data_summary(df: pd.DataFrame) -> dict:
    """Small description of a market DataFrame for display."""
    return {
        "start": df.index.min().date().isoformat(),
        "end": df.index.max().date().isoformat(),
        "rows": int(len(df)),
        "has_vix": bool(df["vix"].notna().mean() > 0.5),
        "has_rates": bool(df["tnx"].notna().mean() > 0.5 and df["irx"].notna().mean() > 0.5),
    }
