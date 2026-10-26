"""Early-warning features.

Every feature at date ``t`` uses information available at the close of
``t`` only (no look-ahead). Feature families:

* Price run-up and acceleration - Greenwood, Shleifer & You (2019) find
  that sharp run-ups raise crash probability and that the *acceleration*
  of the run-up helps tell crashes from continued booms.
* Volatility regime - realised volatility, vol-of-vol, short/long vol ratio,
  higher moments of returns.
* Trend / stretch - distance to the 200-day moving average, drawdown
  from the 1-year high.
* Implied volatility (if VIX is available) - level, change, and the gap
  between implied and realised volatility (variance-risk-premium proxy).
* Yield curve (if 10y and 3m rates are available) - term spread and its
  change; inversions historically precede US recessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret_1m": "Log return over 1 month (21d)",
    "ret_3m": "Log return over 3 months (63d)",
    "ret_6m": "Log return over 6 months (126d)",
    "ret_12m": "Log return over 12 months (252d)",
    "runup_24m": "Log return over 24 months (504d) - run-up size",
    "accel": "Run-up acceleration: last 6m return minus average 6m return of the prior 18m",
    "vol_1m": "Annualised realised volatility, 21d",
    "vol_3m": "Annualised realised volatility, 63d",
    "vol_ratio": "Realised vol 21d / realised vol 252d",
    "vol_of_vol": "Std-dev of the 21d realised vol over 63d",
    "skew_3m": "Skewness of daily returns, 63d",
    "kurt_3m": "Excess kurtosis of daily returns, 63d",
    "dist_ma200": "Price / 200-day moving average - 1",
    "dd_1y": "Drawdown from the 252-day high",
    "vix": "VIX level",
    "vix_chg_1m": "VIX change over 21d",
    "vrp": "VIX minus 21d realised vol (vol points)",
    "term_spread": "US 10y yield minus 3m bill yield (pp)",
    "term_spread_chg_3m": "Change in the term spread over 63d (pp)",
}

BASELINE_FEATURES = ["vol_1m"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    logp = np.log(close)
    r = logp.diff()

    f = pd.DataFrame(index=df.index)
    f["ret_1m"] = logp.diff(21)
    f["ret_3m"] = logp.diff(63)
    f["ret_6m"] = logp.diff(126)
    f["ret_12m"] = logp.diff(252)
    f["runup_24m"] = logp.diff(504)
    f["accel"] = f["ret_6m"] - (f["runup_24m"] - f["ret_6m"]) / 3.0

    vol21 = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_1m"] = vol21
    f["vol_3m"] = r.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f["vol_ratio"] = vol21 / (r.rolling(252).std() * np.sqrt(TRADING_DAYS))
    f["vol_of_vol"] = vol21.rolling(63).std()
    f["skew_3m"] = r.rolling(63).skew()
    f["kurt_3m"] = r.rolling(63).kurt()

    f["dist_ma200"] = close / close.rolling(200).mean() - 1
    f["dd_1y"] = close / close.rolling(252).max() - 1

    if df["vix"].notna().mean() > 0.5:
        vix = df["vix"]
        f["vix"] = vix
        f["vix_chg_1m"] = vix.diff(21)
        f["vrp"] = vix - 100 * vol21

    if df["tnx"].notna().mean() > 0.5 and df["irx"].notna().mean() > 0.5:
        spread = df["tnx"] - df["irx"]
        f["term_spread"] = spread
        f["term_spread_chg_3m"] = spread.diff(63)

    return f.replace([np.inf, -np.inf], np.nan)
