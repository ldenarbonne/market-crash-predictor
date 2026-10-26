"""Target definition and historical crash episodes.

Target
------
For each date ``t`` and horizon ``H`` (trading days), the *forward maximum
drawdown* is

    FMDD_t = min_{t < s <= t+H}  P_s / max_{t <= u <= s} P_u  - 1

i.e. the worst peak-to-trough loss that is *realised inside* the next
``H`` days (the peak may occur after ``t``). The binary crash label is

    y_t = 1  if  FMDD_t <= -threshold,   else 0.

The last ``H`` observations have no label (the future is not yet known).

Episodes
--------
A crash episode is a peak-to-trough decline of at least ``threshold``
where the threshold is breached within ``H`` trading days of the peak
("fast" crash, consistent with the label). Slow bear markets that need
more than ``H`` days to breach the threshold are kept but flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    p = close.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    if n > horizon:
        win = sliding_window_view(p, horizon + 1)  # row t = p[t .. t+H]
        running_peak = np.maximum.accumulate(win, axis=1)
        dd = win / running_peak - 1.0
        out[: n - horizon] = dd[:, 1:].min(axis=1)
    return pd.Series(out, index=close.index, name=f"fmdd_{horizon}")


def crash_label(close: pd.Series, horizon: int, threshold: float) -> pd.Series:
    fmdd = forward_max_drawdown(close, horizon)
    y = (fmdd <= -abs(threshold)).astype(float)
    y[fmdd.isna()] = np.nan
    return y.rename("crash")


def drawdown_episodes(close: pd.Series, threshold: float, horizon: int) -> pd.DataFrame:
    """List drawdown episodes deeper than ``threshold``.

    Columns: peak_date, breach_date (first day below -threshold),
    trough_date, recovery_date (NaT if not recovered), depth,
    days_to_breach (trading days), fast (breach within ``horizon``).
    """
    thr = abs(threshold)
    dates = close.index
    p = close.to_numpy(dtype=float)
    rows = []
    peak_i = 0
    in_ep = False
    breach_i = trough_i = None
    for i in range(1, len(p)):
        if p[i] >= p[peak_i]:
            if in_ep:
                rows.append((peak_i, breach_i, trough_i, i))
                in_ep = False
            peak_i = i
            continue
        dd = p[i] / p[peak_i] - 1
        if not in_ep and dd <= -thr:
            in_ep, breach_i, trough_i = True, i, i
        if in_ep and p[i] < p[trough_i]:
            trough_i = i
    if in_ep:
        rows.append((peak_i, breach_i, trough_i, None))

    records = []
    for pk, br, tr, rc in rows:
        records.append(
            {
                "peak_date": dates[pk],
                "breach_date": dates[br],
                "trough_date": dates[tr],
                "recovery_date": dates[rc] if rc is not None else pd.NaT,
                "depth": p[tr] / p[pk] - 1,
                "days_to_breach": br - pk,
                "fast": (br - pk) <= horizon,
            }
        )
    cols = ["peak_date", "breach_date", "trough_date", "recovery_date", "depth", "days_to_breach", "fast"]
    return pd.DataFrame.from_records(records, columns=cols)
