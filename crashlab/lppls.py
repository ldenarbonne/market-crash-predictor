"""Log-Periodic Power Law Singularity (LPPLS) bubble diagnostics.

Model (Johansen-Ledoit-Sornette, in the Filimonov & Sornette 2013 form):

    ln p(t) = A + B f + C1 f cos(w ln(tc - t)) + C2 f sin(w ln(tc - t)),
    f = (tc - t)^m

Given the three nonlinear parameters (tc, m, w) the four linear ones
(A, B, C1, C2) are obtained by ordinary least squares, which turns a
7-parameter search into a 3-parameter one. We first evaluate a coarse
(tc, m, w) grid in one vectorised pass, then refine the best node with
bounded L-BFGS-B.

A fit is *qualified* when it satisfies the usual filters (bounds on m,
w, tc; number of oscillations O; damping D). The *confidence indicator*
at a date is the share of qualified fits over a family of shrinking
windows ending at that date: positive bubbles (B < 0, super-exponential
growth) and negative bubbles (B > 0) are counted separately.

Time is measured in years inside each window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Search space (relative to the window length dt = t2 - t1, in years)
M_BOUNDS = (0.01, 0.99)
W_BOUNDS = (2.0, 25.0)
TC_MAX_FRAC = 0.30  # tc searched in (t2, t2 + 0.30 dt]


@dataclass
class Filters:
    # Defaults follow the filter used in the open-source `lppls` package docs
    m_min: float = 0.01
    m_max: float = 0.99
    w_min: float = 4.0
    w_max: float = 25.0
    tc_min_days: float = 5.0   # tc >= t2 + 5 calendar days (rejects fits stuck on the bound)
    tc_max_frac: float = 0.10  # tc <= t2 + 0.10 dt
    o_min: float = 2.5
    d_min: float = 0.5


@dataclass
class LPPLSFit:
    tc: float
    m: float
    w: float
    A: float
    B: float
    C1: float
    C2: float
    ssr: float
    t1: float
    t2: float
    oscillations: float
    damping: float
    qualified: bool
    sign: int  # +1 positive bubble (B<0), -1 negative bubble (B>0), 0 not qualified

    def as_dict(self) -> dict:
        return asdict(self)


def _design(t: np.ndarray, tc: float, m: float, w: float) -> np.ndarray:
    dt = tc - t
    f = dt**m
    lg = np.log(dt)
    return np.column_stack([np.ones_like(t), f, f * np.cos(w * lg), f * np.sin(w * lg)])


def _ols(t: np.ndarray, y: np.ndarray, tc: float, m: float, w: float):
    X = _design(t, tc, m, w)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef, float(resid @ resid)


def _grid_search(t: np.ndarray, y: np.ndarray, n_tc=8, n_m=6, n_w=8):
    t2, dt = t[-1], t[-1] - t[0]
    tcs = t2 + dt * np.linspace(0.005, TC_MAX_FRAC, n_tc)
    ms = np.linspace(0.1, 0.9, n_m)
    ws = np.linspace(W_BOUNDS[0] + 1, 15.0, n_w)
    TC, M, W = np.meshgrid(tcs, ms, ws, indexing="ij")
    TC, M, W = TC.ravel(), M.ravel(), W.ravel()

    d = TC[:, None] - t[None, :]  # K x n
    lg = np.log(d)
    f = d ** M[:, None]
    X = np.stack([np.ones_like(f), f, f * np.cos(W[:, None] * lg), f * np.sin(W[:, None] * lg)], axis=2)  # K x n x 4
    XtX = np.einsum("kni,knj->kij", X, X) + 1e-10 * np.eye(4)
    Xty = np.einsum("kni,n->ki", X, y)
    coef = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    resid = y[None, :] - np.einsum("kni,ki->kn", X, coef)
    ssr = np.einsum("kn,kn->k", resid, resid)
    best = int(np.nanargmin(ssr))
    return TC[best], M[best], W[best]


def fit(t: np.ndarray, y: np.ndarray, filters: Filters | None = None) -> LPPLSFit:
    """Fit the LPPLS model to log-prices ``y`` observed at times ``t`` (years)."""
    filters = filters or Filters()
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t1, t2 = t[0], t[-1]
    dt = t2 - t1

    x0 = _grid_search(t, y)
    bounds = [(t2 + 1 / 365.25, t2 + TC_MAX_FRAC * dt), M_BOUNDS, W_BOUNDS]

    def objective(x):
        try:
            return _ols(t, y, *x)[1]
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return 1e12

    res = minimize(objective, x0=np.array(x0), method="L-BFGS-B", bounds=bounds, options={"maxiter": 60, "maxfun": 400})
    tc, m, w = res.x if res.success or res.fun < objective(x0) else x0
    (A, B, C1, C2), ssr = _ols(t, y, tc, m, w)

    C = np.hypot(C1, C2)
    oscillations = (w / (2 * np.pi)) * np.log((tc - t1) / (tc - t2))
    damping = (m * abs(B)) / (w * C) if C > 0 else np.inf
    qualified = (
        filters.m_min <= m <= filters.m_max
        and filters.w_min <= w <= filters.w_max
        and t2 + filters.tc_min_days / 365.25 <= tc <= t2 + filters.tc_max_frac * dt
        and oscillations >= filters.o_min
        and damping >= filters.d_min
    )
    sign = 0
    if qualified:
        sign = 1 if B < 0 else -1
    return LPPLSFit(tc, m, w, A, B, C1, C2, ssr, t1, t2, float(oscillations), float(damping), bool(qualified), sign)


def to_years(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return (index - origin).days.to_numpy() / 365.25


def fit_window(close: pd.Series, filters: Filters | None = None) -> tuple[LPPLSFit, pd.Timestamp]:
    """Fit a single window of closes; returns the fit and the time origin."""
    origin = close.index[0]
    t = to_years(close.index, origin)
    return fit(t, np.log(close.to_numpy(float)), filters), origin


def fitted_curve(f: LPPLSFit, origin: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Series:
    t = to_years(index, origin)
    t = np.minimum(t, f.tc - 1e-6)
    X = _design(t, f.tc, f.m, f.w)
    return pd.Series(np.exp(X @ np.array([f.A, f.B, f.C1, f.C2])), index=index)


def tc_to_date(f: LPPLSFit, origin: pd.Timestamp) -> pd.Timestamp:
    return origin + pd.Timedelta(days=f.tc * 365.25)


def confidence_indicator(
    close: pd.Series,
    end_pos: int,
    max_window: int = 750,
    min_window: int = 125,
    window_step: int = 25,
    filters: Filters | None = None,
) -> dict:
    """Share of qualified positive / negative bubble fits over shrinking windows ending at ``end_pos``."""
    filters = filters or Filters()
    pos = neg = total = 0
    tcs = []
    for L in range(max_window, min_window - 1, -window_step):
        start = end_pos - L + 1
        if start < 0:
            continue
        window = close.iloc[start : end_pos + 1]
        f, origin = fit_window(window, filters)
        total += 1
        if f.sign == 1:
            pos += 1
            tcs.append(tc_to_date(f, origin))
        elif f.sign == -1:
            neg += 1
    return {
        "date": close.index[end_pos],
        "pos_conf": pos / total if total else np.nan,
        "neg_conf": neg / total if total else np.nan,
        "n_fits": total,
        "median_tc": pd.Series(tcs).median() if tcs else pd.NaT,
    }


def rolling_confidence(
    close: pd.Series,
    lookback: int = 504,
    step: int = 10,
    max_window: int = 750,
    min_window: int = 125,
    window_step: int = 25,
    filters: Filters | None = None,
    progress=None,
) -> pd.DataFrame:
    """Confidence indicator on a grid of end dates over the last ``lookback`` days."""
    n = len(close)
    ends = list(range(max(min_window, n - lookback), n, step))
    if ends and ends[-1] != n - 1:
        ends.append(n - 1)
    rows = []
    for k, e in enumerate(ends):
        rows.append(confidence_indicator(close, e, max_window, min_window, window_step, filters))
        if progress is not None:
            progress((k + 1) / len(ends))
    return pd.DataFrame(rows).set_index("date")
