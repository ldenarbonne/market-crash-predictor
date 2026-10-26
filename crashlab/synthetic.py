"""Synthetic regime-switching market, used for the offline demo and tests.

Three regimes follow a Markov chain on business days:

* ``calm``   : moderate drift, low volatility;
* ``bubble`` : strong, accelerating drift, rising volatility, flattening
  then inverting yield curve, compressed implied-volatility premium;
* ``crash``  : large negative drift, very high volatility (20-60 days).

A crash can only start from a bubble, so early-warning features carry
genuine (but noisy) information - which is what a demo needs. Real
markets are far less cooperative: never read synthetic results as
evidence of predictability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_market(n_years: int = 35, seed: int = 11, start: str = "1990-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(n_years * 252)
    dates = pd.bdate_range(start=start, periods=n)

    # regime: 0 calm, 1 bubble, 2 crash
    regime = np.zeros(n, dtype=int)
    state, crash_left, bubble_age = 0, 0, 0
    for t in range(1, n):
        if state == 0:
            state = 1 if rng.random() < 1 / 450 else 0
            bubble_age = 0
        elif state == 1:
            bubble_age += 1
            hazard = min(0.03, 1 / 400 + bubble_age / 40000)
            if rng.random() < hazard:
                state, crash_left = 2, int(rng.integers(20, 60))
            elif rng.random() < 1 / 1500:
                state = 0  # bubble deflates quietly
        else:
            crash_left -= 1
            if crash_left <= 0:
                state = 0
        regime[t] = state

    mu = np.select([regime == 0, regime == 1, regime == 2], [0.08, 0.28, -1.6]) / 252
    vol_target = np.select([regime == 0, regime == 1, regime == 2], [0.12, 0.17, 0.45])

    # Smooth the volatility path (vol clustering) and add GARCH-like noise.
    vol = np.empty(n)
    vol[0] = 0.12
    for t in range(1, n):
        vol[t] = 0.97 * vol[t - 1] + 0.03 * vol_target[t] + 0.004 * rng.standard_normal()
    vol = np.clip(vol, 0.06, 0.9)

    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    rets = mu + vol / np.sqrt(252) * shocks
    close = 1000 * np.exp(np.cumsum(rets))

    # Implied volatility: realised vol + premium; premium compresses in bubbles.
    premium = np.select([regime == 0, regime == 1, regime == 2], [4.0, 1.0, 8.0])
    vix = 100 * vol + premium + 1.5 * rng.standard_normal(n)
    vix = pd.Series(vix).ewm(span=3).mean().to_numpy().clip(9, 90)

    # Rates: 10y random walk; term spread mean-reverts to a regime target.
    tnx = 6 + np.cumsum(0.03 * rng.standard_normal(n))
    tnx = np.clip(tnx, 0.5, 9.0)
    spread_target = np.select([regime == 0, regime == 1, regime == 2], [1.8, -0.3, 1.0])
    spread = np.empty(n)
    spread[0] = 1.5
    for t in range(1, n):
        spread[t] = spread[t - 1] + 0.01 * (spread_target[t] - spread[t - 1]) + 0.04 * rng.standard_normal()
    irx = np.clip(tnx - spread, 0.0, None)

    df = pd.DataFrame({"close": close, "vix": vix, "tnx": tnx, "irx": irx}, index=dates)
    df.index.name = "date"
    df.attrs["regime"] = regime
    return df
