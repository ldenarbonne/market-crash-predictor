# 📉 Market Crash Lab

**An early-warning toolkit for equity-market crashes** - purged walk-forward machine learning, crash-episode
analytics and LPPLS bubble diagnostics, in an interactive Streamlit app.

> Educational and research tool. It estimates *crash-risk regimes*; it does not predict the future and is
> **not investment advice**.

![tests](https://github.com/ldenarbonne/market-crash-lab/actions/workflows/tests.yml/badge.svg)

---

## What it does

| Page | Content |
|---|---|
| 🚦 **Crash risk today** | Probability of a fast crash (drawdown ≥ θ within H trading days), base rate, risk level, out-of-sample probability history vs realised crashes, and the drivers of today's reading |
| 🧪 **Model backtest** | Purged expanding walk-forward for three models, ROC / PR / calibration curves, Brier skill vs climatology, episode-level alarm review (anticipated crashes, lead time, false-alarm spells) |
| 📊 **Data & indicators** | Historical drawdown episodes, indicator explorer, crash frequency by indicator decile, correlation matrix |
| 🫧 **LPPLS bubble diagnostics** | Single-window LPPLS fit with critical time *t_c*, rolling multi-window confidence indicator (positive and negative bubbles) |
| 📚 **Methodology** | Target definition, indicators, validation scheme, limitations, references |

## Method in brief

**Target.** Forward maximum drawdown over the next *H* days
`FMDD_t = min_{t<s≤t+H} P_s / max_{t≤u≤s} P_u − 1`, crash label `y_t = 1{FMDD_t ≤ −θ}`
(defaults: θ = 15 %, H = 63 trading days).

**Indicators (no look-ahead).** Run-up size and acceleration (Greenwood, Shleifer & You, 2019), realised
volatility regime, vol-of-vol, skewness / kurtosis, distance to the 200-day MA, 1-year drawdown, VIX level
and change, implied-minus-realised volatility, US term spread (10y − 3m) and its change.

**Models.** L2 logistic regression (exact log-odds decomposition), regularised histogram gradient boosting,
and a one-feature realised-volatility baseline that any model must beat.

**Validation.** Expanding walk-forward, refitted every *step* days; training rows are purged so that
`i + H < t0` - no training label overlaps the test period. Scaling is fitted inside each training window.
A unit test checks that corrupting future labels cannot change past predictions.

**LPPLS.** Filimonov & Sornette (2013) reformulation: the 4 linear parameters are solved by OLS, the 3
nonlinear ones `(t_c, m, ω)` by a vectorised grid followed by bounded L-BFGS-B; fits are qualified with the
usual filters (m, ω, t_c horizon, oscillations, damping).

## Honest first results (S&P 500, 1990 → Oct 2026, θ = 15 %, H = 63 days)

| Model | ROC AUC | PR AUC | Brier skill vs climatology |
|---|---|---|---|
| Logistic regression (all indicators) | 0.53 | 0.17 | −21 % |
| Gradient boosting (all indicators) | 0.59 | 0.16 | −19 % |
| Baseline: 1-month realised vol | 0.62 | 0.23 | +6 % |

Out-of-sample base rate ≈ 10.6 %. With only a handful of fast crashes since 1990 (1990, 1998, 2018, 2020,
2025), the richer models **do not beat the simple volatility baseline** - consistent with the literature's
view that crash *timing* is hard to forecast even when crash *risk* is not constant. Treat every number as
having very wide confidence intervals.

## Run locally (Windows)

```powershell
git clone https://github.com/ldenarbonne/market-crash-lab.git
cd market-crash-lab
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

(macOS / Linux: `source .venv/bin/activate`.)

## Data

`Auto` mode reads the CSV snapshots in `data/` (S&P 500, Nasdaq 100, Euro Stoxx 50, CAC 40), then falls back
to live Yahoo Finance via `yfinance`, then to a synthetic demo market. Snapshots are refreshed every weekday
by the **update-data** GitHub Action, so the deployed app never needs to call Yahoo (which often rate-limits
shared cloud IPs). Refresh manually with:

```bash
python scripts/update_data.py                 # default indices
python scripts/update_data.py --tickers ^GDAXI ^N225
```

You can also upload your own CSV with columns `date, close` and optionally `vix, tnx, irx`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io) → **Create app** → select the repo, branch `main`,
   main file `app.py`.
3. Optional: in the repo **Settings → Actions → General**, set *Workflow permissions* to **Read and write**
   so the data-refresh workflow can commit the updated CSV files.

## Project structure

```
market-crash-lab/
├── app.py                  # entry point: sidebar settings + navigation
├── views/                  # one file per page
├── crashlab/               # engine (pure Python, unit-tested)
│   ├── data.py             # snapshot / Yahoo / CSV loading
│   ├── labels.py           # forward max drawdown, crash episodes
│   ├── features.py         # early-warning indicators
│   ├── model.py            # purged walk-forward, scores, alarms, explanations
│   ├── lppls.py            # LPPLS calibration and confidence indicator
│   ├── synthetic.py        # regime-switching demo market
│   └── ui.py               # cached computations and charts
├── data/                   # market snapshots (refreshed by GitHub Actions)
├── scripts/update_data.py
├── tests/                  # pytest suite
└── .github/workflows/      # tests + scheduled data refresh
```

## Roadmap ideas

LPPLS confidence as an ML feature · credit-spread and breadth indicators · event-level bootstrap confidence
intervals · probability calibration (isotonic / Platt) · multi-index pooled training.

## References

- Johansen, Ledoit & Sornette (2000), *Crashes as critical points*, IJTAF.
- Filimonov & Sornette (2013), *A stable and robust calibration scheme of the log-periodic power law model*, Physica A - [arXiv:1108.0099](https://arxiv.org/abs/1108.0099).
- Sornette (2003), *Why Stock Markets Crash*, Princeton University Press.
- Greenwood, Shleifer & You (2019), *Bubbles for Fama*, Journal of Financial Economics 131(1) - [NBER w23191](https://www.nber.org/papers/w23191).
- López de Prado (2018), *Advances in Financial Machine Learning*, Wiley.
- [Boulder-Investment-Technologies/lppls](https://github.com/Boulder-Investment-Technologies/lppls) (open-source LPPLS implementation).

## Author

Laurent Motais de Narbonne - [LinkedIn](https://www.linkedin.com/in/laurent-motais-de-narbonne/) ·
[Publications](https://ldenarbonne.github.io/)

MIT License.
