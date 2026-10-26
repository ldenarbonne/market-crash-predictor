import streamlit as st

from crashlab.features import FEATURE_DESCRIPTIONS

st.title("📚 Methodology")

st.markdown(r"""
### 1. Problem statement
The app estimates the probability that an equity index suffers a **fast crash** - a peak-to-trough
loss of at least $\theta$ realised within the next $H$ trading days:

$$
\text{FMDD}_t=\min_{t<s\le t+H}\frac{P_s}{\max_{t\le u\le s}P_u}-1,\qquad
y_t=\mathbb{1}\{\text{FMDD}_t\le-\theta\}.
$$

Unlike a plain forward return, the forward maximum drawdown also captures losses that start after a
further rise. The last $H$ days have no label.

### 2. Early-warning indicators
All indicators at date $t$ use data up to the close of $t$ only.
""")
st.table({"Indicator": list(FEATURE_DESCRIPTIONS), "Definition": list(FEATURE_DESCRIPTIONS.values())})

st.markdown(r"""
Rationale: Greenwood, Shleifer & You (2019) show that sharp price run-ups do not forecast low *average*
returns but do forecast a markedly higher *probability of a crash*, and that the volatility and the
acceleration of the run-up help separate run-ups that crash from those that keep going. Volatility
regimes, implied-volatility premia and the yield-curve slope are standard stress and cycle indicators.

### 3. Models
* **Logistic regression** (standardised inputs, L2 penalty) - transparent, gives an exact
  decomposition of today's log-odds into feature contributions.
* **Gradient boosting** (shallow histogram trees, strong regularisation) - captures non-linearities
  and interactions; explained through permutation importance.
* **Baseline** - logistic regression on 1-month realised volatility only. A model that cannot beat
  this baseline adds little.

### 4. Validation - purged expanding walk-forward
At each refit date $t_0$ the model is trained on rows $i$ with $i+H<t_0$ (labels fully observed
before $t_0$) and predicts $[t_0,t_0+\text{step})$. The purge of $H$ rows removes the overlap between
training labels and the test period, the classic source of look-ahead bias with overlapping targets
(see López de Prado, 2018). Scaling is fitted inside each training window.

Metrics: ROC AUC (ranking), PR AUC (compare with the base rate - more informative for rare events),
Brier score and **Brier skill** against a climatological forecast (expanding crash frequency known at
$t$), calibration curve, and event-level alarm statistics (crashes anticipated, lead time, false-alarm
spells).

### 5. LPPLS bubble diagnostics
$$
\ln p(t)=A+B\,(t_c-t)^m+C_1(t_c-t)^m\cos(\omega\ln(t_c-t))+C_2(t_c-t)^m\sin(\omega\ln(t_c-t))
$$
Following Filimonov & Sornette (2013), $A,B,C_1,C_2$ are solved by OLS for each $(t_c,m,\omega)$,
reducing the search to three nonlinear parameters (vectorised grid, then bounded L-BFGS-B).
A fit is *qualified* if $m$, $\omega$, $t_c$, the number of oscillations $O$ and the damping $D$ pass
the filters; $B<0$ signals a positive (crash-prone) bubble. The **confidence indicator** is the share
of qualified fits across shrinking windows ending at the same date.

### 6. Limitations
* **Few events.** A few dozen fast crashes at most in the history of a major index: every metric
  has wide confidence intervals; results are sensitive to $\theta$, $H$ and the start date.
* **Exogenous shocks** (1987, 2020) are largely unpredictable from market data alone.
* **Regime change and data snooping.** Indicators are chosen with hindsight; out-of-sample here
  means "not used for fitting", not "unknown to the designer".
* **Free data.** Yahoo Finance data are unofficial, may be rate-limited and are not adjusted
  for every corporate event. The VIX starts in 1990.
* **LPPLS** calibration is notoriously unstable, and its predictive value is debated in the literature.
* Educational tool - **not investment advice**.

### 7. References
* Johansen, A., Ledoit, O., Sornette, D. (2000). Crashes as critical points. *IJTAF* 3(2).
* Filimonov, V., Sornette, D. (2013). A stable and robust calibration scheme of the log-periodic power
  law model. *Physica A* 392(17). [arXiv:1108.0099](https://arxiv.org/abs/1108.0099)
* Sornette, D. (2003). *Why Stock Markets Crash*. Princeton University Press.
* Greenwood, R., Shleifer, A., You, Y. (2019). Bubbles for Fama. *Journal of Financial Economics*
  131(1), 20-43. [NBER w23191](https://www.nber.org/papers/w23191)
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (purging, embargo).
* Open-source reference implementation: [Boulder-Investment-Technologies/lppls](https://github.com/Boulder-Investment-Technologies/lppls)
""")
