"""Crash-probability models with purged walk-forward validation.

Leakage control
---------------
The label at date ``i`` depends on prices up to ``i + H``. When the model
is refitted at date ``t0`` to predict ``[t0, t0 + step)``, only rows with
``i + H < t0`` are used for training (a *purge* of ``H`` rows), so no
training label overlaps the test period. Features are standardised inside
the pipeline, i.e. with training-window statistics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BASELINE_FEATURES

MODEL_NAMES = ["Logistic regression", "Gradient boosting", "Baseline: realised vol only"]


def make_model(name: str, balanced: bool = False):
    cw = "balanced" if balanced else None
    if name == "Gradient boosting":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=100,
            l2_regularization=1.0,
            class_weight=cw,
            random_state=0,
        )
    # Logistic regression (also used for the single-feature baseline)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight=cw, max_iter=2000))


def model_features(name: str, all_features: list[str]) -> list[str]:
    if name == "Baseline: realised vol only":
        return [c for c in BASELINE_FEATURES if c in all_features]
    return list(all_features)


def align(features: pd.DataFrame, label: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Keep rows where every feature is defined (labels may be NaN at the end)."""
    X = features.dropna()
    y = label.reindex(X.index)
    return X, y


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    horizon: int,
    min_train: int = 252 * 8,
    step: int = 63,
    balanced: bool = False,
) -> pd.Series:
    """Out-of-sample crash probabilities from an expanding, purged walk-forward."""
    cols = model_features(model_name, list(X.columns))
    Xv = X[cols].to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    labeled = ~np.isnan(yv)
    n = len(X)
    proba = np.full(n, np.nan)
    for t0 in range(min_train, n, step):
        t1 = min(t0 + step, n)
        train_end = t0 - horizon  # rows i <= t0 - H - 1 have labels known at t0
        if train_end <= 0:
            continue
        tr = np.arange(train_end)
        tr = tr[labeled[tr]]
        ytr = yv[tr]
        if len(tr) < 252 or ytr.sum() < 10 or (ytr == 0).sum() < 10:
            continue
        model = make_model(model_name, balanced).fit(Xv[tr], ytr.astype(int))
        proba[t0:t1] = model.predict_proba(Xv[t0:t1])[:, 1]
    return pd.Series(proba, index=X.index, name=model_name)


def expanding_base_rate(y: pd.Series, horizon: int) -> pd.Series:
    """Climatological forecast: frequency of crashes among labels known at t."""
    return y.expanding().mean().shift(horizon + 1)


@dataclass
class Scores:
    n_obs: int
    base_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    brier_climatology: float
    brier_skill: float


def score(y: pd.Series, p: pd.Series, horizon: int) -> Scores | None:
    clim = expanding_base_rate(y, horizon)
    m = y.notna() & p.notna() & clim.notna()
    if m.sum() < 50 or y[m].nunique() < 2:
        return None
    yt, pt, ct = y[m].astype(int), p[m], clim[m]
    b = brier_score_loss(yt, pt)
    bc = brier_score_loss(yt, ct)
    return Scores(
        n_obs=int(m.sum()),
        base_rate=float(yt.mean()),
        roc_auc=float(roc_auc_score(yt, pt)),
        pr_auc=float(average_precision_score(yt, pt)),
        brier=float(b),
        brier_climatology=float(bc),
        brier_skill=float(1 - b / bc) if bc > 0 else float("nan"),
    )


def curves(y: pd.Series, p: pd.Series) -> dict:
    m = y.notna() & p.notna()
    yt, pt = y[m].astype(int), p[m]
    fpr, tpr, _ = roc_curve(yt, pt)
    prec, rec, _ = precision_recall_curve(yt, pt)
    n_bins = 10 if m.sum() > 500 else 5
    frac_pos, mean_pred = calibration_curve(yt, pt, n_bins=n_bins, strategy="quantile")
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec, "cal_pred": mean_pred, "cal_obs": frac_pos}


def evaluate_alarms(
    p: pd.Series, y: pd.Series, episodes: pd.DataFrame, threshold: float, horizon: int
) -> tuple[dict, pd.DataFrame]:
    """Turn probabilities into an alarm (p > threshold) and evaluate it.

    Event level: a fast crash episode is *anticipated* if the alarm was on
    at least once in the ``horizon`` trading days before the breach date.
    Lead time is measured from the first such alarm day to the breach.
    """
    p = p.dropna()
    alarm = p > threshold
    idx = p.index
    ev_rows = []
    fast = episodes[episodes["fast"]] if len(episodes) else episodes
    for _, ep in fast.iterrows():
        b = ep["breach_date"]
        if b <= idx[0] or b > idx[-1]:
            continue
        pos = idx.searchsorted(b)
        lo = max(0, pos - horizon)
        window = alarm.iloc[lo:pos]
        hit = bool(window.any())
        lead = int(len(window) - np.argmax(window.to_numpy())) if hit else None
        ev_rows.append(
            {
                "peak_date": ep["peak_date"].date(),
                "breach_date": b.date(),
                "depth": ep["depth"],
                "anticipated": hit,
                "lead_days": lead,
                "max_proba_before": float(p.iloc[lo:pos].max()) if pos > lo else np.nan,
            }
        )
    events = pd.DataFrame(ev_rows)

    yl = y.reindex(idx)
    m = yl.notna()
    alarm_days = alarm[m]
    precision_days = float(yl[m][alarm_days].mean()) if alarm_days.any() else float("nan")
    recall_days = float(alarm_days[yl[m] == 1].mean()) if (yl[m] == 1).any() else float("nan")

    # Alarm spells: consecutive alarm days; a spell is "false" if no crash label occurs in it.
    spells = (alarm != alarm.shift()).cumsum()[alarm]
    n_spells = int(spells.nunique())
    false_spells = 0
    for _, grp in alarm[alarm].groupby(spells):
        lab = yl.reindex(grp.index)
        if lab.notna().all() and lab.sum() == 0:
            false_spells += 1

    summary = {
        "time_in_alarm": float(alarm.mean()),
        "precision_days": precision_days,
        "recall_days": recall_days,
        "episodes_tested": int(len(events)),
        "episodes_anticipated": int(events["anticipated"].sum()) if len(events) else 0,
        "median_lead_days": float(events["lead_days"].dropna().median()) if len(events) and events["anticipated"].any() else float("nan"),
        "alarm_spells": n_spells,
        "false_alarm_spells": false_spells,
    }
    return summary, events


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str, balanced: bool = False):
    """Fit on every labelled row; return model, feature list and latest probability."""
    cols = model_features(model_name, list(X.columns))
    m = y.notna()
    model = make_model(model_name, balanced).fit(X.loc[m, cols].to_numpy(float), y[m].astype(int).to_numpy())
    latest = X[cols].iloc[[-1]].to_numpy(float)
    p_latest = float(model.predict_proba(latest)[0, 1])
    return model, cols, p_latest


def explain_latest(model, cols: list[str], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Driver table for the latest date.

    * Logistic models: contribution to the log-odds = coefficient x
      standardised feature value (exact decomposition).
    * Tree models: permutation importance (drop in average precision),
      computed in-sample on the last 10 labelled years - a global, not a
      date-specific, explanation.
    """
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        scaler = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        z = scaler.transform(X[cols].iloc[[-1]].to_numpy(float))[0]
        contrib = lr.coef_[0] * z
        out = pd.DataFrame({"feature": cols, "z_score": z, "coefficient": lr.coef_[0], "contribution": contrib})
        out["kind"] = "log-odds contribution (today)"
        return out.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)

    m = y.notna()
    Xl, yl = X.loc[m, cols].iloc[-2520:], y[m].iloc[-2520:].astype(int)
    pi = permutation_importance(model, Xl.to_numpy(float), yl.to_numpy(), scoring="average_precision", n_repeats=5, random_state=0)
    out = pd.DataFrame({"feature": cols, "contribution": pi.importances_mean, "z_score": np.nan, "coefficient": np.nan})
    out["kind"] = "permutation importance (global)"
    return out.sort_values("contribution", ascending=False).reset_index(drop=True)
