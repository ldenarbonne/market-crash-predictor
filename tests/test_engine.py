import io

import numpy as np
import pandas as pd
import pytest

from crashlab import data, features, labels, lppls, model, synthetic


@pytest.fixture(scope="module")
def market():
    return synthetic.generate_market(n_years=20, seed=11)


# ------------------------------------------------------------------ labels
def test_forward_max_drawdown_simple_path():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100, 110, 99, 120, 60, 70], index=idx, dtype=float)
    fmdd = labels.forward_max_drawdown(close, horizon=2)
    # t=0: path 100,110,99 -> worst 99/110-1
    assert fmdd.iloc[0] == pytest.approx(99 / 110 - 1)
    # t=2: path 99,120,60 -> 60/120-1 = -50%
    assert fmdd.iloc[2] == pytest.approx(-0.5)
    assert fmdd.iloc[-2:].isna().all()


def test_crash_label_threshold():
    idx = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100, 110, 99, 120, 60, 70], index=idx, dtype=float)
    y = labels.crash_label(close, 2, 0.2)
    assert y.iloc[0] == 0 and y.iloc[2] == 1 and np.isnan(y.iloc[-1])


def test_episodes_detects_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=8)
    close = pd.Series([100, 105, 90, 80, 95, 106, 107, 100], index=idx, dtype=float)
    eps = labels.drawdown_episodes(close, 0.15, horizon=5)
    assert len(eps) == 1
    ep = eps.iloc[0]
    assert ep["peak_date"] == idx[1] and ep["trough_date"] == idx[3] and ep["recovery_date"] == idx[5]
    assert ep["depth"] == pytest.approx(80 / 105 - 1)
    assert bool(ep["fast"])


# ------------------------------------------------------------------ features
def test_features_have_no_lookahead(market):
    full = features.build_features(market)
    cut = features.build_features(market.iloc[:3000])
    common = cut.dropna().index
    pd.testing.assert_frame_equal(full.loc[common], cut.loc[common])


def test_features_columns(market):
    f = features.build_features(market)
    assert {"vix", "vrp", "term_spread", "accel", "vol_ratio"} <= set(f.columns)
    assert set(f.columns) <= set(features.FEATURE_DESCRIPTIONS)


# ------------------------------------------------------------------ model
def test_walk_forward_is_purged_and_out_of_sample(market):
    H = 63
    y = labels.crash_label(market["close"], H, 0.15)
    X, y = model.align(features.build_features(market), y)
    p = model.walk_forward(X, y, "Logistic regression", H, min_train=252 * 6, step=126)
    assert p.iloc[: 252 * 6].isna().all()          # nothing before the first refit
    assert p.dropna().between(0, 1).all()
    assert p.iloc[-1] == p.iloc[-1]                  # live forecast exists for the last date
    sc = model.score(y, p, H)
    assert sc is not None and 0 <= sc.roc_auc <= 1


def test_walk_forward_does_not_use_future_labels(market):
    """Changing labels after a refit date must not change predictions before it."""
    H = 63
    y = labels.crash_label(market["close"], H, 0.15)
    X, y = model.align(features.build_features(market), y)
    p1 = model.walk_forward(X, y, "Logistic regression", H, min_train=252 * 6, step=126)
    cut = 252 * 10
    y2 = y.copy()
    y2.iloc[cut - H:] = 1 - y2.iloc[cut - H:]       # corrupt labels only knowable after `cut`
    p2 = model.walk_forward(X, y2, "Logistic regression", H, min_train=252 * 6, step=126)
    first_block_end = 252 * 6 + ((cut - 252 * 6) // 126) * 126
    pd.testing.assert_series_equal(p1.iloc[:first_block_end], p2.iloc[:first_block_end])


def test_alarm_evaluation(market):
    H = 63
    y = labels.crash_label(market["close"], H, 0.15)
    X, y = model.align(features.build_features(market), y)
    eps = labels.drawdown_episodes(market["close"], 0.15, H)
    p = model.walk_forward(X, y, "Baseline: realised vol only", H, min_train=252 * 6, step=126)
    summary, events = model.evaluate_alarms(p, y, eps, 0.2, H)
    assert 0 <= summary["time_in_alarm"] <= 1
    assert summary["episodes_anticipated"] <= summary["episodes_tested"]


def test_explain_latest_logistic_decomposition(market):
    H = 63
    y = labels.crash_label(market["close"], H, 0.15)
    X, y = model.align(features.build_features(market), y)
    m, cols, p_now = model.fit_final(X, y, "Logistic regression")
    ex = model.explain_latest(m, cols, X, y)
    intercept = m.named_steps["logisticregression"].intercept_[0]
    logit = intercept + ex["contribution"].sum()
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(p_now, rel=1e-6)


# ------------------------------------------------------------------ LPPLS
def test_lppls_recovers_parameters():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2, 500)
    tc, m, w = 2.1, 0.5, 8.0
    y = 5 - 0.8 * (tc - t) ** m + 0.04 * (tc - t) ** m * np.cos(w * np.log(tc - t)) + 0.003 * rng.standard_normal(500)
    f = lppls.fit(t, y)
    assert f.tc == pytest.approx(tc, abs=0.02)
    assert f.m == pytest.approx(m, abs=0.05)
    assert f.w == pytest.approx(w, abs=0.3)
    assert f.B < 0 and f.qualified and f.sign == 1


def test_lppls_confidence_indicator_bounds(market):
    ci = lppls.confidence_indicator(market["close"], len(market) - 1, max_window=300, min_window=150, window_step=50)
    assert 0 <= ci["pos_conf"] <= 1 and 0 <= ci["neg_conf"] <= 1 and ci["n_fits"] == 4


# ------------------------------------------------------------------ data
def test_parse_csv_aliases():
    csv = "Date,Adj Close,VIX\n2024-01-02,100,13\n2024-01-03,101,14\n"
    df = data.parse_csv(io.BytesIO(csv.encode()))
    assert list(df.columns) == data.COLUMNS
    assert df["close"].tolist() == [100.0, 101.0]
    assert df["vix"].tolist() == [13.0, 14.0]


def test_parse_csv_requires_close():
    with pytest.raises(ValueError):
        data.parse_csv(io.BytesIO(b"date,foo\n2024-01-02,1\n"))
