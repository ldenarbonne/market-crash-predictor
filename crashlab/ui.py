"""Shared Streamlit helpers: cached computations, colours and chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import data as data_mod
from . import features as feat_mod
from . import labels as lab_mod
from . import model as model_mod
from . import synthetic

# Palette (validated reference palette: categorical slots + reserved status colour)
BLUE = "#2a78d6"      # series 1 - index level, main model
ORANGE = "#eb6834"    # series 2 - second model / probability
AQUA = "#1baf7a"      # series 3 - baseline
CRITICAL = "#d03b3b"  # status: crash episodes
MUTED = "#8a8984"     # thresholds, reference lines

MODEL_COLORS = {
    "Logistic regression": BLUE,
    "Gradient boosting": ORANGE,
    "Baseline: realised vol only": AQUA,
}


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading market data from Yahoo Finance...")
def cached_yahoo(ticker: str, start: str) -> pd.DataFrame:
    return data_mod.fetch_yahoo(ticker, start)


@st.cache_data(show_spinner=False)
def cached_snapshot(ticker: str) -> pd.DataFrame | None:
    return data_mod.load_snapshot(ticker)


@st.cache_data(show_spinner=False)
def cached_synthetic(seed: int) -> pd.DataFrame:
    return synthetic.generate_market(seed=seed)


def load_market(source: str, ticker: str, start: str, upload, seed: int) -> tuple[pd.DataFrame, str]:
    """Return (data, description of the source actually used)."""
    if source == "Synthetic demo":
        return cached_synthetic(seed), "Synthetic regime-switching market (demo)"
    if source == "Upload CSV":
        if upload is None:
            st.info("Upload a CSV (columns: date, close, optional vix, tnx, irx) - showing the synthetic demo meanwhile.")
            return cached_synthetic(seed), "Synthetic regime-switching market (demo)"
        return data_mod.parse_csv(upload.getvalue()), f"Uploaded file: {upload.name}"
    if source == "Repository snapshot":
        snap = cached_snapshot(ticker)
        if snap is None:
            st.warning("No snapshot for this index in data/ - falling back to Yahoo Finance.")
        else:
            return snap[snap.index >= start], f"Repository snapshot ({data_mod.snapshot_path(ticker).name})"
    if source in ("Auto", "Repository snapshot", "Yahoo Finance (live)"):
        if source == "Auto":
            snap = cached_snapshot(ticker)
            if snap is not None:
                return snap[snap.index >= start], f"Repository snapshot ({data_mod.snapshot_path(ticker).name})"
        try:
            return cached_yahoo(ticker, start), f"Yahoo Finance (live, {ticker})"
        except Exception as exc:  # rate limit, network...
            st.warning(f"Yahoo Finance unavailable ({exc}). Showing the synthetic demo instead.")
            return cached_synthetic(seed), "Synthetic regime-switching market (fallback)"
    return cached_synthetic(seed), "Synthetic regime-switching market (demo)"


# --------------------------------------------------------------------- modelling
@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame, horizon: int, threshold: float):
    feats = feat_mod.build_features(df)
    y = lab_mod.crash_label(df["close"], horizon, threshold)
    X, y = model_mod.align(feats, y)
    episodes = lab_mod.drawdown_episodes(df["close"], threshold, horizon)
    return X, y, episodes


@st.cache_data(show_spinner=False)
def oos_probabilities(X: pd.DataFrame, y: pd.Series, model_name: str, horizon: int, min_train_years: int, step: int, balanced: bool) -> pd.Series:
    return model_mod.walk_forward(X, y, model_name, horizon, min_train=252 * min_train_years, step=step, balanced=balanced)


@st.cache_resource(show_spinner=False)
def final_model(X: pd.DataFrame, y: pd.Series, model_name: str, balanced: bool):
    return model_mod.fit_final(X, y, model_name, balanced)


def settings() -> dict:
    return st.session_state["settings"]


# ------------------------------------------------------------------------ charts
def add_episode_bands(fig: go.Figure, episodes: pd.DataFrame, start, end, rows=None, only_fast: bool = True):
    eps = episodes[episodes["fast"]] if only_fast and len(episodes) else episodes
    first = True
    for _, ep in eps.iterrows():
        x0, x1 = ep["peak_date"], ep["trough_date"]
        if x1 < start or x0 > end:
            continue
        kwargs = dict(x0=x0, x1=x1, fillcolor=CRITICAL, opacity=0.12, line_width=0, layer="below")
        if rows:
            for r in rows:
                fig.add_vrect(row=r, col=1, **kwargs)
        else:
            fig.add_vrect(**kwargs)
        first = False
    if not first:  # legend entry for the bands
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=CRITICAL, opacity=0.35, symbol="square"),
                       name="Crash episode (peak to trough)"),
            row=1 if rows else None, col=1 if rows else None,
        )


def price_and_probability(df: pd.DataFrame, probas: dict[str, pd.Series], episodes: pd.DataFrame, alarm: float | None, title: str) -> go.Figure:
    """Two stacked panels sharing the time axis (never a dual axis)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4],
                        subplot_titles=("Index level (log scale)", "Out-of-sample crash probability"))
    start = min(p.dropna().index.min() for p in probas.values()) if probas else df.index.min()
    d = df[df.index >= start]
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], name="Index", line=dict(color="#52514e", width=1.5),
                             hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>"), row=1, col=1)
    for name, p in probas.items():
        p = p.dropna()
        fig.add_trace(go.Scatter(x=p.index, y=p, name=name, line=dict(color=MODEL_COLORS.get(name, BLUE), width=1.5),
                                 hovertemplate="%{x|%d %b %Y}<br>%{y:.1%}<extra>" + name + "</extra>"), row=2, col=1)
    if alarm is not None:
        fig.add_hline(y=alarm, row=2, col=1, line=dict(color=MUTED, dash="dash", width=1),
                      annotation_text=f"alarm {alarm:.0%}", annotation_position="top left")
    add_episode_bands(fig, episodes, d.index.min(), d.index.max(), rows=[1, 2])
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=2, col=1)
    fig.update_layout(title=title, height=620, hovermode="x unified", margin=dict(l=10, r=10, t=70, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    return fig


def fmt_pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def fmt_num(x, digits: int = 3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}"
