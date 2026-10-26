import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from crashlab import lppls
from crashlab import ui

s = ui.settings()
df = s["df"]
close = df["close"]

st.title("🫧 LPPLS bubble diagnostics")
st.caption("Log-Periodic Power Law Singularity model (Johansen-Ledoit-Sornette), Filimonov-Sornette calibration. "
           "Detects faster-than-exponential growth decorated by accelerating oscillations.")

with st.expander("Qualification filters", expanded=False):
    c = st.columns(4)
    flt = lppls.Filters(
        m_min=c[0].number_input("m min", 0.0, 1.0, 0.01, 0.01),
        m_max=c[0].number_input("m max", 0.0, 1.0, 0.99, 0.01),
        w_min=c[1].number_input("ω min", 0.0, 30.0, 4.0, 0.5),
        w_max=c[1].number_input("ω max", 0.0, 30.0, 25.0, 0.5),
        tc_min_days=c[2].number_input("tc min (days after window end)", 0.0, 60.0, 5.0, 1.0),
        tc_max_frac=c[2].number_input("tc horizon (share of window)", 0.01, 0.30, 0.10, 0.01),
        o_min=c[3].number_input("Min oscillations O", 0.0, 10.0, 2.5, 0.5),
        d_min=c[3].number_input("Min damping D", 0.0, 5.0, 0.5, 0.1),
    )

tab1, tab2 = st.tabs(["Single fit", "Confidence indicator"])

# ---------------------------------------------------------------- single fit
with tab1:
    c1, c2 = st.columns(2)
    end = c1.date_input("Window end", value=close.index[-1].date(), min_value=close.index[300].date(),
                        max_value=close.index[-1].date())
    length = c2.slider("Window length (trading days)", 100, 1500, 500, 25)
    end_pos = int(close.index.searchsorted(pd.Timestamp(end), side="right")) - 1
    window = close.iloc[max(0, end_pos - length + 1): end_pos + 1]
    fit, origin = lppls.fit_window(window, flt)
    tc_date = lppls.tc_to_date(fit, origin)

    m = st.columns(5)
    m[0].metric("Verdict", "Positive bubble" if fit.sign == 1 else ("Negative bubble" if fit.sign == -1 else "Not qualified"))
    m[1].metric("Critical time tc", f"{tc_date:%d %b %Y}")
    m[2].metric("m / ω", f"{fit.m:.2f} / {fit.w:.1f}")
    m[3].metric("Oscillations O", f"{fit.oscillations:.2f}")
    m[4].metric("Damping D", f"{fit.damping:.2f}")

    ext_index = pd.bdate_range(window.index[0], min(tc_date, window.index[-1] + pd.Timedelta(days=int(0.3 * (window.index[-1] - window.index[0]).days))))
    curve = lppls.fitted_curve(fit, origin, ext_index)
    ctx = close.iloc[max(0, end_pos - length - 100): min(len(close), end_pos + 150)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctx.index, y=ctx, name="Index", line=dict(color="#52514e", width=1.5)))
    fig.add_trace(go.Scatter(x=curve.index, y=curve, name="LPPLS fit", line=dict(color=ui.BLUE, width=2)))
    fig.add_vrect(x0=window.index[0], x1=window.index[-1], fillcolor=ui.BLUE, opacity=0.06, line_width=0,
                  annotation_text="fit window", annotation_position="top left")
    fig.add_vline(x=tc_date, line=dict(color=ui.CRITICAL, dash="dash", width=1.5))
    fig.add_annotation(x=tc_date, y=1, yref="paper", text="tc", showarrow=False, font=dict(color=ui.CRITICAL))
    fig.update_yaxes(type="log")
    fig.update_layout(height=480, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Fitted parameters"):
        st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit.as_dict().items()})
    st.caption("A single fit is fragile: tc is notoriously unstable. Prefer the confidence indicator, "
               "which aggregates many windows.")

# ---------------------------------------------------------------- confidence indicator
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    lookback = c1.slider("Look-back (trading days)", 126, 2520, 504, 21)
    step = c2.slider("Evaluate every (days)", 5, 63, 10, 5)
    max_w = c3.slider("Largest window", 250, 1500, 750, 25)
    min_w = c4.slider("Smallest window", 60, 500, 125, 5)
    window_step = st.select_slider("Window spacing (days)", [10, 25, 50, 100], value=25)
    n_dates = len(range(max(min_w, len(close) - lookback), len(close), step)) + 1
    n_windows = len(range(max_w, min_w - 1, -window_step))
    st.caption(f"≈ {n_dates * n_windows:,} LPPLS fits (~{n_dates * n_windows * 0.04:.0f} s). Results are cached.")

    if st.button("Compute confidence indicator", type="primary"):
        bar = st.progress(0.0, text="Fitting...")

        @st.cache_data(show_spinner=False)
        def _run(close_, lookback_, step_, max_w_, min_w_, ws_, flt_dict):
            return lppls.rolling_confidence(close_, lookback_, step_, max_w_, min_w_, ws_, lppls.Filters(**flt_dict),
                                            progress=lambda x: bar.progress(x, text=f"Fitting... {x:.0%}"))

        st.session_state["lppls_ci"] = _run(close, lookback, step, max_w, min_w, window_step, vars(flt))
        bar.empty()

    ci = st.session_state.get("lppls_ci")
    if ci is not None and len(ci):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.55, 0.45],
                            subplot_titles=("Index level (log scale)", "LPPLS confidence indicator"))
        seg = close[close.index >= ci.index.min() - pd.Timedelta(days=30)]
        fig.add_trace(go.Scatter(x=seg.index, y=seg, name="Index", line=dict(color="#52514e", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ci.index, y=ci["pos_conf"], name="Positive bubble (crash risk)",
                                 line=dict(color=ui.CRITICAL, width=2), mode="lines+markers", marker=dict(size=6)), row=2, col=1)
        fig.add_trace(go.Scatter(x=ci.index, y=ci["neg_conf"], name="Negative bubble (rebound)",
                                 line=dict(color=ui.AQUA, width=2), mode="lines+markers", marker=dict(size=6)), row=2, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
        fig.update_layout(height=600, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        last = ci.iloc[-1]
        st.metric("Latest positive-bubble confidence", ui.fmt_pct(last["pos_conf"], 0),
                  help="Share of windows ending at the latest date with a qualified B<0 fit.")
        if pd.notna(last["median_tc"]):
            st.write(f"Median critical time across qualified windows: **{last['median_tc']:%d %b %Y}**")
        st.download_button("Download indicator (CSV)", ci.to_csv().encode(), "lppls_confidence.csv", "text/csv")
    else:
        st.info("Set the parameters and press **Compute confidence indicator**.")
