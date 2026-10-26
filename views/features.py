import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from crashlab import data as data_mod
from crashlab import ui
from crashlab.features import FEATURE_DESCRIPTIONS

s = ui.settings()
df, H, thr = s["df"], s["horizon"], s["threshold"]

st.title("📊 Data & indicators")
info = data_mod.data_summary(df)
st.caption(f"Source: {s['source_used']} · {info['start']} → {info['end']} · {info['rows']:,} rows · "
           f"VIX: {'yes' if info['has_vix'] else 'no'} · rates: {'yes' if info['has_rates'] else 'no'}")

X, y, episodes = ui.prepare(df, H, thr)

# ------------------------------------------------------------ episodes table
st.subheader(f"Historical drawdowns ≥ {thr:.0%}")
if len(episodes):
    show = episodes.copy()
    for c in ["peak_date", "breach_date", "trough_date", "recovery_date"]:
        show[c] = show[c].dt.date
    st.dataframe(show.style.format({"depth": "{:.1%}"}, na_rep="not recovered"), width="stretch")
    st.caption(f"'fast' = threshold breached within {H} trading days of the peak; only fast episodes count as crashes "
               "for the label and the alarm evaluation.")
else:
    st.info("No drawdown of this size in the sample.")

c1, c2, c3 = st.columns(3)
c1.metric("Labelled days", f"{int(y.notna().sum()):,}")
c2.metric("Crash-labelled days", f"{int(y.sum()):,}")
c3.metric("Base rate", ui.fmt_pct(float(y.mean())))

# ------------------------------------------------------------ indicator explorer
st.subheader("Indicator explorer")
feat = st.selectbox("Indicator", list(X.columns), format_func=lambda c: f"{c} - {FEATURE_DESCRIPTIONS.get(c, '')}")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.5],
                    subplot_titles=("Index level (log scale)", feat))
fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Index", line=dict(color="#52514e", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=X.index, y=X[feat], name=feat, line=dict(color=ui.BLUE, width=1.5)), row=2, col=1)
ui.add_episode_bands(fig, episodes, df.index.min(), df.index.max(), rows=[1, 2])
fig.update_yaxes(type="log", row=1, col=1)
fig.update_layout(height=560, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------ conditional crash rate
st.subheader("Crash frequency by indicator decile")
lab = y.dropna()
xf = X.loc[lab.index, feat]
dec = (xf.rank(pct=True) * 10).clip(upper=9.999).astype(int) + 1
rate = lab.groupby(dec).mean()
fig2 = go.Figure(go.Bar(x=[f"D{d}" for d in rate.index], y=rate.values, marker_color=ui.BLUE,
                        hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
fig2.add_hline(y=float(lab.mean()), line=dict(color=ui.MUTED, dash="dash", width=1), annotation_text="base rate")
fig2.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10),
                   xaxis_title=f"Decile of {feat} (D1 = lowest)", yaxis_title="Share of days followed by a crash")
st.plotly_chart(fig2, width="stretch")
st.caption("Full-sample, in-sample statistic - useful to understand an indicator, not a backtest.")

# ------------------------------------------------------------ correlations
with st.expander("Correlation matrix of indicators"):
    corr = X.corr()
    fig3 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    fig3.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")

st.download_button("Download indicators and labels (CSV)", X.assign(crash_label=y).to_csv().encode(),
                   file_name="crash_indicators.csv", mime="text/csv")
