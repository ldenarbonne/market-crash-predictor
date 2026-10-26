import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crashlab import model as model_mod
from crashlab import ui
from crashlab.features import FEATURE_DESCRIPTIONS

s = ui.settings()
df, H, thr = s["df"], s["horizon"], s["threshold"]

st.title("🚦 Crash risk today")
st.caption(f"Source: {s['source_used']} · last close {df.index[-1]:%d %b %Y} · "
           f"crash = drawdown ≥ {thr:.0%} within {H} trading days")

X, y, episodes = ui.prepare(df, H, thr)
with st.spinner("Running the purged walk-forward backtest..."):
    p_oos = ui.oos_probabilities(X, y, s["model_name"], H, s["min_train_years"], s["step"], s["balanced"])
model, cols, p_now = ui.final_model(X, y, s["model_name"], s["balanced"])
sc = model_mod.score(y, p_oos, H)
base_rate = float(y.mean())

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
level = "🔴 Elevated" if p_now >= s["alarm"] else ("🟠 Above average" if p_now >= 2 * base_rate else "🟢 Normal")
c1.metric(f"Crash probability, next {H} days", ui.fmt_pct(p_now), help=f"{s['model_name']} fitted on all labelled history.")
c2.metric("Historical base rate", ui.fmt_pct(base_rate), help="Share of days followed by a crash within the horizon.")
c3.metric("Risk level", level, help=f"Red above the alarm threshold ({s['alarm']:.0%}); orange above twice the base rate.")
c4.metric("Out-of-sample ROC AUC", ui.fmt_num(sc.roc_auc if sc else None, 2), help="0.5 = no skill, 1 = perfect ranking.")

if s["balanced"]:
    st.warning("Class balancing is on: the probability is a risk score, not a calibrated probability.")

# ---------------------------------------------------------------- main chart
st.plotly_chart(
    ui.price_and_probability(df, {s["model_name"]: p_oos}, episodes, s["alarm"],
                             "Walk-forward probabilities vs realised crashes"),
    width="stretch",
)
st.caption("Each probability was produced by a model trained only on data whose labels were known at that date "
           f"(expanding window, purge of {H} days, refit every {s['step']} days). Red bands: fast crash episodes.")

# ---------------------------------------------------------------- drivers
st.subheader("What drives today's reading?")
drivers = model_mod.explain_latest(model, cols, X, y).head(10)
drivers["description"] = drivers["feature"].map(FEATURE_DESCRIPTIONS)
kind = drivers["kind"].iloc[0]
d = drivers.iloc[::-1]
colors = [ui.CRITICAL if v > 0 else ui.BLUE for v in d["contribution"]] if "log-odds" in kind else [ui.BLUE] * len(d)
fig = go.Figure(go.Bar(x=d["contribution"], y=d["feature"], orientation="h", marker_color=colors,
                       customdata=d["description"], hovertemplate="%{y}: %{x:.3f}<br>%{customdata}<extra></extra>"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title=kind.capitalize(), xaxis_title=kind)
st.plotly_chart(fig, width="stretch")
if "log-odds" in kind:
    st.caption("Red bars push the crash probability up, blue bars push it down (coefficient × standardised value).")

latest = X.iloc[-1].rename("today").to_frame()
latest["percentile vs history"] = [(X[c] <= X[c].iloc[-1]).mean() for c in X.columns]
latest["description"] = latest.index.map(FEATURE_DESCRIPTIONS)
with st.expander("Today's indicator values"):
    st.dataframe(latest.style.format({"today": "{:.3f}", "percentile vs history": "{:.0%}"}), width="stretch")

st.info("**Reading guide.** Crash prediction is a rare-event problem: even a good model produces many false alarms "
        "and misses sudden exogenous shocks (e.g. pandemics). Use the output as a risk-regime indicator, "
        "compare it with the base rate, and check the backtest page before trusting it. Not investment advice.")
