import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crashlab import model as model_mod
from crashlab import ui

s = ui.settings()
df, H, thr = s["df"], s["horizon"], s["threshold"]

st.title("🧪 Model backtest")
st.caption(f"Purged expanding walk-forward · initial window {s['min_train_years']}y · refit every {s['step']} days · "
           f"crash = drawdown ≥ {thr:.0%} within {H} days")

X, y, episodes = ui.prepare(df, H, thr)
chosen = st.multiselect("Models to compare", model_mod.MODEL_NAMES, default=model_mod.MODEL_NAMES)
if not chosen:
    st.stop()

probas, scores = {}, {}
prog = st.progress(0.0, text="Walk-forward...")
for k, name in enumerate(chosen):
    probas[name] = ui.oos_probabilities(X, y, name, H, s["min_train_years"], s["step"], s["balanced"])
    scores[name] = model_mod.score(y, probas[name], H)
    prog.progress((k + 1) / len(chosen), text=f"Walk-forward: {name} done")
prog.empty()

# ------------------------------------------------------------------ scores
rows = []
for name in chosen:
    sc = scores[name]
    if sc is None:
        continue
    summ, _ = model_mod.evaluate_alarms(probas[name], y, episodes, s["alarm"], H)
    rows.append({
        "Model": name,
        "ROC AUC": sc.roc_auc,
        "PR AUC": sc.pr_auc,
        "Base rate": sc.base_rate,
        "Brier": sc.brier,
        "Brier skill vs climatology": sc.brier_skill,
        "Time in alarm": summ["time_in_alarm"],
        "Precision (alarm days)": summ["precision_days"],
        "Crashes anticipated": f"{summ['episodes_anticipated']}/{summ['episodes_tested']}",
        "Median lead (days)": summ["median_lead_days"],
        "False alarm spells": f"{summ['false_alarm_spells']}/{summ['alarm_spells']}",
    })
if not rows:
    st.error("Not enough crash events in the test period to score the models. Lower the threshold or lengthen the history.")
    st.stop()
table = pd.DataFrame(rows).set_index("Model")
st.subheader("Out-of-sample scores")
st.dataframe(
    table.style.format({"ROC AUC": "{:.3f}", "PR AUC": "{:.3f}", "Base rate": "{:.1%}", "Brier": "{:.4f}",
                        "Brier skill vs climatology": "{:+.1%}", "Time in alarm": "{:.1%}",
                        "Precision (alarm days)": "{:.1%}", "Median lead (days)": "{:.0f}"}, na_rep="n/a"),
    width="stretch",
)
st.caption("PR AUC should be compared with the base rate (a random forecast scores the base rate). "
           "Brier skill > 0 means better than the climatological forecast (expanding crash frequency). "
           f"Alarm metrics use the {s['alarm']:.0%} threshold set in the sidebar.")

st.plotly_chart(ui.price_and_probability(df, probas, episodes, s["alarm"], "Probabilities through time"), width="stretch")

# ------------------------------------------------------------------ curves
c1, c2, c3 = st.columns(3)
figs = [go.Figure(), go.Figure(), go.Figure()]
for name in chosen:
    if scores[name] is None:
        continue
    cv = model_mod.curves(y, probas[name])
    col = ui.MODEL_COLORS[name]
    figs[0].add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], name=name, line=dict(color=col, width=2)))
    figs[1].add_trace(go.Scatter(x=cv["recall"], y=cv["precision"], name=name, line=dict(color=col, width=2)))
    figs[2].add_trace(go.Scatter(x=cv["cal_pred"], y=cv["cal_obs"], name=name, mode="lines+markers",
                                 line=dict(color=col, width=2), marker=dict(size=8)))
figs[0].add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="No skill", line=dict(color=ui.MUTED, dash="dash", width=1)))
base = float(y.dropna().mean())
figs[1].add_hline(y=base, line=dict(color=ui.MUTED, dash="dash", width=1), annotation_text="base rate")
mx = max(0.05, *(float(probas[n].max()) for n in chosen))
figs[2].add_trace(go.Scatter(x=[0, mx], y=[0, mx], name="Perfect calibration", line=dict(color=ui.MUTED, dash="dash", width=1)))
titles = [("ROC curve", "False positive rate", "True positive rate"),
          ("Precision-recall", "Recall", "Precision"),
          ("Calibration (quantile bins)", "Mean predicted probability", "Observed crash frequency")]
for fig, (t, xt, yt), col in zip(figs, titles, (c1, c2, c3)):
    fig.update_layout(title=t, xaxis_title=xt, yaxis_title=yt, height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.3))
    col.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ events
st.subheader(f"Episode-level review - {s['model_name']}")
main = s["model_name"] if s["model_name"] in probas else chosen[0]
summ, events = model_mod.evaluate_alarms(probas[main], y, episodes, s["alarm"], H)
if len(events):
    st.dataframe(events.style.format({"depth": "{:.1%}", "max_proba_before": "{:.1%}", "lead_days": "{:.0f}"}, na_rep="—"),
                 width="stretch")
    st.caption(f"A crash counts as anticipated if the alarm fired at least once in the {H} trading days before the "
               "threshold was breached. Lead = trading days between the first such alarm and the breach.")
else:
    st.info("No fast crash episode in the out-of-sample period with these settings.")

st.download_button("Download out-of-sample probabilities (CSV)",
                   pd.DataFrame(probas).assign(crash_label=y).to_csv().encode(),
                   file_name="oos_crash_probabilities.csv", mime="text/csv")
