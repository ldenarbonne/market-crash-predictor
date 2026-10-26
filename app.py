"""Market Crash Lab - Streamlit entry point.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from crashlab import data as data_mod
from crashlab import ui
from crashlab.model import MODEL_NAMES

st.set_page_config(page_title="Market Crash Lab", page_icon="📉", layout="wide")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📉 Market Crash Lab")

    st.subheader("Data")
    source = st.selectbox(
        "Source",
        ["Auto", "Repository snapshot", "Yahoo Finance (live)", "Upload CSV", "Synthetic demo"],
        help="Auto = committed snapshot in data/ if present, else Yahoo Finance, else synthetic demo.",
    )
    index_name = st.selectbox("Equity index", list(data_mod.INDICES), index=0, disabled=source in ("Synthetic demo", "Upload CSV"))
    start = st.date_input("History starts", value=dt.date(1990, 1, 2), min_value=dt.date(1950, 1, 1),
                          help="VIX starts in 1990; earlier history drops the implied-volatility features.")
    upload = st.file_uploader("CSV file", type="csv") if source == "Upload CSV" else None
    seed = st.number_input("Synthetic seed", value=11, step=1) if source == "Synthetic demo" else 11

    st.subheader("Crash definition")
    threshold = st.slider("Drawdown threshold", 0.05, 0.40, 0.15, 0.01, format="%.2f",
                          help="A crash = peak-to-trough loss at least this large...")
    horizon = st.slider("Horizon (trading days)", 10, 252, 63, 1, help="...realised within this many trading days.")

    st.subheader("Model")
    model_name = st.selectbox("Main model", MODEL_NAMES[:2])
    min_train_years = st.slider("Initial training window (years)", 5, 15, 8)
    step = st.select_slider("Refit every (trading days)", options=[21, 63, 126, 252], value=63)
    balanced = st.checkbox("Balance classes", value=False,
                           help="Re-weights rare crash days. Improves recall but distorts probabilities (no longer calibrated).")
    alarm = st.slider("Alarm threshold (probability)", 0.05, 0.90, 0.30, 0.05)

    st.caption("Educational tool - not investment advice.")

# ---------------------------------------------------------------- data load
df, source_used = ui.load_market(source, data_mod.INDICES[index_name], start.isoformat(), upload, int(seed))
if len(df) < 252 * (min_train_years + 3):
    st.error(f"Not enough history ({len(df)} rows) for an initial training window of {min_train_years} years. "
             "Choose an earlier start date or a shorter training window.")
    st.stop()

st.session_state["settings"] = {
    "df": df,
    "source_used": source_used,
    "index_name": index_name if source not in ("Synthetic demo", "Upload CSV") else source_used,
    "threshold": float(threshold),
    "horizon": int(horizon),
    "model_name": model_name,
    "min_train_years": int(min_train_years),
    "step": int(step),
    "balanced": bool(balanced),
    "alarm": float(alarm),
}

pages = [
    st.Page("views/overview.py", title="Crash risk today", icon="🚦", default=True),
    st.Page("views/backtest.py", title="Model backtest", icon="🧪"),
    st.Page("views/features.py", title="Data & indicators", icon="📊"),
    st.Page("views/lppls_page.py", title="LPPLS bubble diagnostics", icon="🫧"),
    st.Page("views/methodology.py", title="Methodology", icon="📚"),
]
st.navigation(pages).run()
