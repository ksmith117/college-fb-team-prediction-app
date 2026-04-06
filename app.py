import pandas as pd
import streamlit as st

st.set_page_config(page_title="College Football Prediction App", layout="centered")

st.title("College Football Team Prediction App")
st.write("DEBUG VERSION LOADED")

data = pd.read_csv("FB_All_Conf.csv")

data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w_]", "", regex=True)
)

data = data.rename(columns={
    "conference_ranking": "conf_rank",
    "conference_rank": "conf_rank",
    "postseasoneff": "postseason_eff",
    "postseason_efficiency": "postseason_eff",
    "weighted_postseason_eff": "postseason_eff",
    "conf_win_percent": "conf_win_pct",
    "conf_win_percentage": "conf_win_pct",
    "conference_win_pct": "conf_win_pct",
    "conference_win_percentage": "conf_win_pct",
})

# remove duplicate columns after renaming
data = data.loc[:, ~data.columns.duplicated()].copy()

st.subheader("Columns")
st.write(data.columns.tolist())

st.subheader("Raw dtypes")
st.write(data.dtypes)

# clean likely numeric columns
for col in ["availability", "conf_win_pct", "postseason", "conf_rank", "postseason_eff"]:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

if "availability" in data.columns and data["availability"].dropna().max() > 1:
    data["availability"] = data["availability"] / 100

st.subheader("Cleaned dtypes")
st.write(data.dtypes)

st.subheader("Sample rows")
cols_to_show = [c for c in ["conference", "availability", "conf_win_pct", "postseason", "conf_rank", "postseason_eff"] if c in data.columns]
st.write(data[cols_to_show].head(10))

st.stop()
