import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Team Prediction App", layout="centered")

st.title("College Football Team Prediction App")

st.markdown("""
### About This

This tool uses machine learning to model relationships between team performance metrics and postseason outcomes.

---

### How to Use

**Forward Mode**
- Select a conference
- Input: Availability + Conference Win %
- Output: Postseason (0/1), Conference Rank, Postseason Efficiency

**Reverse Mode**
- Select a conference
- Input: Postseason, Conference Rank, Postseason Efficiency
- Output: Estimated Availability and Conference Win %

---

### Important Notes

- Predictions are based on historical patterns, not guarantees
- Reverse predictions are approximate (multiple input combinations can produce similar outcomes)
- Results should be interpreted as **estimates**, not exact values
- Source data were compiled from official conference athletics websites for the 2025–2026 football season

---

### Purpose

This app is designed to explore:
- How team performance metrics relate to postseason success
- What conditions might lead to certain outcomes
- How results differ across conferences
""")

st.write("Forward and reverse prediction tool")

# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("FB_All_Conf.csv")
data.columns = data.columns.str.strip()

# Fix availability: "91.13%" -> 0.9113
data["availability"] = (
    data["availability"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

# Drop missing values in needed columns
data = data.dropna(subset=[
    "conference",
    "availability",
    "conf_win_pct",
    "postseason",
    "conf_rank",
    "weighted_postseason_eff"
])

# -----------------------
# CONFERENCE SELECTION
# -----------------------
conferences = sorted(data["conference"].dropna().unique())
selected_conf = st.selectbox("Select Conference", conferences)

filtered_data = data[data["conference"] == selected_conf]

if len(filtered_data) < 5:
    st.warning("Not enough data for this conference to build a reliable model.")
    st.stop()

st.caption(f"Using data for: {selected_conf}")

# -----------------------
# TRAIN FORWARD MODELS
# Inputs: availability, conf_win_pct
# Outputs: postseason, conf_rank, weighted_postseason_eff
# -----------------------
X1 = filtered_data[["availability", "conf_win_pct"]]

y_post = filtered_data["postseason"]
y_rank = filtered_data["conf_rank"]
y_eff = filtered_data["weighted_postseason_eff"]

model_post = RandomForestClassifier(random_state=42)
model_rank = RandomForestRegressor(random_state=42)
model_eff = RandomForestRegressor(random_state=42)

model_post.fit(X1, y_post)
model_rank.fit(X1, y_rank)
model_eff.fit(X1, y_eff)

# -----------------------
# TRAIN REVERSE MODELS
# Inputs: postseason, conf_rank, weighted_postseason_eff
# Outputs: availability, conf_win_pct
# -----------------------
X2 = filtered_data[["postseason", "conf_rank", "weighted_postseason_eff"]]

y_avail = filtered_data["availability"]
y_conf = filtered_data["conf_win_pct"]

model_avail = RandomForestRegressor(random_state=42)
model_conf = RandomForestRegressor(random_state=42)

model_avail.fit(X2, y_avail)
model_conf.fit(X2, y_conf)

# -----------------------
# APP MODE SWITCH
# -----------------------
mode = st.radio("Choose Prediction Mode", ["Forward", "Reverse"])

# -----------------------
# FORWARD MODE
# -----------------------
if mode == "Forward":
    st.subheader(f"Forward Prediction — {selected_conf}")

    availability = st.number_input(
        "Availability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01
    )

    conf_win_pct = st.number_input(
        "Conference Win %",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01
    )

    if st.button("Predict Forward"):
        X = pd.DataFrame([{
            "availability": availability,
            "conf_win_pct": conf_win_pct
        }])

        post = int(model_post.predict(X)[0])
        prob = float(model_post.predict_proba(X)[0][1])
        rank = float(model_rank.predict(X)[0])
        eff = float(model_eff.predict(X)[0])

        st.write("### Results")
        st.write(f"**Conference:** {selected_conf}")
        st.write(f"**Postseason:** {post}")
        st.write(f"**Probability of Postseason:** {round(prob, 3)}")
        st.write(f"**Conference Rank:** {round(rank, 2)}")
        st.write(f"**Weighted Postseason Efficiency:** {round(eff, 3)}")

# -----------------------
# REVERSE MODE
# -----------------------
else:
    st.subheader(f"Reverse Prediction — {selected_conf}")

    postseason = st.number_input(
        "Postseason (0 or 1)",
        min_value=0,
        max_value=1,
        value=1,
        step=1
    )

    conf_rank = st.number_input(
        "Conference Rank",
        min_value=1.0,
        value=5.0,
        step=0.1
    )

    weighted_postseason_eff = st.number_input(
        "Weighted Postseason Efficiency",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

    if st.button("Predict Reverse"):
        X = pd.DataFrame([{
            "postseason": postseason,
            "conf_rank": conf_rank,
            "weighted_postseason_eff": weighted_postseason_eff
        }])

        avail_pred = float(model_avail.predict(X)[0])
        conf_pred = float(model_conf.predict(X)[0])

        st.write("### Results")
        st.write(f"**Conference:** {selected_conf}")
        st.write(f"**Predicted Availability:** {round(avail_pred, 3)}")
        st.write(f"**Predicted Conference Win %:** {round(conf_pred, 3)}")

st.markdown("""
---
### Notes
- Forward predictions estimate postseason outcome, conference rank, and weighted postseason efficiency.
- Reverse predictions estimate the availability and conference win percentage associated with a given outcome profile.
- Predictions are generated using only the selected conference's data.
- Predictions are based on historical data patterns and should be interpreted as estimates, not guarantees.
""")
