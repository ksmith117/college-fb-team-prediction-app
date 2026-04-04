import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.title("Team Prediction App")

# Load data
data = pd.read_csv("FB_All_Conf.csv")
data.columns = data.columns.str.strip()

# Fix availability
data["availability"] = (
    data["availability"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

# Train models
X1 = data[["availability", "conf_win_pct"]]

y_post = data["postseason"]
y_rank = data["conf_rank"]
y_eff = data["weighted_postseason_eff"]

model_post = RandomForestClassifier()
model_rank = RandomForestRegressor()
model_eff = RandomForestRegressor()

model_post.fit(X1, y_post)
model_rank.fit(X1, y_rank)
model_eff.fit(X1, y_eff)

# UI
availability = st.number_input("Availability", 0.0, 1.0, 0.9)
conf_win_pct = st.number_input("Conf Win %", 0.0, 1.0, 0.6)

if st.button("Predict"):
    X = pd.DataFrame([{
        "availability": availability,
        "conf_win_pct": conf_win_pct
    }])

    post = model_post.predict(X)[0]
    rank = model_rank.predict(X)[0]
    eff = model_eff.predict(X)[0]

    st.write("Postseason:", int(post))
    st.write("Conf Rank:", round(rank, 2))
    st.write("Post Eff:", round(eff, 3))
