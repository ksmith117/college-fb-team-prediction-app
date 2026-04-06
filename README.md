# College Football Team Prediction App

This project is a machine learning web application that models relationships between team performance metrics and postseason outcomes in college football. The app allows users to generate forward and reverse predictions within specific conferences using real data from the 2025–2026 season.

---

## Overview

The application predicts:

- Postseason qualification (0 = No, 1 = Yes)
- Probability of making the postseason
- Conference rank
- Postseason efficiency
- Efficiency tier classification

In this model, **postseason qualification represents making a bowl game**.

---

## Features

### Forward Prediction
Inputs:
- Availability
- Conference Win Percentage

Outputs:
- Postseason Qualification
- Postseason Qualification Probability
- Conference Rank
- Postseason Efficiency
- Efficiency Tier

---

### Reverse Prediction
Inputs:
- Postseason Qualification
- Conference Rank
- Postseason Efficiency

Outputs:
- Predicted Availability
- Predicted Conference Win Percentage
- Efficiency Tier

---

## Data Sources

- Data was compiled from official athletics websites for the **Power 4 conferences (ACC, Big Ten, Big 12, SEC)** during the 2025–2026 college football season
- Postseason performance data was aggregated and standardized across conferences
- Postseason results were weighted based on game importance (e.g., bowl games, conference championship implications, College Football Playoff relevance)
- Metrics were aligned across sources to ensure consistency

---

## Postseason Efficiency Metric

Postseason efficiency is a custom metric designed to capture both performance and level of competition.

It is defined as: Efficiency = Weighted Win % × log(Total Postseason Weight)


Where:
- **Weighted Win %** = postseason success adjusted for game importance
- **Total Weight** = cumulative importance of postseason games

This metric rewards teams that:
- Perform well in postseason settings
- Sustain performance across higher-impact games

Higher efficiency values are more consistent with teams that compete for conference championship appearances and the College Football Playoff.

---

## Efficiency Tier Classification

Efficiency values are categorized into tiers based on observed data distribution:

| Range | Tier |
|------|------|
| 0.00 | No Postseason Appearance |
| 0.01 – 0.74 | Below Average |
| 0.75 – 1.24 | Average |
| 1.25 – 1.99 | Strong |
| 2.00+ | Elite |

These ranges are specific to the football dataset.

---

## Model Performance

### Postseason Classification Model
- Accuracy: **0.857**
- Precision: **0.889**
- Recall: **0.889**
- F1 Score: **0.889**
- ROC AUC: **0.933**

This model demonstrates strong ability to distinguish between postseason and non-postseason teams.

---

### Conference Rank Model
- MAE: **1.246**
- RMSE: **1.447**
- R²: **0.799**

The model predicts conference rank within approximately one position on average.

---

### Postseason Efficiency Model
- MAE: **0.126**
- RMSE: **0.223**
- R²: **-0.111**

Performance was limited, suggesting that efficiency is influenced by factors not fully captured by the input variables.

---

### Reverse Models

#### Conference Win Percentage
- MAE: **0.057**
- RMSE: **0.065**
- R²: **0.901**

Strong predictive performance.

#### Availability
- MAE: **0.036**
- RMSE: **0.050**
- R²: **-0.797**

Weak predictive performance, indicating availability cannot be reliably inferred from outcomes.

---

## Limitations

- Predictions are based on historical patterns and should be interpreted as estimates
- Reverse models are less reliable for certain variables
- Efficiency is a custom metric and may not capture all aspects of team performance
- Differences in conference reporting may affect comparability

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Google Sheets (data preparation)

---

## Purpose

This project demonstrates:
- Feature engineering using a custom efficiency metric
- Supervised machine learning (classification and regression)
- Model evaluation and interpretation
- Applied sports analytics
- End-to-end workflow from raw data to deployed application

---

## Live App

(https://cfb-team-prediction-app-gy6skrf3epakqspwxgkhhx.streamlit.app/)
