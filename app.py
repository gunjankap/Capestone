##############################################
# STREAMLIT APP – BIKE SHARING AI ANALYSIS
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Bike Sharing AI Model Analysis",
                   page_icon="🚴‍♀️", layout="wide")

##############################################
# LOAD DATA
##############################################
@st.cache_data
def load_data():
    day = pd.read_csv("day.csv")
    hour = pd.read_csv("hour.csv")

    drop_cols = ["instant","dteday","casual","registered"]
    day = day.drop(columns=[c for c in drop_cols if c in day.columns])
    hour = hour.drop(columns=[c for c in drop_cols if c in hour.columns])

    return day, hour

day, hour = load_data()

##############################################
# Sidebar
##############################################
st.sidebar.title("🚴 Bike Sharing Analysis")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Day", "Hour"])
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)"]
)

df = day if dataset_choice=="Day" else hour
target = "cnt"

##############################################
# Split & Scale
##############################################
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

##############################################
# Train Models
##############################################
def build_model(name):
    if name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    elif name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=8)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    else:
        model = RandomForestRegressor(n_estimators=220, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    return model, preds

model, preds = build_model(model_choice)

##############################################
# UI Layout
##############################################
st.title("🚴 Bike Sharing AI Model Evaluation Dashboard")
st.markdown(f"### Dataset Used: **{dataset_choice}**")
st.markdown(f"### Model: **{model_choice}**")

##############################################
# Metrics
##############################################
c1, c2, c3 = st.columns(3)
c1.metric("MAE", round(mean_absolute_error(y_test, preds),2))
c2.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, preds)),2))
c3.metric("R² Score", round(r2_score(y_test, preds),3))

##############################################
# Actual vs Pred Plot
##############################################
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=preds, ax=ax)
ax.set_xlabel("Actual Count")
ax.set_ylabel("Predicted Count")
ax.set_title("Actual vs Predicted Bike Demand")
st.pyplot(fig)

##############################################
# FEATURE IMPORTANCE (Explainability Alternative to SHAP)
##############################################
if model_choice == "Random Forest (Ensemble)":
    st.subheader("🔍 Feature Importance (Explainability)")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feat_imp)

##############################################
# BLIND SPOT ANALYSIS
##############################################
st.subheader("⚠️ Blind Spot / Subgroup Error Analysis")

group_cols = ["season","weathersit","workingday"]
available = [g for g in group_cols if g in df.columns]

blind_df = X_test.copy()
blind_df["actual"] = y_test
blind_df["pred"] = preds

for g in available:
    st.write(f"### Subgroup RMSE by **{g}**")
    subgroup_rmse = blind_df.groupby(g).apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    )
    st.table(subgroup_rmse)

##############################################
# CMBS CHECK
##############################################
st.subheader("🧠 CMBS — Collective Model Blind Spot Check")

blind_df["lr"] = LinearRegression().fit(X_train, y_train).predict(X_test)
blind_df["tree"] = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train).predict(X_test)
blind_df["rf"] = RandomForestRegressor(n_estimators=200).fit(X_train, y_train).predict(X_test)

def cmbs_check(df, group_col, preds=["lr","tree","rf"], threshold=0.25):
    results = {}
    base = np.sqrt(mean_squared_error(df["actual"], df["rf"])) 

    for g in df[group_col].unique():
        sub = df[df[group_col]==g]
        res = {}
        for p in preds:
            res[p] = round(np.sqrt(mean_squared_error(sub["actual"], sub[p])),2)

        res["Collective_BlindSpot"] = all(
            np.sqrt(mean_squared_error(sub["actual"], sub[p])) > base*(1+threshold)
            for p in preds
        )
        results[g] = res
    return pd.DataFrame(results).T

for g in available:
    st.write(f"### CMBS Results by **{g}**")
    st.table(cmbs_check(blind_df, g))

st.success("Analysis Completed Successfully ✅")

