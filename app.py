##############################################
# STREAMLIT APP – BIKE + AQI AI ANALYSIS
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

st.set_page_config(page_title="AI Model Evaluation Dashboard",
                   page_icon="📊", layout="wide")

##############################################
# LOAD DATA
##############################################
@st.cache_data
def load_bike():
    day = pd.read_csv("day.csv")
    hour = pd.read_csv("hour.csv")

    drop_cols = ["instant","dteday","casual","registered"]
    day = day.drop(columns=[c for c in drop_cols if c in day.columns])
    hour = hour.drop(columns=[c for c in drop_cols if c in hour.columns])
    return day, hour

@st.cache_data
def load_aqi():
    df = pd.read_csv("aqi.csv")
    # clean column names
    df.columns = df.columns.str.strip()
    # drop non-numeric for ML
    keep = df.select_dtypes(include=np.number)
    return df, keep

day, hour = load_bike()
aqi_raw, aqi = load_aqi()

##############################################
# Sidebar
##############################################
st.sidebar.title("📊 AI Model Analysis")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Day", "Hour", "AQI"]
)

model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)"]
)

##############################################
# DATASET HANDLING
##############################################
if dataset_choice == "Day":
    df = day.copy()
    target = "cnt"

elif dataset_choice == "Hour":
    df = hour.copy()
    target = "cnt"

else:
    st.sidebar.markdown("### AQI Target Selection")
    target = st.sidebar.selectbox(
        "Select Prediction Target",
        [c for c in aqi.columns if c not in ["Date","Time"]]
    )
    df = aqi.copy()

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
st.markdown(
    """
    <h1 style="
        text-align:center; 
        color:#0a2a66; 
        font-style:italic;
        font-weight:800;
    ">
        ✨🤖 AI Model Evaluation Dashboard 🚀
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="font-size:18px; font-weight:600; padding:8px; 
                border-radius:8px; background:#f7f9fc;">
        Dataset: <span style="color:#2e7fe8">{dataset_choice}</span> |
        Model: <span style="color:#2e7fe8">{model_choice}</span> |
        Target: <span style="color:#2e7fe8">{target}</span>
    </div>
    """,
    unsafe_allow_html=True
)


##############################################
# Metrics
##############################################

mae = round(mean_absolute_error(y_test, preds),2)
rmse = round(np.sqrt(mean_squared_error(y_test, preds)),2)
r2 = round(r2_score(y_test, preds),3)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"<h5 style='text-align:center; color:#5a6bbf;'>MAE</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{mae}</p>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<h5 style='text-align:center; color:#5a6bbf;'>RMSE</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{rmse}</p>", unsafe_allow_html=True)

with c3:
    st.markdown(f"<h5 style='text-align:center; color:#5a6bbf;'>R2 Score</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{r2}</p>", unsafe_allow_html=True)



##############################################
# Actual vs Pred Plot
##############################################
c1, c2, c3 = st.columns(3)

# -------- VISUAL 1: Actual vs Predicted --------
with c1:
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    sns.scatterplot(x=y_test, y=preds, s=12, color="#2e7fe8", ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            '--', color='gray', linewidth=1)
    ax.set_title("Actual vs Predicted", fontsize=10)
    ax.set_xlabel("Actual", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    st.pyplot(fig)

# -------- VISUAL 2: Residual Plot --------
with c2:
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    sns.scatterplot(x=preds, y=residuals, s=12, color="#e86f2e", ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title("Residual Plot", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Residuals", fontsize=8)
    st.pyplot(fig)

# -------- VISUAL 3: Error Distribution --------
with c3:
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    sns.histplot(residuals, kde=True, color="#4c9c4c", ax=ax)
    ax.set_title("Prediction Error Distribution", fontsize=10)
    ax.set_xlabel("Error", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    st.pyplot(fig)

##############################################
# FEATURE IMPORTANCE
##############################################
if model_choice == "Random Forest (Ensemble)":
    st.subheader("🔍 Feature Importance")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(feat_imp)

##############################################
# BLIND SPOT ANALYSIS
##############################################
st.subheader("⚠️ Blind Spot / Subgroup Error Analysis")

blind_df = X_test.copy()
blind_df["actual"] = y_test
blind_df["pred"] = preds

# Bike Grouping
if dataset_choice in ["Day","Hour"]:
    group_cols = ["season","weathersit","workingday"]
    available = [g for g in group_cols if g in df.columns]

# AQI Grouping
else:
    blind_df["TEMP_BIN"] = pd.qcut(blind_df.iloc[:,0], 4, duplicates="drop")
    blind_df["HUM_BIN"] = pd.qcut(blind_df.iloc[:,1], 4, duplicates="drop")
    available = ["TEMP_BIN","HUM_BIN"]

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


