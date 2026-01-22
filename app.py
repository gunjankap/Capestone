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

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
col1, col2 = st.columns([1, 6])

with col1:
    st.image("college_logo.jpg", width=90)

with col2:
    st.markdown(
        """
        <div style="line-height:1.6; text-align:right;">
            <div style="font-size:16px; font-weight:700; color:#0b2e73;">
                Gunjan Kapoor
            </div>
            <div style="font-size:13px;">
                Roll No: <b>EMBADTA24003</b>
            </div>
            <div style="font-size:13px;">
                Mentor: <b>Dr. Manish Sarkhel</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------
# DATA LOADERS
# ------------------------------------------------
@st.cache_data
def load_bike():
    day = pd.read_csv("day.csv")
    hour = pd.read_csv("hour.csv")

    drop_cols = ["instant", "dteday", "casual", "registered"]
    day = day.drop(columns=[c for c in drop_cols if c in day.columns])
    hour = hour.drop(columns=[c for c in drop_cols if c in hour.columns])

    return day, hour


@st.cache_data
def load_aqi():
    df = pd.read_csv("aqi.csv")
    df.columns = df.columns.str.strip()
    numeric = df.select_dtypes(include=np.number)
    return df, numeric


day, hour = load_bike()
aqi_raw, aqi = load_aqi()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("📊 AI Model Analysis")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Bike Dataset - Day", "Bike Dataset - Hour", "AQI"]
)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)"]
)

# ------------------------------------------------
# DATASET SELECTION
# ------------------------------------------------
if dataset_choice == "Bike Dataset - Day":
    df = day.copy()
    target = "cnt"
    group_cols = ["season", "weathersit", "workingday"]

elif dataset_choice == "Bike Dataset - Hour":
    df = hour.copy()
    target = "cnt"
    group_cols = ["season", "weathersit", "workingday"]

else:
    target = st.sidebar.selectbox(
        "Select AQI Target",
        aqi.columns.tolist()
    )
    df = aqi.copy()
    group_cols = []  # will be created dynamically

# ------------------------------------------------
# TRAIN / TEST
# ------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
def train_model(name):
    if name == "Linear Regression":
        m = LinearRegression()

    elif name == "Decision Tree":
        m = DecisionTreeRegressor(max_depth=8, random_state=42)

    else:
        m = RandomForestRegressor(n_estimators=220, random_state=42)

    m.fit(X_train, y_train)
    return m, m.predict(X_test)


model, preds = train_model(model_choice)

# ------------------------------------------------
# METRICS
# ------------------------------------------------
mae = round(mean_absolute_error(y_test, preds), 2)
rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
r2 = round(r2_score(y_test, preds), 3)

st.markdown(
    """
    <h1 style="text-align:center; color:#0a2a66; font-style:italic;">
        ✨🤖 AI Model Evaluation Dashboard 🚀
    </h1>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", mae)
c2.metric("RMSE", rmse)
c3.metric("R² Score", r2)

# ------------------------------------------------
# VISUAL DIAGNOSTICS
# ------------------------------------------------
st.markdown(
    "<h4 style='text-align:center; color:#0b2e73;'>📊 Model Performance Diagnostics</h4>",
    unsafe_allow_html=True
)

v1, v2, v3 = st.columns(3)

with v1:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(x=y_test, y=preds, s=12, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
    ax.set_title("Actual vs Predicted", fontsize=9)
    st.pyplot(fig)

with v2:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(x=preds, y=y_test - preds, s=12, ax=ax)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_title("Residual Plot", fontsize=9)
    st.pyplot(fig)

with v3:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(y_test - preds, kde=True, ax=ax)
    ax.set_title("Error Distribution", fontsize=9)
    st.pyplot(fig)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
if model_choice == "Random Forest (Ensemble)":
    fi = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(5)
    )

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title("Top 5 Feature Importance", fontsize=9)
    st.pyplot(fig)

# ------------------------------------------------
# BLIND SPOT ANALYSIS (SAFE)
# ------------------------------------------------
st.markdown(
    "<h5 style='color:#0b2e73;'>⚠️ Blind Spot / Subgroup Error Analysis</h5>",
    unsafe_allow_html=True
)

blind_df = X_test.copy()
blind_df["actual"] = y_test
blind_df["pred"] = preds

# AQI bins
if dataset_choice == "AQI":
    blind_df["TEMP_BIN"] = pd.qcut(blind_df.iloc[:, 0], 4, duplicates="drop")
    blind_df["HUM_BIN"] = pd.qcut(blind_df.iloc[:, 1], 4, duplicates="drop")
    group_cols = ["TEMP_BIN", "HUM_BIN"]

# ---- RMSE TABLES ----
rmse_tables = {}
for g in group_cols:
    rmse_tables[g] = (
        blind_df
        .groupby(g, include_groups=False)
        .apply(lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"])))
        .reset_index(name="RMSE")
    )

cols = st.columns(len(group_cols))
for col, g in zip(cols, group_cols):
    with col:
        st.markdown(
            f"<div style='text-align:center; font-size:14px; font-weight:700; color:#0b2e73;'>"
            f"Subgroup RMSE — {g}</div>",
            unsafe_allow_html=True
        )
        st.dataframe(rmse_tables[g], height=180, width="stretch")

# ------------------------------------------------
# CMBS CHECK
# ------------------------------------------------
st.markdown(
    "<h5 style='color:#0b2e73;'>🧠 CMBS — Collective Model Blind Spot Check</h5>",
    unsafe_allow_html=True
)

blind_df["lr"] = LinearRegression().fit(X_train, y_train).predict(X_test)
blind_df["tree"] = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train).predict(X_test)
blind_df["rf"] = RandomForestRegressor(n_estimators=200).fit(X_train, y_train).predict(X_test)

def cmbs(df, group):
    base = np.sqrt(mean_squared_error(df["actual"], df["rf"]))
    out = {}

    for g in df[group].unique():
        sub = df[df[group] == g]
        vals = {
            "lr": round(np.sqrt(mean_squared_error(sub["actual"], sub["lr"])), 2),
            "tree": round(np.sqrt(mean_squared_error(sub["actual"], sub["tree"])), 2),
            "rf": round(np.sqrt(mean_squared_error(sub["actual"], sub["rf"])), 2),
        }
        vals["Collective_BlindSpot"] = all(v > base * 1.25 for v in vals.values())
        out[g] = vals

    return pd.DataFrame(out).T.reset_index()

cmbs_cols = st.columns(len(group_cols))
for col, g in zip(cmbs_cols, group_cols):
    with col:
        st.markdown(
            f"<div style='text-align:center; font-weight:700; color:#0b2e73;'>CMBS — {g}</div>",
            unsafe_allow_html=True
        )
        st.dataframe(cmbs(blind_df, g), height=220, width="stretch")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown(
    """
    <div style="
        margin-top:20px;
        padding:12px;
        text-align:center;
        background:linear-gradient(135deg,#e8f0ff,#ffffff);
        border-radius:10px;
        font-weight:700;
        color:#0b2e73;">
        ✨ Analysis Completed Successfully — Insights Unlocked!
    </div>
    """,
    unsafe_allow_html=True
)
