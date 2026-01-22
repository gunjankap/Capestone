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

# ---------- STUDENT HEADER ----------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("college_logo.jpg", width=90)
with col2:
    st.markdown("""
    <div style="line-height:1.6; text-align:right;">
        <div style="font-size:16px; font-weight:700; color:#0b2e73;">Gunjan Kapoor</div>
        <div style="font-size:13px;">Roll No: <b>EMBADTA24003</b></div>
        <div style="font-size:13px;">Mentor: <b>Dr. Manish Sarkhel</b></div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

##############################################
# LOAD DATA
##############################################
@st.cache_data
def load_bike():
    day = pd.read_csv("day.csv")
    hour = pd.read_csv("hour.csv")
    drop_cols = ["instant","dteday","casual","registered"]
    return (
        day.drop(columns=[c for c in drop_cols if c in day.columns]),
        hour.drop(columns=[c for c in drop_cols if c in hour.columns])
    )

@st.cache_data
def load_aqi():
    df = pd.read_csv("aqi.csv")
    df.columns = df.columns.str.strip()
    return df.select_dtypes(include=np.number)

day, hour = load_bike()
aqi = load_aqi()

##############################################
# SIDEBAR
##############################################
st.sidebar.title("📊 AI Model Analysis")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Bike Dataset - Day", "Bike Dataset - Hour", "AQI"]
)
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)"]
)

##############################################
# DATASET HANDLING
##############################################
if dataset_choice == "Bike Dataset - Day":
    df, target = day.copy(), "cnt"
elif dataset_choice == "Bike Dataset - Hour":
    df, target = hour.copy(), "cnt"
else:
    target = st.sidebar.selectbox("AQI Target", aqi.columns)
    df = aqi.copy()

##############################################
# TRAIN / TEST
##############################################
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

##############################################
# MODEL
##############################################
def build_model(name):
    if name == "Linear Regression":
        m = LinearRegression()
    elif name == "Decision Tree":
        m = DecisionTreeRegressor(max_depth=8)
    else:
        m = RandomForestRegressor(n_estimators=200, random_state=42)
    m.fit(X_train, y_train)
    return m, m.predict(X_test)

model, preds = build_model(model_choice)

##############################################
# METRICS
##############################################
mae = round(mean_absolute_error(y_test, preds),2)
rmse = round(mean_squared_error(y_test, preds, squared=False),2)
r2 = round(r2_score(y_test, preds),3)

c1, c2, c3 = st.columns(3)
for c, name, val in zip([c1,c2,c3], ["MAE","RMSE","R2"], [mae,rmse,r2]):
    with c:
        st.markdown(f"<h5 style='text-align:center'>{name}</h5>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;font-size:20px'><b>{val}</b></p>", unsafe_allow_html=True)

##############################################
# PERFORMANCE PLOTS
##############################################
st.markdown("### 📊 Model Performance Diagnostics")
c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(x=y_test, y=preds, s=10, ax=ax)
    ax.set_title("Actual vs Pred")
    st.pyplot(fig)

with c2:
    res = y_test - preds
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(x=preds, y=res, s=10, ax=ax)
    ax.axhline(0)
    ax.set_title("Residuals")
    st.pyplot(fig)

with c3:
    fig, ax = plt.subplots(figsize=(3,3))
    sns.histplot(res, kde=True, ax=ax)
    ax.set_title("Error Distribution")
    st.pyplot(fig)

##############################################
# FEATURE IMPORTANCE (SAFE)
##############################################
if model_choice == "Random Forest (Ensemble)":
    fi = pd.Series(model.feature_importances_, index=X.columns).nlargest(5)
    fig, ax = plt.subplots(figsize=(3,2))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title("Top 5 Feature Importance")
    st.pyplot(fig)

##############################################
# BLIND SPOT ANALYSIS (FIXED)
##############################################
st.markdown("### ⚠️ Blind Spot / Subgroup Error Analysis")

blind_df = X_test.copy()
blind_df["actual"] = y_test
blind_df["pred"] = preds

if dataset_choice.startswith("Bike"):
    available = [c for c in ["season","weathersit","workingday"] if c in df.columns]
else:
    blind_df["TEMP_BIN"] = pd.qcut(blind_df.iloc[:,0], 4, duplicates="drop")
    blind_df["HUM_BIN"] = pd.qcut(blind_df.iloc[:,1], 4, duplicates="drop")
    available = ["TEMP_BIN","HUM_BIN"]

rmse_tables = {}
for g in available:
    rmse_tables[g] = (
        blind_df.groupby(g)
        .apply(lambda x: mean_squared_error(x["actual"], x["pred"], squared=False))
        .reset_index(name="RMSE")
    )

cols = st.columns(len(available))
for col, g in zip(cols, available):
    with col:
        st.markdown(f"**Subgroup RMSE — {g}**")
        st.dataframe(rmse_tables[g], height=180)

##############################################
# CMBS (FIXED)
##############################################
st.markdown("### 🧠 CMBS — Collective Model Blind Spot Check")

blind_df["lr"] = LinearRegression().fit(X_train,y_train).predict(X_test)
blind_df["tree"] = DecisionTreeRegressor(max_depth=8).fit(X_train,y_train).predict(X_test)
blind_df["rf"] = RandomForestRegressor(n_estimators=200).fit(X_train,y_train).predict(X_test)

def cmbs(df, g):
    base = mean_squared_error(df["actual"], df["rf"], squared=False)
    out = []
    for v in df[g].unique():
        sub = df[df[g]==v]
        out.append([
            v,
            mean_squared_error(sub["actual"], sub["lr"], squared=False),
            mean_squared_error(sub["actual"], sub["tree"], squared=False),
            mean_squared_error(sub["actual"], sub["rf"], squared=False),
        ])
    return pd.DataFrame(out, columns=[g,"lr","tree","rf"])

cols = st.columns(len(available))
for col, g in zip(cols, available):
    with col:
        st.markdown(f"**CMBS — {g}**")
        st.dataframe(cmbs(blind_df, g), height=200)

##############################################
# FOOTER
##############################################
st.markdown("""
<div style="padding:12px;text-align:center;font-weight:700;color:#0b2e73;">
✨ Analysis Completed Successfully — Results Ready!
</div>
""", unsafe_allow_html=True)
