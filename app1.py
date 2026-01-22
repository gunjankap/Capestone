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
# HEADER
##############################################
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
                Roll No: <b>EMBADTA24003</b><br>
                Mentor: <b>Dr. Manish Sarkhel</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

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
    df.columns = df.columns.str.strip()
    df.replace(-200, np.nan, inplace=True)
    df = df.dropna()
    numeric = df.select_dtypes(include=np.number)
    return numeric

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
# DATASET SELECTION
##############################################
if dataset_choice == "Bike Dataset - Day":
    df = day.copy()
    target = "cnt"

elif dataset_choice == "Bike Dataset - Hour":
    df = hour.copy()
    target = "cnt"

else:
    target = st.sidebar.selectbox(
        "Select AQI Target",
        aqi.columns
    )
    df = aqi.copy()

##############################################
# SPLIT
##############################################
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

##############################################
# MODEL BUILDER
##############################################
def build_model(name):
    if name == "Linear Regression":
        model = LinearRegression()
    elif name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=8)
    else:
        model = RandomForestRegressor(n_estimators=220, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

model, preds = build_model(model_choice)

##############################################
# METRICS
##############################################
mae = round(mean_absolute_error(y_test, preds),2)
rmse = round(np.sqrt(mean_squared_error(y_test, preds)),2)
r2 = round(r2_score(y_test, preds),3)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", mae)
c2.metric("RMSE", rmse)
c3.metric("R²", r2)

##############################################
# PLOTS
##############################################
c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(x=y_test, y=preds, s=10, ax=ax)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], '--')
    ax.set_title("Actual vs Predicted", fontsize=9)
    st.pyplot(fig)

with c2:
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(x=preds, y=residuals, s=10, ax=ax)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residuals", fontsize=9)
    st.pyplot(fig)

with c3:
    fig, ax = plt.subplots(figsize=(3,3))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Error Distribution", fontsize=9)
    st.pyplot(fig)

##############################################
# FEATURE IMPORTANCE
##############################################
if model_choice == "Random Forest (Ensemble)":
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top = imp.sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(3,2))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top 5 Feature Importance", fontsize=9)
    st.pyplot(fig)

##############################################
# BLIND SPOT ANALYSIS
##############################################
blind_df = X_test.copy()
blind_df["actual"] = y_test.values
blind_df["pred"] = preds

rmse_tables = {}

# BIKE
if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    for col in ["season","weathersit","workingday"]:
        if col in blind_df.columns:
            rmse_tables[col] = (
                blind_df.groupby(col)
                .apply(lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"])))
                .reset_index(name="RMSE")
            )

# AQI
else:
    blind_df["TEMP_BIN"] = pd.qcut(
    blind_df["T"], 4,
    labels=["Cold","Mild","Warm","Hot"]
)

blind_df["HUM_BIN"] = pd.qcut(
    blind_df["RH"], 4,
    labels=["Dry","Normal","Humid","Very Humid"]
)


    for col in ["TEMP_BIN","HUM_BIN"]:
        rmse_tables[col] = (
            blind_df.groupby(col)
            .apply(lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"])))
            .reset_index(name="RMSE")
        )

cols = st.columns(len(rmse_tables))
for ui, (k, v) in zip(cols, rmse_tables.items()):
    with ui:
        st.markdown(f"**Subgroup RMSE — {k}**")
        st.dataframe(v, height=180)

##############################################
# CMBS
##############################################
blind_df["lr"] = LinearRegression().fit(X_train, y_train).predict(X_test)
blind_df["tree"] = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train).predict(X_test)
blind_df["rf"] = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train).predict(X_test)

def cmbs_check(df, group_col, threshold=0.25):
    base = np.sqrt(mean_squared_error(df["actual"], df["rf"]))
    rows = []
    for g in df[group_col].dropna().unique():
        sub = df[df[group_col]==g]
        row = {"Group": g}
        for p in ["lr","tree","rf"]:
            row[p] = round(np.sqrt(mean_squared_error(sub["actual"], sub[p])),2)
        row["Collective_BlindSpot"] = all(
            row[p] > base*(1+threshold) for p in ["lr","tree","rf"]
        )
        rows.append(row)
    return pd.DataFrame(rows)

cmbs_tables = {k: cmbs_check(blind_df, k) for k in rmse_tables.keys()}

cols = st.columns(len(cmbs_tables))
for ui, (k, v) in zip(cols, cmbs_tables.items()):
    with ui:
        st.markdown(f"**CMBS — {k}**")
        st.dataframe(v, height=200)

st.success("✨ Analysis Completed Successfully")
