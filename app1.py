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



# ---------- STUDENT & PROJECT HEADER ----------
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
            <div style="font-size:13px; color:#333;">
                Roll No: <b>EMBADTA24003</b>
            </div>
            <div style="font-size:13px; color:#333;">
                Mentor: <b>Dr. Manish Sarkhel</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='margin-top:8px; margin-bottom:8px;'>", unsafe_allow_html=True)


 # ---------- BANNER IMAGE ----------
st.image("banner.png", use_container_width=True)


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
    df = day.copy()
    target = "cnt"

elif dataset_choice == "Bike Dataset - Hour":
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
##############################################)

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

st.markdown(f"""
<div style="font-size:12px; padding:8px; background:#ffffff;
            border:1px dashed #d6e1ff;">
<b>Data Snapshot</b><br>
• Records: {df.shape[0]}<br>
• Features: {X.shape[1]}<br>
• Missing Values: {df.isna().sum().sum()}<br>
• Target Range: {round(y.min(),1)} – {round(y.max(),1)}
</div>
""", unsafe_allow_html=True)


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
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        📊 Model Performance Diagnostics
    </h4>
    """,
    unsafe_allow_html=True
)

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

pred_std = np.std(preds)

st.markdown(f"""
<div style="font-size:12px; padding:8px; background:#fff7f0;
            border-left:4px solid #ff9f40;">
<b>Prediction Stability Indicator</b><br>
• Std Dev of Predictions: {round(pred_std,2)}<br>
• Interpretation: Higher values indicate inconsistent model behavior across samples.
</div>
""", unsafe_allow_html=True)

# ================= FEATURE IMPORTANCE =================
if model_choice == "Random Forest (Ensemble)":

    feat_imp = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    # ---- Top 5 Features ----
    top_feat = feat_imp.head(5)

    fig, ax = plt.subplots(figsize=(3.0, 2.2))  # compact figure

    sns.barplot(
        x=top_feat.values,
        y=top_feat.index,
        color="#2e7fe8",
        ax=ax
    )

    ax.set_title("Top 5 Feature Importance", fontsize=9, color="#0b2e73")
    ax.set_xlabel("Importance", fontsize=8)
    ax.set_ylabel("")

    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)


##############################################
# BLIND SPOT ANALYSIS
##############################################
st.markdown(
    """
    <h5 style='color:#0b2e73;'>
        ⚠️ Blind Spot / Subgroup Error Analysis
    </h5>
    """,
    unsafe_allow_html=True
)

# ---------- Create Blind Spot DF ----------
blind_df = X_test.copy()
blind_df["actual"] = y_test
blind_df["pred"] = preds


# ==========================================
# BIKE DATASET GROUPING
# ==========================================
if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:

    # ---- Subgroup RMSE Tables ----
    season_rmse = blind_df.groupby("season").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    weather_rmse = blind_df.groupby("weathersit").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    working_rmse = blind_df.groupby("workingday").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")


# ==========================================
# AQI DATASET GROUPING
# ==========================================
else:

    # ---- Create Temperature + Humidity Bins ----
    blind_df["TEMP_BIN"] = pd.qcut(
        blind_df.iloc[:, 0], 4, duplicates="drop"
    )

    blind_df["HUM_BIN"] = pd.qcut(
        blind_df.iloc[:, 1], 4, duplicates="drop"
    )

    # ---- RMSE Tables for AQI ----
    temp_rmse = blind_df.groupby("TEMP_BIN").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    hum_rmse = blind_df.groupby("HUM_BIN").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")


# ---------- Observations Box ----------
st.markdown("""
<div style="font-size:13px; padding:12px;
            background:#f4f6fc;
            border-radius:8px;">
<b>Key Observations</b><br>
• High overall accuracy can hide subgroup-specific failures.<br>
• Certain environmental or seasonal conditions show higher RMSE.<br>
• Blind spots indicate structural weaknesses in data coverage.<br>
• Subgroup diagnostics help ensure fairness + robustness.
</div>
""", unsafe_allow_html=True)


# ==========================================
# DISPLAY TABLES
# ==========================================
st.markdown("### 📌 Subgroup RMSE Results")

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Season RMSE**")
        st.dataframe(season_rmse, use_container_width=True)

    with c2:
        st.markdown("**Weather Situation RMSE**")
        st.dataframe(weather_rmse, use_container_width=True)

    with c3:
        st.markdown("**Working Day RMSE**")
        st.dataframe(working_rmse, use_container_width=True)

else:

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temperature Bin RMSE**")
        st.dataframe(temp_rmse, use_container_width=True)

    with c2:
        st.markdown("**Humidity Bin RMSE**")
        st.dataframe(hum_rmse, use_container_width=True)


# -------- SMALL TABLE CSS --------
st.markdown("""
<style>
.small-table table {
    font-size:11px !important;
}
.small-table th {
    font-size:11px !important;
    color:#0b2e73;
}
.small-title{
    font-size:12px;
    color:#0b2e73;
    font-weight:600;
    margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DISPLAY TABLES
# ==========================================
st.markdown("### 📌 Subgroup RMSE Results")

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Season RMSE**")
        st.dataframe(season_rmse, use_container_width=True)

    with c2:
        st.markdown("**Weather Situation RMSE**")
        st.dataframe(weather_rmse, use_container_width=True)

    with c3:
        st.markdown("**Working Day RMSE**")
        st.dataframe(working_rmse, use_container_width=True)

else:

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temperature Bin RMSE**")
        st.dataframe(temp_rmse, use_container_width=True)

    with c2:
        st.markdown("**Humidity Bin RMSE**")
        st.dataframe(hum_rmse, use_container_width=True)




##############################################
# CMBS CHECK — Collective Model Blind Spot
##############################################

st.markdown(
    """
    <h5 style='color:#0b2e73;'>
        🧠 CMBS — Collective Model Blind Spot Check
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="font-size:13px; padding:12px;
            background:#fff;
            border:1px solid #e0e6ff;
            border-left:4px solid #d9534f;">
<b>CMBS Interpretation</b><br>
A subgroup is flagged as a Collective Model Blind Spot when multiple
independent models simultaneously exhibit elevated error rates,
indicating a structural data or representation issue rather than
a model-specific weakness.
</div>
""", unsafe_allow_html=True)


# ==========================================
# MULTI-MODEL PREDICTIONS
# ==========================================
blind_df["lr"] = LinearRegression().fit(X_train, y_train).predict(X_test)
blind_df["tree"] = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train).predict(X_test)
blind_df["rf"] = RandomForestRegressor(
    n_estimators=200, random_state=42
).fit(X_train, y_train).predict(X_test)


# ==========================================
# CMBS FUNCTION
# ==========================================
def cmbs_check(df, group_col, preds=["lr", "tree", "rf"], threshold=0.25):

    results = {}

    # Base RMSE reference (overall RF performance)
    base = np.sqrt(mean_squared_error(df["actual"], df["rf"]))

    # Loop through subgroup values
    for g in df[group_col].dropna().unique():

        sub = df[df[group_col] == g]

        # Skip tiny subgroups
        if len(sub) < 5:
            continue

        res = {}

        # RMSE for each model
        for p in preds:
            res[p] = round(
                np.sqrt(mean_squared_error(sub["actual"], sub[p])), 2
            )

        # Collective Blind Spot Condition
        res["Collective_BlindSpot"] = all(
            np.sqrt(mean_squared_error(sub["actual"], sub[p])) > base * (1 + threshold)
            for p in preds
        )

        results[g] = res

    return pd.DataFrame(results).T


# ==========================================
# CREATE CMBS TABLES BASED ON DATASET
# ==========================================

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:

    season_cmbs = cmbs_check(blind_df, "season")\
        .reset_index().rename(columns={"index": "season"})

    weather_cmbs = cmbs_check(blind_df, "weathersit")\
        .reset_index().rename(columns={"index": "weathersit"})

    working_cmbs = cmbs_check(blind_df, "workingday")\
        .reset_index().rename(columns={"index": "workingday"})


else:
    # AQI grouping bins already exist from Blind Spot section:
    # TEMP_BIN and HUM_BIN

    temp_cmbs = cmbs_check(blind_df, "TEMP_BIN")\
        .reset_index().rename(columns={"index": "TEMP_BIN"})

    hum_cmbs = cmbs_check(blind_df, "HUM_BIN")\
        .reset_index().rename(columns={"index": "HUM_BIN"})


# ==========================================
# DISPLAY TABLES (SAFE)
# ==========================================

st.markdown("### ✅ CMBS Subgroup Results")

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Season CMBS**")
        st.dataframe(season_cmbs, use_container_width=True)

    with c2:
        st.markdown("**Weather CMBS**")
        st.dataframe(weather_cmbs, use_container_width=True)

    with c3:
        st.markdown("**Working Day CMBS**")
        st.dataframe(working_cmbs, use_container_width=True)

else:

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temperature Bin CMBS**")
        st.dataframe(temp_cmbs, use_container_width=True)

    with c2:
        st.markdown("**Humidity Bin CMBS**")
        st.dataframe(hum_cmbs, use_container_width=True)

st.markdown("""
<div style="
    background:linear-gradient(135deg, #e8f0ff, #ffffff);
    padding:12px 18px;
    border-radius:10px;
    border:1px solid #d6e1ff;
    text-align:center;
    font-size:15px;
    color:#0b2e73;
    font-weight:700;">
✨ Analysis Completed Successfully — Results Ready!
</div>
""", unsafe_allow_html=True)
