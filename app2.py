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
from sklearn.neural_network import MLPRegressor



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
st.image("banner1.png", use_container_width=True)


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
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)", "Neural Network (MLP)"]
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

    elif name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=8)

    elif name == "Random Forest (Ensemble)":
        model = RandomForestRegressor(n_estimators=220, random_state=42)

    else:  # Neural Network
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=800,
            random_state=42
        )

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
# FIGURE 5.1 — GLOBAL FEATURE INFLUENCE ACROSS MODELS
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🌍 Figure 5.1: Global Feature Influence Across Models
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f7f9fc;
                border-left:5px solid #2e7fe8;
                border-radius:8px;">
    <b>Figure 5.1: Global Feature Influence Across Models in High-Agreement Regions</b><br>
    The figure shows that different model families rely on the same dominant features,
    explaining convergence in predictions despite architectural differences.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Train all 3 models for comparison
# -----------------------------
lr_model = LinearRegression().fit(X_train, y_train)
tree_model = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# -----------------------------
# Extract Feature Influence
# -----------------------------
features = X.columns

lr_imp = np.abs(lr_model.coef_)
tree_imp = tree_model.feature_importances_
rf_imp = rf_model.feature_importances_

# Create Combined DataFrame
imp_df = pd.DataFrame({
    "Feature": features,
    "Linear Regression": lr_imp,
    "Decision Tree": tree_imp,
    "Random Forest": rf_imp
})

# Normalize for fair comparison
imp_df.iloc[:, 1:] = imp_df.iloc[:, 1:].apply(lambda x: x / x.max())

# Select Top 8 most influential overall
imp_df["Overall"] = imp_df.iloc[:, 1:].mean(axis=1)
top_imp = imp_df.sort_values("Overall", ascending=False).head(8)

# -----------------------------
# Plot Side-by-Side Comparison
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4))

top_imp.set_index("Feature")[["Linear Regression",
                             "Decision Tree",
                             "Random Forest"]].plot(kind="barh", ax=ax)

ax.set_title("Top Global Drivers Across Model Families", fontsize=11)
ax.set_xlabel("Normalized Influence Score", fontsize=9)
ax.set_ylabel("")

plt.tight_layout()
st.pyplot(fig)

##############################################
# FIGURE 5.2 — LOCAL EXPLAINABILITY OF A FAILURE CASE
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🎯 Figure 5.2: Local Explainability of a Collective Failure Instance
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#fff7f0;
                border-left:5px solid #ff9f40;
                border-radius:8px;">
    <b>Figure 5.2: Local Explainability of a Collective Failure Instance</b><br>
    Despite differing architectures, models attribute the incorrect prediction
    to the same feature contributions, indicating shared reasoning behind failure.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Step 1: Train all models again (for explanation)
# ----------------------------------------------------
lr_model = LinearRegression().fit(X_train, y_train)
tree_model = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# Predictions
lr_preds = lr_model.predict(X_test)
tree_preds = tree_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# ----------------------------------------------------
# Step 2: Find worst collective failure instance
# ----------------------------------------------------
error_df = pd.DataFrame({
    "Actual": y_test.values,
    "LR_Error": np.abs(y_test.values - lr_preds),
    "Tree_Error": np.abs(y_test.values - tree_preds),
    "RF_Error": np.abs(y_test.values - rf_preds)
})

# Combined error score
error_df["Total_Error"] = error_df[["LR_Error","Tree_Error","RF_Error"]].mean(axis=1)

# Index of worst failure
worst_index = error_df["Total_Error"].idxmax()

st.markdown(
    f"""
    ✅ Selected instance: <b>Worst collective failure point</b><br>
    • Actual = {round(error_df.loc[worst_index,"Actual"],2)}<br>
    • RF Prediction = {round(rf_preds[worst_index],2)}<br>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Step 3: Extract that specific data row
# ----------------------------------------------------
x_instance = X_test.iloc[worst_index:worst_index+1]

# ----------------------------------------------------
# Step 4: Local Feature Attribution
# (Approximate local influence)
# ----------------------------------------------------

# Linear Regression Contribution
lr_contrib = lr_model.coef_ * x_instance.values[0]

# Random Forest + Tree use impurity-based proxy
tree_imp = tree_model.feature_importances_
rf_imp = rf_model.feature_importances_

# Build comparison table
local_df = pd.DataFrame({
    "Feature": X.columns,
    "LR_Local_Impact": np.abs(lr_contrib),
    "Tree_Local_Proxy": tree_imp,
    "RF_Local_Proxy": rf_imp
})

# Normalize for comparison
local_df.iloc[:,1:] = local_df.iloc[:,1:].apply(lambda x: x/x.max())

# Top 6 contributing features
top_local = local_df.sort_values("LR_Local_Impact", ascending=False).head(6)

# ----------------------------------------------------
# Step 5: Plot — Local Failure Explanation
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7,4))

top_local.set_index("Feature")[[
    "LR_Local_Impact",
    "Tree_Local_Proxy",
    "RF_Local_Proxy"
]].plot(kind="barh", ax=ax)

ax.set_title("Shared Feature Attribution in Collective Failure Case", fontsize=11)
ax.set_xlabel("Normalized Local Contribution", fontsize=9)
ax.set_ylabel("")

plt.tight_layout()
st.pyplot(fig)

##############################################
# FIGURE 4.1 — AGGREGATE BASELINE MODEL COMPARISON
##############################################

st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Figure 4.1: Aggregate Baseline Performance Comparison Across Models
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f7f9fc;
                border-left:5px solid #2e7fe8;
                border-radius:8px;">
    In addition to performance, baseline evaluation revealed substantial agreement
    across models. Despite architectural differences, models frequently produced
    similar predictions for identical inputs. At this stage, agreement is treated
    as an observation rather than validation.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# Train All Models for Baseline Comparison
# ----------------------------------------------------
from sklearn.neural_network import MLPRegressor

baseline_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(64,32),
                                   max_iter=800,
                                   random_state=42)
}

results = []

for name, mdl in baseline_models.items():
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)

    rmse_val = np.sqrt(mean_squared_error(y_test, pred))
    r2_val = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "RMSE": rmse_val,
        "R2 Score": r2_val
    })

# Create results dataframe
perf_df = pd.DataFrame(results)

# ----------------------------------------------------
# Plot Performance Comparison
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7,4))

# RMSE bar plot
sns.barplot(
    data=perf_df,
    x="Model",
    y="RMSE",
    ax=ax
)

ax.set_title("Baseline RMSE Comparison Across Models", fontsize=11)
ax.set_ylabel("RMSE (Lower is Better)")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=20)

st.pyplot(fig)

# ----------------------------------------------------
# Display R² as Table (Compact)
# ----------------------------------------------------
st.markdown("### Baseline R² Scores (Higher is Better)")

st.dataframe(
    perf_df[["Model", "R2 Score"]].sort_values("R2 Score", ascending=False),
    use_container_width=True,
    height=180
)

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
blind_df["nn"] = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=800,
    random_state=42
).fit(X_train, y_train).predict(X_test)


# ==========================================
# CMBS FUNCTION
# ==========================================
def cmbs_check(df, group_col, preds=["lr", "tree", "rf","nn"], threshold=0.25):

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




##############################################
# FIGURE 6.1 — CMBS FRAMEWORK OVERVIEW
##############################################

st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Figure 6.1: CMBS Framework Overview
    </h4>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(8,5))
ax.axis("off")

# --- Title Box ---
ax.text(0.5, 0.92, "Collective Model Blind Spot (CMBS) Framework",
        ha="center", fontsize=14, fontweight="bold")

# --- Model Boxes ---
models = ["Linear Model", "Tree Model", "Ensemble Model", "Neural Network"]
x_pos = [0.15, 0.38, 0.62, 0.85]

for i, m in enumerate(models):
    ax.text(x_pos[i], 0.78, m,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray"))

# --- Subgroup Box ---
ax.text(0.5, 0.62, "Test Data Subgroup",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#d6eaff"))

# Arrows down from models
for x in x_pos:
    ax.annotate("", xy=(x, 0.67), xytext=(x, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# --- Prediction Outputs ---
ax.text(0.2, 0.42, "LR Predictions\nRMSE ↑",
        ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))

ax.text(0.5, 0.42, "Tree Predictions\nRMSE ↑",
        ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))

ax.text(0.8, 0.42, "RF / NN Predictions\nRMSE ↑",
        ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))

# --- Compare RMSE Box ---
ax.text(0.5, 0.25, "Compare Subgroup RMSE Across Models",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2cc"))

# Arrow
ax.annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.38),
            arrowprops=dict(arrowstyle="->", lw=2))

# --- Final Output Box ---
ax.text(0.5, 0.1, "⚠ Identify Collective Blind Spot",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffcccc"))

st.pyplot(fig)

st.caption(
    "Figure 6.1: CMBS Framework Overview — Multiple independent models are compared across subgroups to detect shared failure regions."
)
##############################################
# FIGURE 6.2 — CONCEPTUAL COLLECTIVE BLIND SPOT
##############################################

st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Figure 6.2: Conceptual Collective Blind Spot
    </h4>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(6,6))
ax.axis("off")

# --- Draw concentric zones ---
safe = plt.Circle((0.5, 0.5), 0.42, color="#b6f2b6", ec="black", lw=1.5)
agree = plt.Circle((0.5, 0.5), 0.28, color="#ffe39f", ec="black", lw=1.5)
blind = plt.Circle((0.5, 0.5), 0.14, color="#ff7f7f", ec="black", lw=1.5)

ax.add_patch(safe)
ax.add_patch(agree)
ax.add_patch(blind)

# --- Labels ---
ax.text(0.5, 0.75, "Safe Zone\nAccurate Predictions",
        ha="center", fontsize=11, fontweight="bold")

ax.text(0.5, 0.58, "Agreement Region\nLow Error Across Models",
        ha="center", fontsize=10)

ax.text(0.5, 0.48, "Blind Spot Zone\nHigh Error for ALL Models",
        ha="center", fontsize=10, color="white", fontweight="bold")

# --- Model arrows into blind spot ---
ax.annotate("Model A", xy=(0.55,0.52), xytext=(0.85,0.60),
            arrowprops=dict(arrowstyle="->", lw=2))

ax.annotate("Model B", xy=(0.55,0.50), xytext=(0.85,0.50),
            arrowprops=dict(arrowstyle="->", lw=2))

ax.annotate("Model C", xy=(0.55,0.48), xytext=(0.85,0.40),
            arrowprops=dict(arrowstyle="->", lw=2))

# Warning symbol
ax.text(0.5, 0.36, "⚠", ha="center", fontsize=22)

st.pyplot(fig)

st.caption(
    "Figure 6.2: Conceptual Collective Blind Spot — Even when models agree, a central zone may exist where all fail together due to shared reasoning limitations."
)

