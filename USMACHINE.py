import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import seaborn as sns
import streamlit as st

# --- 0. Streamlit Page + Theme ---
st.set_page_config(layout="centered", page_title="🚀 Hydrogen LCOH VR Dashboard", page_icon="⛽")
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #F5F5F5; }
    .sidebar .sidebar-content { background-color: #161A23; color: #F0F0F0; }
    h1, h2, h3 { color: #0CF; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. Load Data ---
paths = {
    "ALKALINE": r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION\USAALKALINE.csv",
    "PEM":      r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION\USAPEM.csv",
    "SOEC":     r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION\USASOEC.csv"
}
dfs = []
for tech, path in paths.items():
    df = pd.read_csv(path)
    df = df.rename(columns={c: "LCOH_$kg" for c in df.columns if c.startswith("LCOH_")})
    df["Tech"] = tech
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Features & Targets
X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
y = df["LCOH_$kg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# --- 2. Train Models ---
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pr = m.predict(X_test)
    results[name] = {
        "model": m,
        "r2": r2_score(y_test, pr),
        "rmse": np.sqrt(mean_squared_error(y_test, pr)),
        "mae": mean_absolute_error(y_test, pr),
        "preds": pr
    }
best_name = max(results, key=lambda k: results[k]["r2"])
best = results[best_name]["model"]

# --- 3. Sidebar Inputs ---
st.sidebar.header("🔧 Scenario Simulator")
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
tech = st.sidebar.selectbox("Technology", df["Tech"].unique())
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
cap = st.sidebar.slider("CAPEX ($/kW)", float(df["CAPEX_$/kW"].min()), float(df["CAPEX_$/kW"].max()), float(df["CAPEX_$/kW"].mean()))
eff = st.sidebar.slider("Efficiency (kWh/kg)", float(df["Efficiency_kWh_kg"].min()), float(df["Efficiency_kWh_kg"].max()), float(df["Efficiency_kWh_kg"].mean()))

# Build input row
row = {
    "Year": year,
    "CF": int(df[(df.State==state)&(df.Year==year)&(df.Tech==tech)]["CF"].iloc[0]),
    "CAPEX_$/kW": cap,
    "Efficiency_kWh_kg": eff,
    "Electricity_$/kWh": df[df.State==state]["Electricity_$/kWh"].median(),
    "Water_$kg": df["Water_$kg"].median(),
    "CO2_$kg": df["CO2_$kg"].median(),
    "Transport_$kg": df["Transport_$kg"].median(),
    "Storage_$kg": df["Storage_$kg"].median(),
    "Tech_PEM": int(tech=="PEM"),
    "Tech_SOEC": int(tech=="SOEC")
}
input_df = pd.DataFrame([row])
pred = best.predict(input_df)[0]

# --- 4. Main Panel with Tabs ---
tabs = st.tabs(["Prediction 🚀", "Model Stats", "Visuals", "Download"])
with tabs[0]:
    st.markdown(f"### 🧪 Predicted LCOH → **${pred:.2f}/kg** using **{best_name}**")
with tabs[1]:
    df_stats = pd.DataFrame({
        name: {"R²": results[name]["r2"], "RMSE": results[name]["rmse"], "MAE": results[name]["mae"]}
        for name in results
    }).T.round(3)
    st.dataframe(df_stats.style.background_gradient(cmap="viridis"))
with tabs[2]:
    # Feature Importance - Plotly
    fi = pd.Series(best.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    fig1 = px.bar(x=fi.values, y=fi.index, orientation="h", title="Feature Importance")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df, x="Year", y="LCOH_$kg", color="Tech", markers=True, title="LCOH Over Time")
    st.plotly_chart(fig2, use_container_width=True)

    avg = df.groupby("State")["LCOH_$kg"].mean().reset_index()
    fig3 = px.bar(avg, x="LCOH_$kg", y="State", orientation="h", title="Avg LCOH by State")
    st.plotly_chart(fig3, use_container_width=True)

with tabs[3]:
    st.markdown("Download Predictions & Feature Importance")
    output = pd.DataFrame({
        "True": y_test.values,
        "Predicted": results[best_name]["preds"],
        "Error": y_test.values - results[best_name]["preds"]
    })
    with st.expander("Download CSV"):
        st.download_button("Download LCOH Data", output.to_csv(index=False), file_name="lcoh_preds.csv")
    st.download_button("Download Feature Importance", fi.to_csv(), file_name="fi.csv")
