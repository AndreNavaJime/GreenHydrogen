import os
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import streamlit as st

# --- 0. Streamlit Config + Dark Theme ---
st.set_page_config(layout="centered", page_title="Hydrogen LCOH Dashboard", page_icon="⛽")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #F5F5F5; }
    .sidebar .sidebar-content { background-color: #161A23; color: #F0F0F0; }
    h1, h2, h3 { color: #0CF; }
    </style>
""", unsafe_allow_html=True)

# --- 1. Cached Data Loader ---
@st.cache_data
def load_data():
    base_path = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base_path): base_path = "."
    paths = {
        "ALKALINE": os.path.join(base_path, "USAALKALINE.csv"),
        "PEM":      os.path.join(base_path, "USAPEM.csv"),
        "SOEC":     os.path.join(base_path, "USASOEC.csv")
    }
    dfs = []
    for tech, path in paths.items():
        df0 = pd.read_csv(path)
        df0 = df0.rename(columns={c: "LCOH_$kg" for c in df0.columns if c.startswith("LCOH_")})
        df0["Tech"] = tech
        dfs.append(df0)
    df = pd.concat(dfs, ignore_index=True)

    gdf = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    gdf = gdf[["name", "geometry"]].rename(columns={"name": "State"})
    return df, gdf

df, gdf = load_data()

# --- 2. Load or Train Model (cached locally) ---
@st.cache_resource
def get_best_model(df):
    X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
    y = df["LCOH_$kg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    model_file = "best_model.pkl"
    results_file = "model_results.pkl"

    if os.path.exists(model_file) and os.path.exists(results_file):
        best_model = joblib.load(model_file)
        results = joblib.load(results_file)
    else:
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
                "preds": pr,
                "X_train": X_train
            }
        best_name = max(results, key=lambda k: results[k]["r2"])
        best_model = results[best_name]["model"]
        joblib.dump(best_model, model_file)
        joblib.dump(results, results_file)

    best_name = max(results, key=lambda k: results[k]["r2"])
    return best_model, results, best_name

best_model, results, best_name = get_best_model(df)
X_train = results[best_name]["X_train"]
y_test = df["LCOH_$kg"].iloc[X_train.shape[0]:]  # approx

# --- 3. Sidebar Controls ---
st.sidebar.header("🔧 Scenario Simulator")
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
tech = st.sidebar.selectbox("Technology", sorted(df["Tech"].unique()))
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
cap = st.sidebar.slider("CAPEX ($/kW)", float(df["CAPEX_$/kW"].min()), float(df["CAPEX_$/kW"].max()), float(df["CAPEX_$/kW"].mean()))
eff = st.sidebar.slider("Efficiency (kWh/kg)", float(df["Efficiency_kWh_kg"].min()), float(df["Efficiency_kWh_kg"].max()), float(df["Efficiency_kWh_kg"].mean()))
elec = st.sidebar.slider("Electricity price ($/kWh)", float(df["Electricity_$/kWh"].min()), float(df["Electricity_$/kWh"].max()), float(df["Electricity_$/kWh"].median()))

row = {
    "Year": year,
    "CF": float(df[(df.State == state) & (df.Year == year) & (df.Tech == tech)]["CF"].iloc[0]),
    "CAPEX_$/kW": cap,
    "Efficiency_kWh_kg": eff,
    "Electricity_$/kWh": elec,
    "Water_$kg": df["Water_$kg"].median(),
    "CO2_$kg": df["CO2_$kg"].median(),
    "Transport_$kg": df["Transport_$kg"].median(),
    "Storage_$kg": df["Storage_$kg"].median(),
    "Tech_PEM": int(tech == "PEM"),
    "Tech_SOEC": int(tech == "SOEC")
}
input_df = pd.DataFrame([row])
pred = best_model.predict(input_df)[0]

# --- 4. Tabs Layout ---
tabs = st.tabs(["Prediction 🚀", "Model Stats", "Visuals", "Map", "Download"])

with tabs[0]:
    st.markdown(f"### 🧪 Predicted LCOH → **${pred:.2f}/kg** using **{best_name}**")

with tabs[1]:
    df_stats = pd.DataFrame({
        name: {"R²": results[name]["r2"], "RMSE": results[name]["rmse"], "MAE": results[name]["mae"]}
        for name in results
    }).T.round(3)
    st.dataframe(df_stats.style.background_gradient(cmap="viridis"))

with tabs[2]:
    fi = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    st.plotly_chart(px.bar(x=fi.values, y=fi.index, orientation="h", title="Feature Importance"), use_container_width=True)
    st.plotly_chart(px.line(df, x="Year", y="LCOH_$kg", color="Tech", markers=True, title="LCOH Over Time"), use_container_width=True)

with tabs[3]:
    avg_by_state = df[df["Year"] == year].groupby("State")["LCOH_$kg"].mean().reset_index()
    merged = gdf.merge(avg_by_state, on="State", how="left")
    fig_map = px.choropleth(merged,
                            geojson=merged.geometry,
                            locations=merged.index,
                            color="LCOH_$kg",
                            hover_name="State",
                            color_continuous_scale="Viridis",
                            labels={"LCOH_$kg": "LCOH ($/kg)"},
                            title=f"LCOH by State ({year})")
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

with tabs[4]:
    st.markdown("📥 Download predictions & feature importances")
    out = pd.DataFrame({"True": y_test.values, "Predicted": results[best_name]["preds"], "Error": y_test.values - results[best_name]["preds"]})
    st.download_button("Download LCOH Predictions", out.to_csv(index=False), "lcoh_preds.csv")
    st.download_button("Download Feature Importances", fi.to_csv(), "feature_importance.csv")
