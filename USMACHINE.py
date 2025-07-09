import os
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px

# --- Streamlit Config + Light Theme ---
st.set_page_config(
    layout="centered",
    page_title="Hydrogen LCOH Dashboard",
    page_icon="⚗️"
)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .sidebar .sidebar-content { background-color: #ffffff; color: #212529; }
    h1, h2, h3 { color: #0d6efd; }
    .css-1d391kg { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# --- 1. Caching Data & Models ---
@st.cache_data
def load_data():
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base):
        base = "."
    dfs = []
    for tech, fname in [("ALKALINE", "USAALKALINE.csv"),
                        ("PEM", "USAPEM.csv"),
                        ("SOEC", "USASOEC.csv")]:
        df = pd.read_csv(os.path.join(base, fname))
        df = df.rename(columns={c: "LCOH_$kg" for c in df if c.startswith("LCOH_")})
        df["Tech"] = tech
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    gdf = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    )[["name", "geometry"]].rename(columns={"name": "State"})
    return df, gdf

@st.cache_resource
def load_or_train_model(df):
    X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
    y = df["LCOH_$kg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_path = "best_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        results = joblib.load("results.pkl")
    else:
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        }
        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            results[name] = {
                "model": m,
                "r2": r2_score(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "mae": mean_absolute_error(y_test, preds),
                "preds": preds
            }
        best = max(results, key=lambda k: results[k]["r2"])
        joblib.dump(results[best]["model"], model_path)
        joblib.dump(results, "results.pkl")
        model = results[best]["model"]
    return model, results

df, gdf = load_data()
best_model, results = load_or_train_model(df)
best_name = max(results, key=lambda name: results[name]["r2"])

# --- 3. Sidebar for Inputs ---
st.sidebar.header("Scenario Simulator 🧮")
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
tech = st.sidebar.selectbox("Technology", sorted(df["Tech"].unique()))
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
cap = st.sidebar.slider("CAPEX $/kW", float(df["CAPEX_$/kW"].min()), float(df["CAPEX_$/kW"].max()), float(df["CAPEX_$/kW"].mean()))
eff = st.sidebar.slider("Efficiency (kWh/kg)", float(df["Efficiency_kWh_kg"].min()), float(df["Efficiency_kWh_kg"].max()), float(df["Efficiency_kWh_kg"].mean()))
elec = st.sidebar.slider("Electricity $/kWh", float(df["Electricity_$/kWh"].min()), float(df["Electricity_$/kWh"].max()), float(df["Electricity_$/kWh"].median()))

# Build prediction input
cf = float(df[(df.State == state) & (df.Year == year) & (df.Tech == tech)]["CF"].iloc[0])
row = {
    "Year": year, "CF": cf, "CAPEX_$/kW": cap,
    "Efficiency_kWh_kg": eff, "Electricity_$/kWh": elec,
    "Water_$kg": df["Water_$kg"].median(),
    "CO2_$kg": df["CO2_$kg"].median(),
    "Transport_$kg": df["Transport_$kg"].median(),
    "Storage_$kg": df["Storage_$kg"].median(),
    "Tech_PEM": int(tech == "PEM"), "Tech_SOEC": int(tech == "SOEC")
}
input_df = pd.DataFrame([row])
pred = best_model.predict(input_df)[0]

# --- 4. Tabbed Interface ---
tabs = st.tabs(["Prediction", "Model Stats", "Visuals", "Map", "Download"])

with tabs[0]:
    st.header("Predicted LCOH")
    st.metric(label="Cost ($/kg)", value=f"${pred:.2f}", delta=f"using {best_name}")

with tabs[1]:
    stats = {name: {
        "R²": results[name]["r2"],
        "RMSE": results[name]["rmse"],
        "MAE": results[name]["mae"]
    } for name in results}
    st.table(pd.DataFrame(stats).T.style.format("{:.3f}"))

with tabs[2]:
    fi = pd.Series(best_model.feature_importances_, index=input_df.columns).sort_values(ascending=True)
    st.subheader("Feature Importance")
    st.plotly_chart(px.bar(x=fi.values, y=fi.index, orientation="h", template="ggplot2"), use_container_width=True)
    st.subheader("LCOH Over Time by Tech")
    st.plotly_chart(px.line(df, x="Year", y="LCOH_$kg", color="Tech", markers=True, template="ggplot2"), use_container_width=True)

with tabs[3]:
    st.subheader(f"LCOH by State ({year})")
    agg = df[df["Year"] == year].groupby("State")["LCOH_$kg"].mean().reset_index()
    merged = gdf.merge(agg, on="State", how="left")
    fig = px.choropleth(merged, geojson=merged.geometry,
                        locations=merged.index, color="LCOH_$kg",
                        hover_name="State", color_continuous_scale="Blues",
                        labels={"LCOH_$kg": "Cost $/kg"}, title="")
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Download Data")
    out = pd.DataFrame({"True": results[best_name]["preds"], "Pred": results[best_name]["preds"], "Error": results[best_name]["preds"]})
    st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")
