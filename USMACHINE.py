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

# --- Streamlit Config ---
st.set_page_config(layout="centered", page_title="Hydrogen LCOH Dashboard", page_icon="⛽")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #F5F5F5; }
    .sidebar .sidebar-content { background-color: #161A23; color: #F0F0F0; }
    h1, h2, h3 { color: #0CF; }
    </style>
""", unsafe_allow_html=True)

# --- Cached Data Loader ---
@st.cache_data
def load_data():
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base): base = "."
    dfs = []
    for tech, fname in [("ALKALINE", "USAALKALINE.csv"),
                        ("PEM", "USAPEM.csv"),
                        ("SOEC", "USASOEC.csv")]:
        df0 = pd.read_csv(os.path.join(base, fname))
        df0 = df0.rename(columns={c: "LCOH_$kg" for c in df0.columns if c.startswith("LCOH_")})
        df0["Tech"] = tech
        dfs.append(df0)
    df = pd.concat(dfs, ignore_index=True)
    gdf = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    gdf = gdf[["name", "geometry"]].rename(columns={"name": "State"})
    return df, gdf

df, gdf = load_data()

# --- Cached Model Loader/Trainer ---
@st.cache_resource
def load_or_train_model(df):
    X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
    y = df["LCOH_$kg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    if os.path.exists("best_model.pkl") and os.path.exists("results.pkl"):
        best_model = joblib.load("best_model.pkl")
        results = joblib.load("results.pkl")
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
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(results, "results.pkl")

    best_name = max(results, key=lambda k: results[k]["r2"])
    return results[best_name]["model"], results, best_name, results[best_name]["X_train"]

best_model, results, best_name, X_train = load_or_train_model(df)
y_test = df["LCOH_$kg"].iloc[X_train.shape[0]:]  # approximation

# --- Sidebar Controls ---
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

# --- Tabs Layout ---
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

    # Feature Importance
    fig_bar = px.bar(
        x=fi.values, y=fi.index, orientation="h", color=fi.values,
        color_continuous_scale="Viridis", title="🔍 Feature Importance",
        template="plotly_dark"
    )
    fig_bar.update_traces(marker_line_width=0)
    fig_bar.update_layout(height=500, margin=dict(l=70, r=30, t=50, b=30), font=dict(size=12))
    st.plotly_chart(fig_bar, use_container_width=True)

    # LCOH Over Time
    fig_line = px.line(
        df, x="Year", y="LCOH_$kg", color="Tech",
        line_group="State", markers=True,
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_line.update_traces(line=dict(width=1.2), marker=dict(size=4))
    st.plotly_chart(fig_line, use_container_width=True)

with tabs[3]:
    avg_by_state = df[df["Year"] == year].groupby("State")["LCOH_$kg"].mean().reset_index()
    merged = gdf.merge(avg_by_state, on="State", how="left")
    fig_map = px.choropleth(
        merged, geojson=merged.geometry, locations=merged.index,
        color="LCOH_$kg", hover_name="State", color_continuous_scale="Viridis",
        labels={"LCOH_$kg": "LCOH ($/kg)"},
        title=f"LCOH by State ({year})",
        template="plotly_dark"
    )
    fig_map.update_geos(
        showcoastlines=True, coastlinecolor="white",
        showland=True, landcolor="#1e1e1e",
        showlakes=True, lakecolor="#1e1e1e",
        oceancolor="#1e1e1e",
        fitbounds="locations", visible=False
    )
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with tabs[4]:
    st.markdown("📥 Download predictions & feature importances")
    out = pd.DataFrame({
        "True": y_test.values,
        "Predicted": results[best_name]["preds"],
        "Error": y_test.values - results[best_name]["preds"]
    })
    st.download_button("Download LCOH Predictions", out.to_csv(index=False), "lcoh_preds.csv")
    st.download_button("Download Feature Importances", fi.to_csv(), "feature_importance.csv")
