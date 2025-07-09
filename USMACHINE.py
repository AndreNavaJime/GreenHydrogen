import os
import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import streamlit as st
import time

# --- Streamlit Setup ---
st.set_page_config(layout="centered", page_title="Hydrogen LCOH Dashboard", page_icon="⛽")
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #F5F5F5; }
.sidebar .sidebar-content { background-color: #161A23; color: #F0F0F0; }
h1,h2,h3 { color: #0CF; }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Shapefile (Lower 48) ---
@st.cache_data
def load_data():
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base):
        base = "."

    frames = []
    for tech, fname in [("ALKALINE", "USAALKALINE.csv"),
                        ("PEM", "USAPEM.csv"),
                        ("SOEC", "USASOEC.csv")]:
        df0 = pd.read_csv(os.path.join(base, fname))
        df0 = df0.rename({c: "LCOH_$kg" for c in df0.columns if c.startswith("LCOH_")}, axis=1)
        df0["Tech"] = tech
        frames.append(df0)
    df = pd.concat(frames, ignore_index=True)

    lower48 = {
        "Alabama","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia",
        "Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
        "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey",
        "New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
        "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
        "West Virginia","Wisconsin","Wyoming"
    }
    df = df[df["State"].isin(lower48)]

    geo_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    gdf = gpd.read_file(geo_url).rename(columns={"name": "State"})
    gdf = gdf[gdf["State"].isin(lower48)]
    return df, gdf

df, gdf = load_data()

# --- Train or Load Models ---
@st.cache_resource
def train_models(df):
    start = time.time()
    X = pd.get_dummies(df.drop(columns=["State", "LCOH_$kg"]), columns=["Tech"], drop_first=True)
    y = df["LCOH_$kg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        y_list = y_test.reset_index(drop=True).tolist()
        p_list = [float(v) for v in preds]
        results[name] = {
            "model": model,
            "r2": r2_score(y_list, p_list),
            "rmse": np.sqrt(mean_squared_error(y_list, p_list)),
            "mae": mean_absolute_error(y_list, p_list),
            "preds": p_list,
            "y_true": y_list
        }

    joblib.dump(results, "results.pkl")
    st.sidebar.write(f"⏱️ Full training time: {time.time() - start:.1f}s")
    return results, X_train

results, X_train = train_models(df)

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Scenario Simulator")
models_list = list(results.keys())
model_name = st.sidebar.selectbox("Choose Model", models_list, index=models_list.index("XGBoost"))
best_model = results[model_name]["model"]

year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
tech = st.sidebar.selectbox("Technology", sorted(df["Tech"].unique()))
state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
cap = st.sidebar.slider("CAPEX ($/kW)", float(df["CAPEX_$/kW"].min()),
                        float(df["CAPEX_$/kW"].max()), float(df["CAPEX_$/kW"].mean()))
eff = st.sidebar.slider("Efficiency (kWh/kg)", float(df["Efficiency_kWh_kg"].min()),
                        float(df["Efficiency_kWh_kg"].max()), float(df["Efficiency_kWh_kg"].mean()))
elec = st.sidebar.slider("Electricity price ($/kWh)", float(df["Electricity_$/kWh"].min()),
                          float(df["Electricity_$/kWh"].max()), float(df["Electricity_$/kWh"].median()))

row = {
    "Year": year,
    "CF": float(df.query("State==@state & Year==@year & Tech==@tech")["CF"].iloc[0]),
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

# --- Output Tabs ---
tabs = st.tabs(["Prediction", "Stats", "Visuals", "Map", "Download"])

with tabs[0]:
    st.markdown(f"### 🧪 Predicted LCOH → **${pred:.2f}/kg** using **{model_name}**")

with tabs[1]:
    stats_df = pd.DataFrame({
        name: {"R²": results[name]["r2"],
               "RMSE": results[name]["rmse"],
               "MAE": results[name]["mae"]}
        for name in results
    }).T.round(3)
    st.dataframe(stats_df.style.background_gradient(cmap="viridis"))

with tabs[2]:
    fi = getattr(best_model, "feature_importances_", None)
    if fi is not None:
        fi_series = pd.Series(fi, index=X_train.columns).sort_values()
        fig_bar = px.bar(x=fi_series.values, y=fi_series.index, orientation="h",
                         color=fi_series.values, color_continuous_scale="Viridis",
                         title="🔍 Feature Importance", template="plotly_dark")
        fig_bar.update_layout(height=450, xaxis_title="Importance", yaxis_title="Feature")
        fig_bar.update_coloraxes(colorbar_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)

    pivot = df.groupby(["Tech", "Year"])["LCOH_$kg"].mean().reset_index()
    pivot = pivot.pivot(index="Tech", columns="Year", values="LCOH_$kg").sort_index(axis=1)
    fig_heat = px.imshow(pivot, color_continuous_scale="Viridis",
                         labels={"y": "Tech", "x": "Year", "color": "LCOH ($/kg)"},
                         title="LCOH Trend Heatmap", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

with tabs[3]:
    avg = df[df["Year"] == year].groupby("State")["LCOH_$kg"].mean().reset_index()
    merged = gdf.merge(avg, on="State", how="left")
    fig_map = px.choropleth(merged, geojson=merged.geometry, locations=merged.index,
                            color="LCOH_$kg", hover_name="State",
                            color_continuous_scale="Viridis",
                            title=f"LCOH by State ({year})", template="plotly_dark")
    fig_map.update_coloraxes(colorbar_tickformat=".0f")
    fig_map.update_geos(scope="usa", coastlinecolor="white",
                        showland=True, landcolor="#1e1e1e",
                        showlakes=True, lakecolor="#1e1e1e")
    st.plotly_chart(fig_map, use_container_width=True)

with tabs[4]:
    y_true = results[model_name]["y_true"]
    preds = results[model_name]["preds"]
    df_out = pd.DataFrame({"True": y_true, "Predicted": preds})
    df_out["Error"] = df_out["True"] - df_out["Predicted"]
    st.download_button("📥 Download CSV", df_out.to_csv(index=False), "predictions.csv")
