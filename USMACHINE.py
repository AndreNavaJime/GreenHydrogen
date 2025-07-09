import os, time
import pandas as pd, numpy as np, geopandas as gpd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px

# --- Streamlit setup ---
st.set_page_config(layout="centered", page_title="Hydrogen LCOH Dashboard", page_icon="⛽")
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #F5F5F5; }
.sidebar .sidebar-content { background-color: #161A23; color: #F0F0F0; }
h1,h2,h3 { color: #0CF; }
</style>
""", unsafe_allow_html=True)

# --- Data loader with lower‑48 filter ---
@st.cache_data
def load_data():
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base): base = "."
    dfs = []
    for tech, fname in [("ALKALINE","USAALKALINE.csv"),("PEM","USAPEM.csv"),("SOEC","USASOEC.csv")]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            df0 = pd.read_csv(path)
            df0 = df0.rename(columns={c: "LCOH_$kg" for c in df0.columns if c.startswith("LCOH_")})
            df0["Tech"] = tech
            dfs.append(df0)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    lower48 = {
        "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
        "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
        "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
        "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
        "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
        "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
        "West Virginia", "Wisconsin", "Wyoming"
    }
    df = df[df.State.isin(lower48)]
    geo_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    gdf = gpd.read_file(geo_url).rename(columns={"name":"State"})
    gdf = gdf[gdf.State.isin(lower48)]
    return df, gdf

df, gdf = load_data()
st.sidebar.markdown(f"<b>Rows loaded:</b> {len(df)}", unsafe_allow_html=True)
if df.empty:
    st.error("No data loaded — check your CSV path!")
    st.stop()

# --- Model trainer ---
@st.cache_resource
def train_models(df):
    if len(df) < 10:
        return {}
    X = pd.get_dummies(df.drop(columns=["State","LCOH_$kg"]), columns=["Tech"], drop_first=True)
    y = df["LCOH_$kg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "GBRegressor": GradientBoostingRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100),
        "CatBoost": CatBoostRegressor(verbose=0)
    }
    results = {}
    t0 = time.time()
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        results[name] = {
            "model": m,
            "r2": r2_score(y_te, preds),
            "rmse": np.sqrt(mean_squared_error(y_te, preds)),
            "mae": mean_absolute_error(y_te, preds),
            "preds": preds.tolist(),
            "y_true": y_te.tolist()
        }
    st.sidebar.write(f"⏱ Training finished in {time.time() - t0:.1f}s")
    return results, X_tr

results, X_train = train_models(df)
if not results:
    st.error("Training skipped: not enough data!")
    st.stop()

# --- Sidebar controls ---
st.sidebar.header("🔧 Scenario Simulator")
model_name = st.sidebar.selectbox("Pick model", list(results.keys()), index=list(results.keys()).index("XGBoost"))
best = results[model_name]["model"]

year = st.sidebar.selectbox("Year", sorted(df.Year.unique()))
tech = st.sidebar.selectbox("Technology", sorted(df.Tech.unique()))
state = st.sidebar.selectbox("State", sorted(df.State.unique()))
cap = st.sidebar.slider("CAPEX ($/kW)", float(df["CAPEX_$/kW"].min()), float(df["CAPEX_$/kW"].max()), float(df["CAPEX_$/kW"].mean()))
eff = st.sidebar.slider("Efficiency (kWh/kg)", float(df.Efficiency_kWh_kg.min()), float(df.Efficiency_kWh_kg.max()), float(df.Efficiency_kWh_kg.mean()))
elec = st.sidebar.slider("Electricity price ($/kWh)", float(df["Electricity_$/kWh"].min()), float(df["Electricity_$/kWh"].max()), float(df["Electricity_$/kWh"].median()))

row = {
    "Year": year,
    "CF": float(df.query("State==@state & Year==@year & Tech==@tech").CF.iloc[0]),
    "CAPEX_$/kW": cap,
    "Efficiency_kWh_kg": eff,
    "Electricity_$/kWh": elec,
    "Water_$kg": df["Water_$kg"].median(),
    "CO2_$kg": df["CO2_$kg"].median(),
    "Transport_$kg": df["Transport_$kg"].median(),
    "Storage_$kg": df["Storage_$kg"].median(),
    "Tech_PEM": int(tech=="PEM"),
    "Tech_SOEC": int(tech=="SOEC")
}
pred = best.predict(pd.DataFrame([row]))[0]

# --- Layout Tabs ---
tabs = st.tabs(["Prediction", "Stats", "Visuals", "Map", "Download"])

with tabs[0]:
    st.markdown(f"### 🧪 Predicted LCOH → **${pred:.2f}/kg** using **{model_name}**")

with tabs[1]:
    stats_df = pd.DataFrame({k: {"R²": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]} for k, v in results.items()}).T.round(3)
    st.dataframe(stats_df.style.background_gradient(cmap="viridis"))

with tabs[2]:
    fi = getattr(best, "feature_importances_", None)
    if fi is not None:
        cols = X_train.columns
        fi_s = pd.Series(fi, index=cols).sort_values()
        fig = px.bar(fi_s.values, y=fi_s.index, orientation="h",
                     color=fi_s.values, color_continuous_scale="Viridis",
                     template="plotly_dark", title="🔍 Feature Importance")
        fig.update_coloraxes(colorbar_title=None)
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

    pivot = df.groupby(["Tech", "Year"])["LCOH_$kg"].mean().reset_index().pivot(index="Tech", columns="Year", values="LCOH_$kg")
    pivot = pivot.sort_index(axis=1)
    fig2 = px.imshow(pivot, color_continuous_scale="Viridis", template="plotly_dark",
                     labels={"y":"Tech","x":"Year","color":"LCOH ($/kg)"},
                     title="LCOH Trend Heatmap")
    st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    avg = df[df.Year == year].groupby("State")["LCOH_$kg"].mean().reset_index()
    merged = gdf.merge(avg, on="State", how="left")
    fig3 = px.choropleth(merged, geojson=merged.geometry, locations=merged.index,
                         color="LCOH_$kg", hover_name="State",
                         color_continuous_scale="Viridis", template="plotly_dark",
                         title=f"LCOH by State ({year})")
    fig3.update_geos(scope="usa")
    fig3.update_coloraxes(colorbar_tickformat=".0f")
    st.plotly_chart(fig3, use_container_width=True)

with tabs[4]:
    y_true = results[model_name]["y_true"]
    preds = results[model_name]["preds"]
    df_out = pd.DataFrame({"True": y_true, "Predicted": preds})
    df_out["Error"] = df_out["True"] - df_out["Predicted"]
    st.download_button("📥 Download Predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile, io

# ➕ Save prediction figure
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(y_true, label="True", linewidth=2)
ax.plot(preds, label="Predicted", linestyle="--")
ax.set_title(f"{model_name} – Prediction vs True")
ax.legend()
fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
fig.savefig(fig_path, bbox_inches="tight")

# ➕ PDF report
pdf = FPDF()
pdf.add_page()

# ❌ Don't register .ttf, just use built-in Times
pdf.set_font("Times", "B", 16)
pdf.cell(0, 10, "LCOH Model Report", ln=True)

pdf.set_font("Times", "", 12)
pdf.cell(0, 10, f"Model: {model_name}", ln=True)
pdf.cell(0, 10, f"R²: {results[model_name]['r2']:.3f}", ln=True)
pdf.cell(0, 10, f"RMSE: {results[model_name]['rmse']:.3f}", ln=True)
pdf.cell(0, 10, f"MAE: {results[model_name]['mae']:.3f}", ln=True)
pdf.ln(5)

pdf.image(fig_path, x=10, w=180)

pdf.set_font("Times", "B", 12)
pdf.cell(40, 10, "True", 1)
pdf.cell(40, 10, "Predicted", 1)
pdf.cell(40, 10, "Error", 1)
pdf.ln()

pdf.set_font("Times", "", 12)
for i in range(min(10, len(y_true))):
    pdf.cell(40, 10, f"{y_true[i]:.2f}", 1)
    pdf.cell(40, 10, f"{preds[i]:.2f}", 1)
    pdf.cell(40, 10, f"{(y_true[i] - preds[i]):.2f}", 1)
    pdf.ln()

# ➕ Download PDF
pdf_bytes = io.BytesIO()
pdf_bytes.write(pdf.output(dest="S").encode("latin1"))
pdf_bytes.seek(0)

st.download_button("📄 Download PDF Report", pdf_bytes, "model_report.pdf", mime="application/pdf")
