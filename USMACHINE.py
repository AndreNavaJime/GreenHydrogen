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
    if not os.path.exists(base):
        base = "."
        
    dfs = []
    for tech, fname in [("ALKALINE", "USAALKALINE.csv"), ("PEM", "USAPEM.csv"), ("SOEC", "USASOEC.csv")]:
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

    df = df[df["State"].isin(lower48)]

    geo_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    gdf = gpd.read_file(geo_url).rename(columns={"name": "State"})

    # Clean up and strictly filter to Lower 48 only
    gdf["State"] = gdf["State"].str.strip()
    gdf = gdf[gdf["State"].isin(lower48)].copy()

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
# --- Dynamic Slider Ranges Based on Tech ---
if tech == "ALKALINE":
    cap_min, cap_max = 150, 1000
    eff_min, eff_max = 48, 55
elif tech == "PEM":
    cap_min, cap_max = 150, 1200
    eff_min, eff_max = 46, 55
elif tech == "SOEC":
    cap_min, cap_max = 200, 2500
    eff_min, eff_max = 34, 47
else:
    cap_min, cap_max = 100, 3000
    eff_min, eff_max = 30, 60

# --- Show range for clarity ---
st.sidebar.caption(f"CAPEX range: {cap_min}–{cap_max} | Efficiency range: {eff_min}–{eff_max}")

# --- Sliders with helpful tooltips ---
cap = st.sidebar.slider("CAPEX ($/kW)", min_value=int(cap_min), max_value=int(cap_max), value=int((cap_min + cap_max) / 2), help="Capital expenditure range depends on selected technology")
eff = st.sidebar.slider("Efficiency (kWh/kg)", min_value=float(eff_min), max_value=float(eff_max), value=round((eff_min + eff_max) / 2, 1), help="Energy used per kg of H₂ produced — varies by tech")

state_rate_cents = float(df[(df.State == state) & (df.Year == year) & (df.Tech == tech)]["Electricity_$/kWh"].median())
state_rate = state_rate_cents / 100  # ✅ Convert to dollars
elec = st.sidebar.slider("Electricity price ($/kWh)", float(df["Electricity_$/kWh"].min())/100, float(df["Electricity_$/kWh"].max())/100, state_rate)

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
tabs = st.tabs([
    "Prediction", 
    "Stats", 
    "Visuals", 
    "Map", 
    "State Comparison", 
    "U.S. Summary",    
    "Download"
])



with tabs[0]:
    st.markdown(f"### Predicted LCOH → **${pred:.2f}/kg** using **{model_name}**")

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
                     template="plotly_dark", title=" Feature Importance")
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
    dynamic_rows = []
    for st_name in sorted(df.State.unique()):
        try:
            subset = df.query("State==@st_name & Year==@year & Tech==@tech")
            if subset.empty:
                continue
            cf_val = float(subset["CF"].iloc[0])
            elec_val = float(subset["Electricity_$/kWh"].median()) / 100
        except (IndexError, ValueError):
            continue

        dynamic_rows.append({
            "Year": year,
            "CF": cf_val,
            "CAPEX_$/kW": cap,
            "Efficiency_kWh_kg": eff,
            "Electricity_$/kWh": elec_val,
            "Water_$kg": df["Water_$kg"].median(),
            "CO2_$kg": df["CO2_$kg"].median(),
            "Transport_$kg": df["Transport_$kg"].median(),
            "Storage_$kg": df["Storage_$kg"].median(),
            "Tech_PEM": int(tech == "PEM"),
            "Tech_SOEC": int(tech == "SOEC"),
            "State": st_name
        })

    # --- Predictions ---
    df_inputs = pd.DataFrame(dynamic_rows)
    df_pred = df_inputs.drop(columns=["State"])
    df_inputs["LCOH ($/kg)"] = best.predict(df_pred)

    # --- Merge with geometry ---
    merged = gdf.merge(df_inputs[["State", "LCOH ($/kg)"]], on="State", how="left")

    # Ensure geometry is valid GeoJSON
    merged["geometry"] = merged["geometry"].simplify(0.01)  # Optional smoothing
    merged = merged.set_geometry("geometry")

    fig_map = px.choropleth(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color="LCOH ($/kg)",
        hover_name="State",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title=f"Predicted LCOH by State ({year}, {tech}, Model: {model_name})"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_coloraxes(colorbar_tickformat=".2f")
    st.plotly_chart(fig_map, use_container_width=True)


with tabs[4]:
    st.subheader("State-wise LCOH Summary (Simulated Variations)")

    variations = []
    cap_values = [cap * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    eff_values = [eff * f for f in [0.95, 0.975, 1.0, 1.025, 1.05]]
    elec_factors = [0.9, 0.95, 1.0, 1.05, 1.1]  # NEW

    for st_name in sorted(df.State.unique()):
        try:
            subset = df.query("State == @st_name & Year == @year & Tech == @tech")
            if subset.empty:
                continue
            cf_val = float(subset["CF"].iloc[0])
            elec_cents = float(subset["Electricity_$/kWh"].median())
            elec_base = elec_cents / 100.0  # ✅ Cents → Dollars
        except (IndexError, ValueError):
            continue

        for c, e, ep in zip(cap_values, eff_values, elec_factors):
            row = {
                "Year": year,
                "CF": cf_val,
                "CAPEX_$/kW": round(c, 2),
                "Efficiency_kWh_kg": round(e, 2),
                "Electricity_$/kWh": round(elec_base * ep, 4),  # ✅ Vary electricity too
                "Water_$kg": round(df["Water_$kg"].median(), 4),
                "CO2_$kg": round(df["CO2_$kg"].median(), 4),
                "Transport_$kg": round(df["Transport_$kg"].median(), 4),
                "Storage_$kg": round(df["Storage_$kg"].median(), 4),
                "Tech_PEM": int(tech == "PEM"),
                "Tech_SOEC": int(tech == "SOEC"),
                "State": st_name
            }
            variations.append(row)

    df_var = pd.DataFrame(variations)
    df_pred = df_var.drop(columns=["State"])
    df_var["LCOH ($/kg)"] = best.predict(df_pred)

    summary_stats = (
        df_var.groupby("State")["LCOH ($/kg)"]
        .agg(["mean", "min", "max", "std"])
        .applymap(lambda x: float(f"{x:.2f}"))
    )

    if summary_stats["std"].eq(0).all():
        summary_stats.drop(columns="std", inplace=True)

    st.dataframe(
        summary_stats.style
        .format(precision=2)
        .background_gradient(cmap="viridis"),
        use_container_width=True
    )

    # Store for downloads
    y_true = results[model_name]["y_true"]
    preds = results[model_name]["preds"]
    df_out = pd.DataFrame({"True": y_true, "Predicted": preds})
    df_out["Error"] = df_out["True"] - df_out["Predicted"]

with tabs[5]:
    st.subheader("🇺🇸 National LCOH Overview")

    # Mean LCOH per year (national)
    national_trend = df[df.Tech == tech].groupby("Year")["LCOH_$kg"].mean().reset_index()
    national_trend["LCOH_$kg"] = national_trend["LCOH_$kg"].round(2)

    fig_nat = px.line(
        national_trend,
        x="Year", y="LCOH_$kg",
        title=f" National Average LCOH by Year ({tech})",
        markers=True,
        template="plotly_dark"
    )
    fig_nat.update_layout(yaxis_title="LCOH ($/kg)", xaxis_title="Year")
    st.plotly_chart(fig_nat, use_container_width=True)

    st.dataframe(national_trend.rename(columns={"LCOH_$kg": "Average LCOH ($/kg)"}))

with tabs[6]:
    st.subheader("Download Center")

    import matplotlib.pyplot as plt
    from fpdf import FPDF
    import tempfile, io, zipfile

    download_mode = st.radio("Select download type", ["Scenario Report Only", "Full ZIP + Report"])

    # --- Common PDF plot ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y_true, label="True", linewidth=2)
    ax.plot(preds, label="Predicted", linestyle="--")
    ax.set_title(f"{model_name} – Prediction vs True")
    ax.legend()
    fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    fig.savefig(fig_path, bbox_inches="tight")

    # --- PDF generator ---
    def generate_pdf(full=False):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "LCOH Model Report", ln=True)

        # Model performance
        for name, res in results.items():
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, f"Model: {name}", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"R²: {res['r2']:.3f} | RMSE: {res['rmse']:.3f} | MAE: {res['mae']:.3f}", ln=True)
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(40, 10, "True", 1)
            pdf.cell(40, 10, "Predicted", 1)
            pdf.cell(40, 10, "Error", 1)
            pdf.ln()
            pdf.set_font("Arial", "", 12)
            for i in range(min(5, len(res["y_true"]))):
                yt, yp = res["y_true"][i], res["preds"][i]
                pdf.cell(40, 10, f"{yt:.2f}", 1)
                pdf.cell(40, 10, f"{yp:.2f}", 1)
                pdf.cell(40, 10, f"{yt - yp:.2f}", 1)
                pdf.ln()
            pdf.ln(5)

        pdf.image(fig_path, x=10, w=180)

        # Scenario Info
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Selected Scenario", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"Tech: {tech}\nYear: {year}\nState: {state}\nCAPEX: ${cap}/kW\nEfficiency: {eff} kWh/kg\nElectricity Price: ${elec}/kWh")

        if full:
            # National Trend
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "National LCOH by Year", ln=True)
            pdf.set_font("Arial", "", 12)
            for _, row in national_trend.iterrows():
                pdf.cell(0, 10, f"{int(row['Year'])}: ${row['LCOH_$kg']:.2f}/kg", ln=True)

            # State Summary
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "State-wise LCOH Summary", ln=True)
            pdf.set_font("Arial", "B", 12)
            for col in summary_stats.reset_index().columns:
                pdf.cell(40, 10, str(col), 1)
            pdf.ln()
            pdf.set_font("Arial", "", 12)
            for _, row in summary_stats.reset_index().iterrows():
                for val in row:
                    pdf.cell(40, 10, str(val), 1)
                pdf.ln()
        return pdf

    if download_mode == "Scenario Report Only":
        scenario_pdf = generate_pdf(full=False)
        buffer = io.BytesIO()
        buffer.write(scenario_pdf.output(dest="S").encode("latin1"))
        buffer.seek(0)
        st.download_button("Download Scenario PDF", data=buffer, file_name="Scenario_Report.pdf", mime="application/pdf")

    else:
        full_pdf = generate_pdf(full=True)
        pdf_bytes = full_pdf.output(dest="S").encode("latin1")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            zipf.writestr("LCOH_Full_Report.pdf", pdf_bytes)
            zipf.writestr("predictions.csv", df_out.to_csv(index=False))
            zipf.writestr("state_summary.csv", summary_stats.to_csv())
            zipf.writestr("national_trend.csv", national_trend.to_csv(index=False))
        zip_buffer.seek(0)
        st.download_button(" Download Full ZIP", data=zip_buffer, file_name="LCOH_Complete_Outputs.zip", mime="application/zip")
