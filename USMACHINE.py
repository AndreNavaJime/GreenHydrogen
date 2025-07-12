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
@st.cache_data
def load_electricity_prices():
    file_path = "ElectricityUSA.xlsx"
    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_path}. Please upload it to your working directory.")
        st.stop()

    df_elec = pd.read_excel(file_path)
    df_elec.columns = ["State", "Electricity_cents_kWh"]
    df_elec["Electricity_$/kWh_fixed"] = df_elec["Electricity_cents_kWh"] / 100
    return df_elec


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

    elec_df = load_electricity_prices()
    df = df.merge(elec_df[["State", "Electricity_$/kWh_fixed"]], on="State", how="left")

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

    df["Year_val"] = df["Year"]
    X = pd.get_dummies(
        df.drop(columns=["State", "LCOH_$kg", "Year"]),
        columns=["Tech"],
        drop_first=True
    )
    X["Year_val_squared"] = df["Year_val"] ** 2
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
            "y_true": y_te.tolist(),
            "features": X.columns.tolist()
        }

    st.sidebar.write(f"✅ Models trained in {time.time() - t0:.1f}s")
    return results, X_tr


results, X_train = train_models(df)
if not results:
    st.error("🚫 Training failed: not enough data.")
    st.stop()

# --- 🔧 Scenario Simulator ---
st.sidebar.header("🔧 Scenario Simulator")

calculation_mode = st.sidebar.radio("Prediction mode", ["ML Model", "True Equation"])
year_options = sorted(set(df.Year.unique()) | {2040, 2050, 2060})
year = st.sidebar.selectbox("Year", year_options)
tech = st.sidebar.selectbox("Technology", sorted(df.Tech.unique()))
state = st.sidebar.selectbox("State", sorted(df.State.unique()))

# Tech defaults
tech_settings = {
    "ALKALINE": {"cap_min": 150, "cap_max": 1000, "eff_min": 48.0, "eff_max": 55.0, "cap_ref": 500},
    "PEM": {"cap_min": 150, "cap_max": 1200, "eff_min": 46.0, "eff_max": 55.0, "cap_ref": 1000},
    "SOEC": {"cap_min": 200, "cap_max": 2500, "eff_min": 34.0, "eff_max": 47.0, "cap_ref": 2500}
}.get(tech, {"cap_min": 100, "cap_max": 3000, "eff_min": 30.0, "eff_max": 60.0, "cap_ref": 500})

cap_min, cap_max = tech_settings["cap_min"], tech_settings["cap_max"]
eff_min, eff_max = tech_settings["eff_min"], tech_settings["eff_max"]
cap_ref = tech_settings["cap_ref"]

# DOE trend helpers
def adjust_capex_by_year(tech, year):
    t = {
        "ALKALINE": {2022: 500, 2026: 250, 2031: 150, 2040: 100, 2050: 80, 2060: 60},
        "PEM": {2022: 1000, 2026: 250, 2031: 150, 2040: 100, 2050: 80, 2060: 60},
        "SOEC": {2022: 2500, 2026: 500, 2031: 200, 2040: 160, 2050: 120, 2060: 100}
    }.get(tech, {})
    years = sorted(t)
    if year <= years[0]: return t[years[0]]
    if year >= years[-1]: return t[years[-1]]
    for i in range(len(years)-1):
        if years[i] <= year <= years[i+1]:
            return round(np.interp(year, [years[i], years[i+1]], [t[years[i]], t[years[i+1]]]), 2)

def get_efficiency_by_year(tech, year):
    t = {
        "ALKALINE": {2022: 55, 2026: 52, 2031: 48, 2040: 45, 2050: 43, 2060: 42},
        "PEM": {2022: 55, 2026: 51, 2031: 46, 2040: 43, 2050: 42, 2060: 41},
        "SOEC": {2022: 47, 2026: 44, 2031: 42, 2040: 41, 2050: 40, 2060: 38}
    }.get(tech, {})
    years = sorted(t)
    if year <= years[0]: return t[years[0]]
    if year >= years[-1]: return t[years[-1]]
    for i in range(len(years)-1):
        if years[i] <= year <= years[i+1]:
            return round(np.interp(year, [years[i], years[i+1]], [t[years[i]], t[years[i+1]]]), 2)

# Pull data from filtered CSV
row_df = df.query("State == @state and Year == @year and Tech == @tech")
if row_df.empty:
    st.error("⚠️ No data for selected combination.")
    st.stop()
row_base = row_df.iloc[0]

# --- Inputs ---
if calculation_mode == "ML Model":
    model_name = st.sidebar.selectbox("Pick model", list(results.keys()))
    best = results[model_name]["model"]

    st.sidebar.caption(f"CAPEX range: {cap_min}–{cap_max} | Efficiency range: {eff_min}–{eff_max}")

    # 📘 Dynamic DOE help links per technology
    tech_info = {
        "ALKALINE": "https://www.energy.gov/eere/fuelcells/technical-targets-liquid-alkaline-electrolysis",
        "PEM": "https://www.energy.gov/eere/fuelcells/technical-targets-proton-exchange-membrane-electrolysis",
        "SOEC": "https://www.energy.gov/eere/fuelcells/technical-targets-high-temperature-electrolysis"
    }

    auto_mode = st.sidebar.checkbox(
        "Apply U.S. DOE Electrolyzer Targets",
        value=True,
        help=f"Uses CAPEX and efficiency projections from the U.S. Department of Energy. Learn more: {tech_info.get(tech, '')}"
    )

    cap_doe = adjust_capex_by_year(tech, year)
    eff_doe = get_efficiency_by_year(tech, year)

    cap_default = cap_doe if auto_mode else (cap_min + cap_max) / 2
    eff_default = eff_doe if auto_mode else (eff_min + eff_max) / 2
    elec_default = row_base["Electricity_$/kWh"]

    cap_val = st.sidebar.slider("CAPEX ($/kW)", int(cap_min), int(cap_max), int(cap_default), step=1, disabled=auto_mode)
    eff_val = st.sidebar.slider("Efficiency (kWh/kg)", float(eff_min), float(eff_max), float(eff_default), step=0.1, disabled=auto_mode)
    elec = st.sidebar.slider("Electricity price ($/kWh)", 0.01, 0.30, float(round(elec_default, 4)), step=0.01)

    cap_dynamic = cap_doe if auto_mode else cap_val
    eff_dynamic = eff_doe if auto_mode else eff_val

else:
    st.sidebar.markdown("### Using CSV values (True Equation mode)")
    st.sidebar.metric("CAPEX ($/kW)", f"{row_base['CAPEX_$/kW']:.2f}")
    st.sidebar.metric("Efficiency (kWh/kg)", f"{row_base['Efficiency_kWh_kg']:.2f}")
    st.sidebar.metric("Electricity ($/kWh)", f"{row_base['Electricity_$/kWh']:.4f}")
    cap_dynamic = row_base["CAPEX_$/kW"]
    eff_dynamic = row_base["Efficiency_kWh_kg"]
    elec = row_base["Electricity_$/kWh"]


# --- Predict ---
row = {
    "CF": row_base["CF"],
    "CAPEX_$/kW": cap_dynamic,
    "Efficiency_kWh_kg": eff_dynamic,
    "Electricity_$/kWh": elec,
    "Water_$kg": row_base["Water_$kg"],
    "CO2_$kg": row_base["CO2_$kg"],
    "Transport_$kg": row_base["Transport_$kg"],
    "Storage_$kg": row_base["Storage_$kg"],
    "Year_val": year,
    "Year_val_squared": year ** 2
}

if calculation_mode == "ML Model":
    for c in results[model_name]["features"]:
        if c.startswith("Tech_"):
            row[c] = 1 if c == f"Tech_{tech}" else 0
    df_in = pd.DataFrame([row])
    for c in results[model_name]["features"]:
        if c not in df_in: df_in[c] = 0
    df_in = df_in[results[model_name]["features"]]
    pred = best.predict(df_in)[0]
else:
    def true_lcoh(cap, eff, cf, elec_p, wc, cs, tc, sc, cap_ref=500, disc=0.08):
        crf = disc * (1 + disc) ** 20 / ((1 + disc) ** 20 - 1)
        kgH2 = cf * 8760 / eff
        return round(cap * crf / kgH2 +
                     cap * 0.05 * (cap / cap_ref) / kgH2 +
                     elec_p * eff +
                     wc + cs + tc + sc, 4)
    pred = true_lcoh(cap_dynamic, eff_dynamic, row["CF"], elec,
                     row["Water_$kg"], row["CO2_$kg"],
                     row["Transport_$kg"], row["Storage_$kg"],
                     cap_ref=cap_ref)

# --- UI Output & Stats ---
tabs = st.tabs(["Prediction", "Stats", "Visuals", "Map", "State Comparison", "U.S. Summary", "Download"])

# 🧪 Tab 0: Prediction
with tabs[0]:
    st.markdown("## Scenario Prediction")
    st.metric("Predicted LCOH", f"${pred:.2f}/kg")
    with st.expander("Inputs Used"):
        st.write({
            "Mode": calculation_mode,
            "Technology": tech,
            "Year": year,
            "State": state,
            "CAPEX ($/kW)": cap_dynamic,
            "Efficiency (kWh/kg)": eff_dynamic,
            "Electricity Price ($/kWh)": elec,
            "Capacity Factor": row["CF"]
        })

# 📊 Tab 1: Stats
with tabs[1]:
    st.markdown("## Model Stats")
    if calculation_mode == "ML Model":
        stats_df = pd.DataFrame({
            k: {"R²": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]}
            for k, v in results.items()
        }).T.round(3)
        st.dataframe(stats_df)
        st.success(f"**{model_name}** — R²: {stats_df.loc[model_name,'R²']}, RMSE: {stats_df.loc[model_name,'RMSE']}, MAE: {stats_df.loc[model_name,'MAE']}")
    else:
        st.info(" Model stats only apply in ML Model mode.")


with tabs[2]:
    st.markdown("## Feature Importance & Trends")

    if calculation_mode == "ML Model":
        if 'best' in locals():
            fi = getattr(best, "feature_importances_", None)
            if fi is not None:
                cols = results[model_name]["features"]
                fi_s = pd.Series(fi, index=cols).sort_values()
                fig = px.bar(
                    fi_s.values, y=fi_s.index, orientation="h",
                    color=fi_s.values, color_continuous_scale="Viridis",
                    template="plotly_dark", title="Feature Importance"
                )
                fig.update_coloraxes(colorbar_title=None)
                fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No model found to display feature importance.")
    else:
        st.info("ℹ️ Feature importance is only available for ML models.")

    # 🔥 Trend Heatmap — always safe to run
    pivot = df.groupby(["Tech", "Year"])["LCOH_$kg"].mean().reset_index().pivot(index="Tech", columns="Year", values="LCOH_$kg")
    pivot = pivot.sort_index(axis=1)
    fig2 = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        template="plotly_dark",
        labels={"y": "Tech", "x": "Year", "color": "LCOH ($/kg)"},
        title="LCOH Trend Heatmap"
    )
    st.plotly_chart(fig2, use_container_width=True)
with tabs[3]:
    st.subheader("Predicted LCOH by State")

    if calculation_mode == "ML Model":
        dynamic_rows = []
        cf_vals = df.set_index("State")["CF"].to_dict()

        for st_name in sorted(df.State.unique()):
            cf_val = cf_vals.get(st_name, 0.6)  # fallback CF

            row = {
                "CF": cf_val,
                "CAPEX_$/kW": cap_dynamic,
                "Efficiency_kWh_kg": eff_dynamic,
                "Electricity_$/kWh": elec,
                "Water_$kg": df["Water_$kg"].median(),
                "CO2_$kg": df["CO2_$kg"].median(),
                "Transport_$kg": df["Transport_$kg"].median(),
                "Storage_$kg": df["Storage_$kg"].median(),
                "Tech_PEM": int(tech == "PEM"),
                "Tech_SOEC": int(tech == "SOEC"),
                "State": st_name
            }

            # One-hot encode year
            for col in results[model_name]["features"]:
                if col.startswith("Year_"):
                    row[col] = 1 if col == f"Year_{year}" else 0

            dynamic_rows.append(row)

        df_inputs = pd.DataFrame(dynamic_rows)

        # Align features with model
        df_features = df_inputs.drop(columns=["State"])
        df_features = pd.get_dummies(df_features)

        # Fill missing features with 0
        for c in results[model_name]["features"]:
            if c not in df_features:
                df_features[c] = 0
        df_features = df_features[results[model_name]["features"]]

        # Predict and clip
        preds = best.predict(df_features)
        df_inputs["LCOH ($/kg)"] = np.clip(preds, 0, None).round(2)

        merged = gdf.merge(df_inputs[["State", "LCOH ($/kg)"]], on="State", how="left")

        fig_map = px.choropleth(
            merged,
            geojson=merged.geometry,
            locations=merged.index,
            color="LCOH ($/kg)",
            hover_name="State",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            title=f"Predicted LCOH by State ({year}, {tech}, Model: {model_name})"
        )
        fig_map.update_geos(scope="usa")
        fig_map.update_coloraxes(colorbar_tickformat=".2f")
        st.plotly_chart(fig_map, use_container_width=True)

    elif calculation_mode == "True Equation":
        tech_file_map = {
            "ALKALINE": ("USAALKALINE.csv", "LCOH_Alk_$kg"),
            "PEM": ("USAPEM.csv", "LCOH_PEM_$kg"),
            "SOEC": ("USASOEC.csv", "LCOH_SOEC_$kg")
        }
        file_name, lcoh_col = tech_file_map[tech]

        try:
            df_tech = pd.read_csv(file_name)
            df_year = df_tech[df_tech["Year"] == year].copy()

            if df_year.empty:
                st.warning("No data available for selected year/tech.")
            else:
                df_year.rename(columns={lcoh_col: "LCOH ($/kg)"}, inplace=True)
                df_year["LCOH ($/kg)"] = df_year["LCOH ($/kg)"].round(2)
                merged = gdf.merge(df_year[["State", "LCOH ($/kg)"]], on="State", how="left")

                fig_map = px.choropleth(
                    merged,
                    geojson=merged.geometry,
                    locations=merged.index,
                    color="LCOH ($/kg)",
                    hover_name="State",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title=f"True Equation LCOH by State ({tech}, {year})"
                )
                fig_map.update_geos(scope="usa")
                fig_map.update_coloraxes(colorbar_tickformat=".2f")
                st.plotly_chart(fig_map, use_container_width=True)
        except FileNotFoundError:
            st.error(f"Missing data file: `{file_name}`")
    else:
        st.info("🌐 Map available for ML Model or True Equation only.")


with tabs[4]:
    st.subheader("State-wise LCOH Summary (Simulated Variations)")

    variations = []
    cap_values = [cap_dynamic * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    eff_values = [eff_dynamic * f for f in [0.95, 0.975, 1.0, 1.025, 1.05]]

    for st_name in sorted(df.State.unique()):
        try:
            subset = df.query("State == @st_name & Year == @year & Tech == @tech")
            if subset.empty:
                continue
            cf_val = float(subset["CF"].iloc[0])
            elec_val = float(subset["Electricity_$/kWh"].iloc[0])
        except (IndexError, ValueError):
            continue

        for c, e in zip(cap_values, eff_values):
            if calculation_mode == "ML Model":
                row = {
                    "CF": cf_val,
                    "CAPEX_$/kW": round(c, 2),
                    "Efficiency_kWh_kg": round(e, 2),
                    "Electricity_$/kWh": round(elec_val, 4),
                    "Water_$kg": round(df["Water_$kg"].median(), 4),
                    "CO2_$kg": round(df["CO2_$kg"].median(), 4),
                    "Transport_$kg": round(df["Transport_$kg"].median(), 4),
                    "Storage_$kg": round(df["Storage_$kg"].median(), 4),
                    "Tech_PEM": int(tech == "PEM"),
                    "Tech_SOEC": int(tech == "SOEC"),
                    "State": st_name
                }
                for col in results[model_name]["features"]:
                    if col.startswith("Year_"):
                        row[col] = 1 if col == f"Year_{year}" else 0
                row["LCOH ($/kg)"] = best.predict(pd.DataFrame([row]).reindex(columns=results[model_name]["features"], fill_value=0))[0]

            else:
                # True Equation calculation
                row = {
                    "CF": cf_val,
                    "CAPEX_$/kW": round(c, 2),
                    "Efficiency_kWh_kg": round(e, 2),
                    "Electricity_$/kWh": round(elec_val, 4),
                    "Water_$kg": round(df["Water_$kg"].median(), 4),
                    "CO2_$kg": round(df["CO2_$kg"].median(), 4),
                    "Transport_$kg": round(df["Transport_$kg"].median(), 4),
                    "Storage_$kg": round(df["Storage_$kg"].median(), 4),
                    "State": st_name
                }
                row["LCOH ($/kg)"] = round(
                    row["CAPEX_$/kW"] * 0.089 / (row["CF"] * 8760 / row["Efficiency_kWh_kg"]) +
                    row["CAPEX_$/kW"] * 0.05 * (row["CAPEX_$/kW"] / cap_ref) / (row["CF"] * 8760 / row["Efficiency_kWh_kg"]) +
                    row["Electricity_$/kWh"] * row["Efficiency_kWh_kg"] +
                    row["Water_$kg"] + row["CO2_$kg"] + row["Transport_$kg"] + row["Storage_$kg"], 4
                )

            variations.append(row)

    df_var = pd.DataFrame(variations)

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
    
with tabs[5]:
    st.subheader("🇺🇸 National LCOH Overview")

    national_df = df[df.Tech == tech]

    if national_df.empty:
        st.warning(f"No data available for {tech}.")
    else:
        if calculation_mode == "ML Model":
            national_trend = (
                national_df
                .groupby("Year")["LCOH_$kg"]
                .mean()
                .reset_index()
                .rename(columns={"LCOH_$kg": "Average LCOH ($/kg)"})
            )
        else:
            years = sorted(national_df["Year"].unique())
            rows = []
            for y in years:
                subset = national_df[national_df["Year"] == y]
                if subset.empty:
                    continue

                cap_val = adjust_capex_by_year(tech, y)
                eff_val = get_efficiency_by_year(tech, y)
                elec_val = subset["Electricity_$/kWh"].mean()
                cf_val = subset["CF"].mean()

                wc = subset["Water_$kg"].mean()
                cs = subset["CO2_$kg"].mean()
                tc = subset["Transport_$kg"].mean()
                sc = subset["Storage_$kg"].mean()

                crf = 0.08 * (1 + 0.08) ** 20 / ((1 + 0.08) ** 20 - 1)
                kgH2 = cf_val * 8760 / eff_val
                lcoh = (
                    cap_val * crf / kgH2 +
                    cap_val * 0.05 * (cap_val / cap_ref) / kgH2 +
                    elec_val * eff_val +
                    wc + cs + tc + sc
                )

                rows.append({"Year": y, "Average LCOH ($/kg)": round(lcoh, 2)})

            national_trend = pd.DataFrame(rows)

        # Plotting
        fig_nat = px.line(
            national_trend,
            x="Year",
            y="Average LCOH ($/kg)",
            title=f"🇺🇸 National Average LCOH by Year ({tech})",
            markers=True,
            template="plotly_dark"
        )
        fig_nat.update_layout(
            yaxis_title="LCOH ($/kg)",
            xaxis_title="Year",
            title_font_size=20
        )

        st.plotly_chart(fig_nat, use_container_width=True)
        st.dataframe(national_trend, use_container_width=True)

with tabs[6]:
    st.subheader("Download Center")

    import matplotlib.pyplot as plt
    from fpdf import FPDF
    import tempfile, io, zipfile
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_mode = st.radio("Choose export type", ["Scenario Report Only", "Full ZIP + Report"])

    # Optional plot only for ML model
    fig_path = None
    if calculation_mode == "ML Model":
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(results[model_name]["y_true"], label="True", linewidth=2)
        ax.plot(results[model_name]["preds"], label="Predicted", linestyle="--")
        ax.set_title(f"{model_name} – Prediction vs True")
        ax.legend()
        fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    # --- PDF Builder ---
    def generate_pdf(row, pred, full=False):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "LCOH Scenario Report", ln=True)

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10,
            f"Mode: {calculation_mode}\n"
            f"Technology: {tech}\n"
            f"Year: {year}\n"
            f"State: {state}\n"
            f"CAPEX: ${cap_dynamic}/kW\n"
            f"Efficiency: {eff_dynamic} kWh/kg\n"
            f"Electricity Price: ${elec}/kWh\n"
            f"Capacity Factor: {row['CF']}\n"
            f"Predicted LCOH: ${pred:.2f}/kg\n"
            f"Generated: {timestamp}"
        )

        if fig_path:
            pdf.image(fig_path, x=10, w=180)

        if full and calculation_mode == "ML Model":
            # National Trend
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "National LCOH by Year", ln=True)
            pdf.set_font("Arial", "", 12)
            for _, r in national_trend.iterrows():
                pdf.cell(0, 10, f"{int(r['Year'])}: ${r['Average LCOH ($/kg)']:.2f}/kg", ln=True)

            # State Summary
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "State-wise LCOH Summary", ln=True)
            pdf.set_font("Arial", "B", 12)
            for col in summary_stats.reset_index().columns:
                pdf.cell(40, 10, str(col), 1)
            pdf.ln()
            pdf.set_font("Arial", "", 12)
            for _, r in summary_stats.reset_index().iterrows():
                for val in r:
                    pdf.cell(40, 10, str(val), 1)
                pdf.ln()
        return pdf

    # --- Output Download Handling ---
    if download_mode == "Scenario Report Only":
        pdf = generate_pdf(row, pred, full=False)
        buffer = io.BytesIO()
        buffer.write(pdf.output(dest="S").encode("latin1"))
        buffer.seek(0)
        st.download_button(
            "Download Scenario PDF",
            data=buffer,
            file_name=f"Scenario_Report_{timestamp}.pdf",
            mime="application/pdf"
        )
    else:
        pdf = generate_pdf(row, pred, full=True)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            z.writestr(f"LCOH_Full_Report_{timestamp}.pdf", pdf.output(dest="S").encode("latin1"))
            if 'df_out' in locals():
                z.writestr("predictions.csv", df_out.to_csv(index=False))
            if 'summary_stats' in locals():
                z.writestr("state_summary.csv", summary_stats.to_csv())
            if 'national_trend' in locals():
                z.writestr("national_trend.csv", national_trend.to_csv(index=False))
        zip_buffer.seek(0)
        st.download_button(
            "Download Full ZIP",
            data=zip_buffer,
            file_name=f"LCOH_Report_Bundle_{timestamp}.zip",
            mime="application/zip"
        )
