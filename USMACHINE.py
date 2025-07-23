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
import os, io, zipfile, tempfile
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt     # if you use it anywhere


# --- Streamlit setup ---
def read_quoted_csv(path, col_names):
    """
    Handles CSV files where every row—including the header—is wrapped in quotes.
    """
    return (pd.read_csv(
                path, quotechar='"', header=None, names=col_names, skipinitialspace=True
            )
            .assign(**{c: lambda d, c=c: pd.to_numeric(d[c], errors="coerce")
                       for c in col_names[1:]}))

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
    # ---------- 1) TARGET tables ----------
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base):
        base = "."

    target_dfs = []
    for tech, csv_name in [
        ("ALKALINE", "USAALKALINE.csv"),
        ("PEM",       "USAPEM.csv"),
        ("SOEC",      "USASOEC.csv")
    ]:
        fp = os.path.join(base, csv_name)
        if not os.path.exists(fp):
            st.error(f"Missing target file: {fp}")
            st.stop()
        d = pd.read_csv(fp)
        tgt = [c for c in d.columns if "LCOH" in c.upper()][0]
        d = d.rename(columns={tgt: "LCOH_$kg"})
        d["Tech"] = tech
        target_dfs.append(d)

    lcoh_df = pd.concat(target_dfs, ignore_index=True)

    # ---------- 2) FEATURE tables ----------
    elec = load_electricity_prices()

    ghi = read_quoted_csv(
        "GHIUSA.csv",
        ["State", "GHI_min", "GHI_max", "GHI_mean", "GHI_median", "GHI_std"]
    )

    wind = read_quoted_csv(
        "WINDUSA.csv",
        ["State", "Wind_min", "Wind_max", "Wind_mean", "Wind_median", "Wind_std"]
    )

    # GRID distance
    grid = pd.read_csv("STATEGRIDDISTANCE.csv").rename(columns=lambda c: c.strip())
    if "State" not in grid.columns:
        for c in grid.columns:
            if c.lower().startswith("state"):
                grid = grid.rename(columns={c: "State"})
                break

    # WATER
    water = pd.read_csv("USWATER.csv").rename(columns=lambda c: c.strip())
    if "State" not in water.columns:
        for c in water.columns:
            if c.lower().startswith("state"):
                water = water.rename(columns={c: "State"})
                break
    water = water[[c for c in ["State", "runoff_km3", "runoff_mm"] if c in water.columns]]

    # LAND
    land_raw = pd.read_csv("USLAND.csv").rename(
        columns={"NAME": "State",
                 "solar": "Solar_idx",
                 "wind":  "LandWind_idx",
                 "suitability": "Suitability_idx"}
    )
    desired = ["State", "Solar_idx", "LandWind_idx", "Suitability_idx"]
    land = land_raw[[c for c in desired if c in land_raw.columns]]

    # INFRA
    infra = (
        pd.read_excel("INFRAUSA.xlsx")
          .rename(columns={"Energy": "Infra_Energy",
                           "Water":  "Infra_Water",
                           "Dams":   "Infra_Dams"})
          [["State", "Infra_Energy", "Infra_Water", "Infra_Dams"]]
    )

    # CO2
    co2 = (
        pd.read_csv("USACO2.csv")
          .rename(columns={"2023 (million metric tons CO₂)": "CO2_2023_Mt"})
          [["State", "CO2_2023_Mt"]]
    )

    # BIOMASS
    biomass = pd.read_csv("USABIOMASS.csv").rename(
        columns={"NAME": "State", "AGB_Mg_ha": "AGB_Mg_ha"}
    )

    # ---------- 3) MERGE ----------
    features_df = (
        elec
        .merge(ghi,     on="State", how="left")
        .merge(wind,    on="State", how="left")
        .merge(grid,    on="State", how="left")
        .merge(water,   on="State", how="left")
        .merge(land,    on="State", how="left")
        .merge(infra,   on="State", how="left")
        .merge(co2,     on="State", how="left")
        .merge(biomass, on="State", how="left")
    )

    # ---------- 4) COMBINE ----------
    lower48 = set(features_df.State)
    lcoh_df = lcoh_df[lcoh_df.State.isin(lower48)]
    df = lcoh_df.merge(features_df, on="State", how="left")

    # ---------- 5) SHAPES ----------
    geo_url = (
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
        "master/data/geojson/us-states.json"
    )
    gdf = (
        gpd.read_file(geo_url)
          .rename(columns={"name": "State"})
          .query("State in @lower48")
    )

    return df, gdf


    # Keep only lower-48 states (already present in several tables)
    lower48 = set(features_df.State)
    lcoh_df = lcoh_df[lcoh_df.State.isin(lower48)]

    # ---------- 4) Combine targets + features ----------
    df = lcoh_df.merge(features_df, on="State", how="left")

    # ---------- 5) Geo shapes ----------
    geo_url = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
               "master/data/geojson/us-states.json")
    gdf = (gpd.read_file(geo_url)
             .rename(columns={"name":"State"})
             .query("State in @lower48"))

    return df, gdf

df, gdf = load_data()
st.sidebar.markdown(f"<b>Rows loaded:</b> {len(df)}", unsafe_allow_html=True)
if df.empty:
    st.error("No data loaded — check your CSV path!")
    st.stop()
    
# --- Model trainer ---------------------------------------------------------- #
@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Fit several regressors.

    • Two-step imputation:
        1. Fill each numeric column’s NaN with its median.
        2. Convert any remaining NaN or ±Inf to 0.
    • If, after cleaning, < 5 rows remain, the function returns {} so the
      dashboard falls back to True-Equation mode.
    """
    # 1️⃣  Build feature matrix ------------------------------------------------
    df_proc = df.copy()
    df_proc["Year_val"] = df_proc["Year"]

    X = pd.get_dummies(
        df_proc.drop(columns=["State", "LCOH_$kg", "Year"]),
        columns=["Tech"],
        drop_first=True,
    )
    X["Year_val_squared"] = df_proc["Year_val"] ** 2
    y = df_proc["LCOH_$kg"]

    # 2️⃣  Robust imputation ---------------------------------------------------
    X_imputed = X.copy()

    # 2-a) median for numeric cols
    medians = X_imputed.median(numeric_only=True)
    X_imputed[medians.index] = X_imputed[medians.index].fillna(medians)

    # 2-b) replace ±Inf → NaN, then any leftover NaN → 0
    X_imputed = (
        X_imputed.replace([np.inf, -np.inf], np.nan)
                 .fillna(0)
    )

    # 3️⃣  Keep rows where target is finite -----------------------------------
    finite_mask = np.isfinite(y)
    X_clean, y_clean = X_imputed[finite_mask], y[finite_mask]

    # Diagnostics
    st.sidebar.write(f"Rows after impute/clean: {len(X_clean)}")
    if len(X_clean) < 5:
        st.sidebar.warning("❌ Fewer than 5 usable rows — ML mode disabled.")
        return {}

    # 4️⃣  Train / test split --------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.20, random_state=42
    )

    # 5️⃣  Candidate models ----------------------------------------------------
    models = {
        "Linear":       LinearRegression(),
        "Ridge":        Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "GBRegressor":  GradientBoostingRegressor(n_estimators=100),
        "XGBoost":      XGBRegressor(n_estimators=100, verbosity=0),
        "LightGBM":     LGBMRegressor(n_estimators=100),
        "CatBoost":     CatBoostRegressor(verbose=0),
    }

    results = {}
    t0 = time.time()
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results[name] = {
            "model":    model,
            "r2":       r2_score(y_te, preds),
            "rmse":     np.sqrt(mean_squared_error(y_te, preds)),
            "mae":      mean_absolute_error(y_te, preds),
            "preds":    preds.tolist(),
            "y_true":   y_te.tolist(),
            "features": X_clean.columns.tolist(),
        }

    st.sidebar.success(f"✅ Models trained in {time.time() - t0:.1f}s")
    return results, X_tr
# --------------------------------------------------------------------------- #

# ── Train models (or fail fast) ─────────────────────────────────────────────
out = train_models(df)

ml_available = isinstance(out, tuple) and len(out) == 2 and bool(out[0])

if ml_available:
    results, X_train = out
else:
    results, X_train = {}, None  # lets the app keep running in Baseline TEA mode
# --------------------------------------------------------------------------- #

# --- 🔧 Scenario Simulator ---
st.sidebar.header("🔧 Scenario Simulator")

calculation_mode = st.sidebar.radio("Prediction mode", ["ML Model", "Baseline TEA"])
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

# ---------- Handy one-liner to bullet-proof the row -------------------------
def sanitize(df, feature_order):
    """
    Align to training feature order, coerce everything to numeric,
    replace ±Inf → NaN → 0, and return a float64 matrix.
    """
    df_out = (df.reindex(columns=feature_order, fill_value=np.nan)
                 .apply(pd.to_numeric, errors="coerce")
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0)
                 .astype("float64"))
    return df_out
# ---------------------------------------------------------------------------

# Pull data from filtered CSV
row_df = df.query("State == @state and Year == @year and Tech == @tech")

if row_df.empty:
    st.error("⚠️ No data for that State / Tech / Year. Pick a different year.")
    st.stop()                      # exits cleanly inside Streamlit
else:
    row_base = row_df.iloc[0]

# --- Inputs -----------------------------------------------------------------
ml_mode = (calculation_mode == "ML Model") and ml_available   # True only if
                                                              #  • the sidebar
                                                              #    toggle says
                                                              #    “ML Model”, AND
                                                              #  • we managed to
                                                              #    train models

# Always start with the CSV defaults
cap_dynamic = row_base["CAPEX_$/kW"]
eff_dynamic = row_base["Efficiency_kWh_kg"]
elec        = row_base["Electricity_$/kWh"]

if ml_mode:                          # ── branch ①: ML prediction ────────────
    model_name = st.sidebar.selectbox("Pick model", list(results.keys()))
    best       = results[model_name]["model"]

    st.sidebar.caption(
        f"CAPEX range: {cap_min}–{cap_max} | "
        f"Efficiency range: {eff_min}–{eff_max}"
    )

else:                                # ── branch ②: Baseline TEA / no-ML ────
    st.sidebar.markdown("### Using CSV values (Baseline TEA mode)")
    st.sidebar.metric("CAPEX ($/kW)",        f"{cap_dynamic:.2f}")
    st.sidebar.metric("Efficiency (kWh/kg)", f"{eff_dynamic:.2f}")
    st.sidebar.metric("Electricity ($/kWh)", f"{elec:.4f}")

    # Optional U.S.-DOE slider overrides
    tech_info = {
        "ALKALINE": "https://www.energy.gov/eere/fuelcells/technical-targets-liquid-alkaline-electrolysis",
        "PEM":      "https://www.energy.gov/eere/fuelcells/technical-targets-proton-exchange-membrane-electrolysis",
        "SOEC":     "https://www.energy.gov/eere/fuelcells/technical-targets-high-temperature-electrolysis",
    }

    auto_mode = st.sidebar.checkbox(
        "Apply U.S. DOE Electrolyzer Targets",
        value=True,
        help=f"Uses CAPEX and efficiency projections from the U.S. Department of Energy. Learn more: {tech_info.get(tech, '')}"
    )

    cap_doe = adjust_capex_by_year(tech, year)
    eff_doe = get_efficiency_by_year(tech, year)

    cap_default  = cap_doe if auto_mode else (cap_min + cap_max) / 2
    eff_default  = eff_doe if auto_mode else (eff_min + eff_max) / 2
    elec_default = elec

    cap_val = st.sidebar.slider("CAPEX ($/kW)",
                                int(cap_min), int(cap_max), int(cap_default),
                                step=1, disabled=auto_mode)
    eff_val = st.sidebar.slider("Efficiency (kWh/kg)",
                                float(eff_min), float(eff_max), float(eff_default),
                                step=0.1, disabled=auto_mode)
    elec    = st.sidebar.slider("Electricity price ($/kWh)",
                                0.01, 0.30, float(round(elec_default, 4)),
                                step=0.01)

    cap_dynamic = cap_doe if auto_mode else cap_val
    eff_dynamic = eff_doe if auto_mode else eff_val

# --- Predict -----------------------------------------------------------------
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

# ▸ Add any missing ML features (only when an ML model is active)
if ml_mode:
    # 1. Tech one-hot flags
    for col in results[model_name]["features"]:
        if col.startswith("Tech_"):
            row[col] = 1 if col == f"Tech_{tech}" else 0

    # 2. Ensure row has every expected column
    df_in = pd.DataFrame([row])
    for col in results[model_name]["features"]:
        if col not in df_in:
            df_in[col] = 0
    df_in = df_in[results[model_name]["features"]]

       # 3. Final cleanup & alignment
    df_in = sanitize(df_in, results[model_name]["features"])

    pred = float(best.predict(df_in)[0])


else:  # ▸ Baseline TEA path
    def true_lcoh(cap, eff, cf, elec_p, wc, cs, tc, sc,
                  cap_ref=500, disc=0.08):
        crf = disc * (1 + disc) ** 20 / ((1 + disc) ** 20 - 1)
        kgH2 = cf * 8760 / eff
        return round(
            cap * crf / kgH2 +
            cap * 0.05 * (cap / cap_ref) / kgH2 +
            elec_p * eff +
            wc + cs + tc + sc, 4
        )

    pred = true_lcoh(
        cap_dynamic, eff_dynamic, row["CF"], elec,
        row["Water_$kg"], row["CO2_$kg"],
        row["Transport_$kg"], row["Storage_$kg"],
        cap_ref=cap_ref
    )

# --- UI Output & Stats ----------------------------------------------------
tabs = st.tabs(
    ["Prediction", "Stats", "Visuals", "Map",
     "State Comparison", "U.S. Summary", "Download"]
)

def generate_pdf(row, pred, fig_path=None, full=False):
    """
    Build a scenario-only PDF (1 page) or a full multi-page report.
    Pass full=True only from the Download Center tab.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LCOH Scenario Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(
        0, 10,
        f"Mode: {calculation_mode}\n"
        f"Technology: {tech}\n"
        f"Year: {year}\n"
        f"State: {state}\n"
        f"CAPEX: ${row['CAPEX_$/kW']}/kW\n"
        f"Efficiency: {row['Efficiency_kWh_kg']} kWh/kg\n"
        f"Electricity Price: ${row['Electricity_$/kWh']}/kWh\n"
        f"Capacity Factor: {row['CF']}\n"
        f"Predicted LCOH: ${pred:.2f}/kg"
    )

    # Optional residual plot
    if fig_path and os.path.exists(fig_path):
        pdf.image(fig_path, x=10, w=180)

    # ---------- Extra pages (only if full=True) ---------------------------
    if full:
        # NATIONAL TREND page (if available)
        if 'national_trend' in globals() and national_trend is not None:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "National LCOH by Year", ln=True)
            pdf.set_font("Arial", "", 12)
            for _, r in national_trend.iterrows():
                pdf.cell(
                    0, 10,
                    f"{int(r['Year'])}: "
                    f"${r['Average LCOH ($/kg)']:.2f}/kg",
                    ln=True
                )

        # STATE SUMMARY table (if available)
        if 'summary_stats' in globals() and summary_stats is not None:
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
# -------------------------------------------------------------------------
# 🧪 Tab 0: Prediction
with tabs[0]:

    st.markdown("## Scenario Prediction")
    st.metric("Predicted LCOH", f"${pred:.2f}/kg")

    # ── Show inputs ───────────────────────────────────────────────────────
    with st.expander("Inputs Used", expanded=False):
        st.write({
            "Mode": calculation_mode,
            "Technology": tech,
            "Year": year,
            "State": state,
            "CAPEX ($/kW)": cap_dynamic,
            "Efficiency (kWh/kg)": eff_dynamic,
            "Electricity Price ($/kWh)": elec,
            "Capacity Factor": row["CF"],
        })



with tabs[1]:
    st.markdown("## Model Stats")

    if ml_mode:                              # ← ONLY fires when model_name exists
        stats_df = pd.DataFrame({
            k: {"R²": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]}
            for k, v in results.items()
        }).T.round(3)
        st.dataframe(stats_df, use_container_width=True)

        st.success(
            f"**{model_name}** — "
            f"R²: {stats_df.loc[model_name, 'R²']}, "
            f"RMSE: {stats_df.loc[model_name, 'RMSE']}, "
            f"MAE: {stats_df.loc[model_name, 'MAE']}"
        )
    else:
        st.info("Model stats only apply in ML Model mode.")


with tabs[2]:
    st.markdown("## Feature Importance & Trends")

    if calculation_mode == "ML Model":
        # 'best' exists globally once an ML model is selected
        if 'best' in globals():
            # ① Native importances for tree models
            fi = getattr(best, "feature_importances_", None)

            # ② Fallback: absolute coefficients for linear models
            if fi is None and hasattr(best, "coef_"):
                fi = np.abs(best.coef_)

            if fi is not None:
                cols = results[model_name]["features"]
                fi_s = pd.Series(np.ravel(fi), index=cols).sort_values()

                fig = px.bar(
                    fi_s.values,
                    y=fi_s.index,
                    orientation="h",
                    color=fi_s.values,
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title=f"Feature Importance – {model_name}"
                )
                fig.update_coloraxes(colorbar_title=None)
                fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "ℹ️ This estimator doesn’t expose feature importances "
                    "or coefficients."
                )
        else:
            st.warning("⚠️ No model found to display feature importance.")
    else:
        st.info("ℹ️ Feature importance is only available for ML models.")

    # 🔥 Trend Heatmap — always rendered
    pivot = (
        df.groupby(["Tech", "Year"])["LCOH_$kg"]
          .mean()
          .reset_index()
          .pivot(index="Tech", columns="Year", values="LCOH_$kg")
          .sort_index(axis=1)
    )
    fig2 = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        template="plotly_dark",
        labels={"y": "Tech", "x": "Year", "color": "LCOH ($/kg)"},
        title="LCOH Trend Heatmap"
    )
    st.plotly_chart(fig2, use_container_width=True)


with tabs[3]:
    if calculation_mode == "ML Model":
        st.markdown(f"###  {state}: LCOH = ${pred:.2f}/kg")

        # Prepare map DataFrame
        map_df = gdf.copy()
        map_df["Highlight"] = (map_df["State"] == state)

        # Clean interactive choropleth map
        fig_map = px.choropleth(
            map_df,
            geojson=map_df.geometry,
            locations=map_df.index,
            color="Highlight",
            color_discrete_map={True: "yellow", False: "#cccccc"},
            hover_name="State",     # ✅ Only show state name
            hover_data={},          # ✅ Hide index/Highlight columns
            template="plotly_dark"
        )

        fig_map.update_traces(marker_line_color="white", showlegend=False)

        fig_map.update_layout(
            geo=dict(scope="usa", visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
        )

        st.plotly_chart(fig_map, use_container_width=True)

    elif calculation_mode == "Baseline TEA":
        st.subheader(f"{tech} LCOH Map – {year}")

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

                fig_te = px.choropleth(
                    merged,
                    geojson=merged.geometry,
                    locations=merged.index,
                    color="LCOH ($/kg)",
                    hover_name="State",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    title=f"{tech} LCOH Map – {year}"
                )

                fig_te.update_traces(marker_line_color="white")
                fig_te.update_layout(
                    geo=dict(scope="usa"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                )
                fig_te.update_coloraxes(colorbar_tickformat=".2f")

                st.plotly_chart(fig_te, use_container_width=True)
        except FileNotFoundError:
            st.error(f"Missing data file: `{file_name}`")

    else:
        st.info("🌐 Map available for ML Model or True Equation only.")


with tabs[4]:
    st.subheader("State-wise LCOH Summary (Simulated Variations)")

    variations  = []
    cap_values  = [cap_dynamic * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    eff_values  = [eff_dynamic * f for f in [0.95, 0.975, 1.0, 1.025, 1.05]]
    features    = results[model_name]["features"] if ml_mode else []

    for st_name in sorted(df.State.unique()):
        subset = df.query("State == @st_name & Year == @year & Tech == @tech")
        if subset.empty:
            continue

        cf_val   = float(subset["CF"].iloc[0])
        elec_val = float(subset["Electricity_$/kWh"].iloc[0])
        wc       = float(subset["Water_$kg"].iloc[0])
        cs       = float(subset["CO2_$kg"].iloc[0])
        tc       = float(subset["Transport_$kg"].iloc[0])
        sc       = float(subset["Storage_$kg"].iloc[0])

        for c, e in zip(cap_values, eff_values):
            row_sim = {
                "State": st_name,
                "CF": cf_val,
                "CAPEX_$/kW": round(c, 2),
                "Efficiency_kWh_kg": round(e, 2),
                "Electricity_$/kWh": round(elec_val, 4),
                "Water_$kg": wc,
                "CO2_$kg": cs,
                "Transport_$kg": tc,
                "Storage_$kg": sc
            }

            if ml_mode:                                     # ← ML prediction path
                row_sim.update({"Year_val": year,
                                "Year_val_squared": year ** 2})
                # one-hot tech columns expected by the model
                for col in features:
                    if col.startswith("Tech_"):
                        row_sim[col] = 1 if col == f"Tech_{tech}" else 0
                df_row = pd.DataFrame([row_sim])
                for col in features:
                    if col not in df_row:
                        df_row[col] = 0
                df_row = df_row[features]
                row_sim["LCOH ($/kg)"] = float(
                    results[model_name]["model"].predict(df_row)[0]
                )
            else:                                           # ← True-Equation path
                crf  = 0.08 * (1 + 0.08) ** 20 / ((1 + 0.08) ** 20 - 1)
                kgH2 = cf_val * 8760 / row_sim["Efficiency_kWh_kg"]
                row_sim["LCOH ($/kg)"] = round(
                    row_sim["CAPEX_$/kW"] * crf / kgH2 +
                    row_sim["CAPEX_$/kW"] * 0.05 *
                    (row_sim["CAPEX_$/kW"] / cap_ref) / kgH2 +
                    row_sim["Electricity_$/kWh"] *
                    row_sim["Efficiency_kWh_kg"] +
                    wc + cs + tc + sc,
                    4
                )

            variations.append(row_sim)

    if not variations:
        st.warning("No data available for the selected state/tech/year.")
    else:
        df_var = pd.DataFrame(variations)
        summary_stats = (
            df_var.groupby("State")["LCOH ($/kg)"]
                  .agg(["mean", "min", "max", "std"])
                  .applymap(lambda x: float(f"{x:.2f}"))
        )
        if summary_stats["std"].eq(0).all():    # drop std if all zeros
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

    # ── Optional residual plot (only for ML mode) ─────────────────────────
    fig_path = None
    if ml_mode:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(results[model_name]["y_true"], label="True", linewidth=2)
        ax.plot(results[model_name]["preds"], label="Predicted", linestyle="--")
        ax.set_title(f"{model_name} – Prediction vs True")
        ax.legend()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        fig.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig_path = tmpfile.name

    # ── Choose export type ────────────────────────────────────────────────
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_mode = st.radio(
        "Choose export type",
        ["Scenario Report Only", "Full ZIP + Report"]
    )

    # ---------- Scenario-only PDF ----------------------------------------
    if download_mode == "Scenario Report Only":
        pdf_one = generate_pdf(row, pred, fig_path, full=False)
        buf     = io.BytesIO(pdf_one.output(dest="S").encode("latin1"))
        buf.seek(0)

        st.download_button(
            "Download Scenario PDF",
            data=buf,
            file_name=f"Scenario_Report_{timestamp}.pdf",
            mime="application/pdf",
            key="dl_scenario_only"
        )

    # ---------- Full ZIP bundle -----------------------------------------
    else:
        pdf_full = generate_pdf(row, pred, fig_path, full=True)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            # 1) main report
            z.writestr(
                f"LCOH_Full_Report_{timestamp}.pdf",
                pdf_full.output(dest="S").encode("latin1")
            )

            # 2) optional extras – only add if they exist
            df_out_opt         = globals().get("df_out")
            summary_stats_opt  = globals().get("summary_stats")
            national_trend_opt = globals().get("national_trend")

            if isinstance(df_out_opt, pd.DataFrame):
                z.writestr("predictions.csv", df_out_opt.to_csv(index=False))

            if isinstance(summary_stats_opt, pd.DataFrame):
                z.writestr("state_summary.csv", summary_stats_opt.to_csv())

            if isinstance(national_trend_opt, pd.DataFrame):
                z.writestr(
                    "national_trend.csv",
                    national_trend_opt.to_csv(index=False)
                )

        zip_buf.seek(0)
        st.download_button(
            "Download Full ZIP",
            data=zip_buf,
            file_name=f"LCOH_Report_Bundle_{timestamp}.zip",
            mime="application/zip",
            key="dl_full_zip"
        )

