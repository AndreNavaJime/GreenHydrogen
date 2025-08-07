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
import shap                     #  ← new
from sklearn.model_selection import KFold, cross_val_score   # ← for CV check
import streamlit as st, sys
st.sidebar.info(f"Running Python: {sys.version.split()[0]}")

# ─── SHAP helper: embed JS so force_plot works in Streamlit ──────────────
def st_shap(plot, height=300):
    """
    Streamlit wrapper for SHAP's force_plot.

    Parameters
    ----------
    plot : shap._force.AdditiveForceVisualizer
        The object returned by shap.force_plot().
    height : int
        iframe height in pixels.
    """
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


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
    Fit several regressors and attach SHAP + CV stats.
    """
    import shap  # local import keeps app light
    from sklearn.model_selection import train_test_split, KFold, cross_val_score

    # 1️⃣  Build feature matrix ------------------------------------------------
    df_p = df.copy()
    df_p["Year_val"] = df_p["Year"]

    X = pd.get_dummies(
        df_p.drop(columns=["State", "LCOH_$kg", "Year"]),
        columns=["Tech"],
        drop_first=True,
    )
    X["Year_val_squared"] = df_p["Year_val"] ** 2
    y = df_p["LCOH_$kg"]

    # 2️⃣  Robust imputation ---------------------------------------------------
    X = (
        X.replace([np.inf, -np.inf], np.nan)
        .fillna(X.median(numeric_only=True))
        .fillna(0)
    )

    # 3️⃣  Drop rows where y is non-finite ------------------------------------
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    st.sidebar.write(f"Rows after cleaning: {len(X)}")
    if len(X) < 5:
        st.sidebar.warning("Too few rows – ML disabled.")
        return {}

    # 4️⃣  Split ---------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5️⃣  Candidate models ----------------------------------------------------
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
        "GBRegressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, verbosity=0, random_state=42
        ),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    }

    results, t0 = {}, time.time()

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        # --- SHAP ------------------------------------------------------------
        try:
            bg = X_tr.sample(min(200, len(X_tr)), random_state=42)

            if name.lower() in {"linear", "ridge"}:
                explainer = shap.LinearExplainer(model, bg)
            elif name.lower() in {
                "xgboost",
                "lightgbm",
                "catboost",
                "randomforest",
                "gbregressor",
            }:
                explainer = shap.TreeExplainer(
                    model, feature_perturbation="tree_path_dependent"
                )
            else:
                explainer = shap.KernelExplainer(model.predict, bg)

            shap_raw = explainer.shap_values(bg.iloc[:500])

            # 🔧 Ensure a clean float64 array (handles list-return for multiclass)
            if isinstance(shap_raw, list):
                shap_vals = np.asarray(shap_raw[0], dtype=np.float64)
            else:
                shap_vals = np.asarray(shap_raw, dtype=np.float64)

        except Exception as e:
            explainer, shap_vals = None, None
            st.sidebar.warning(f"SHAP skipped for {name}: {e}")

        # --- quick 5-fold CV -------------------------------------------------
        cv_r2 = cross_val_score(
            model,
            X_tr,
            y_tr,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="r2",
        ).mean()

        # --- store -----------------------------------------------------------
        results[name] = {
            "model": model,
            "r2": r2_score(y_te, preds),
            "rmse": np.sqrt(mean_squared_error(y_te, preds)),
            "mae": mean_absolute_error(y_te, preds),
            "cv_r2": cv_r2,
            "preds": preds.tolist(),
            "y_true": y_te.tolist(),
            "features": X.columns.tolist(),
            "explainer": explainer,
            "shap_values": shap_vals,
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
# ---------------------------------------------------------------------------
# Cached helper – predict 48 states for current sliders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _predict48(df_v):
    """
    Return a Series of 48 LCOH predictions (index = state names).
    Uses the *current* sidebar sliders and calculation_mode.
    """
    if ml_mode:                       # ML-prediction path
        feats = results[model_name]["features"]
        X = pd.DataFrame(0, index=df_v.index, columns=feats)

        # core features
        X["CF"]                 = df_v["CF"]
        X["CAPEX_$/kW"]         = cap_dynamic
        X["Efficiency_kWh_kg"]  = eff_dynamic
        X["Electricity_$/kWh"]  = df_v["Electricity_$/kWh"]
        X["Water_$kg"]          = df_v["Water_$kg"]
        X["CO2_$kg"]            = df_v["CO2_$kg"]
        X["Transport_$kg"]      = df_v["Transport_$kg"]
        X["Storage_$kg"]        = df_v["Storage_$kg"]
        X["Year_val"]           = year
        X["Year_val_squared"]   = year ** 2

        tc_flag = f"Tech_{tech}"
        if tc_flag in feats:
            X[tc_flag] = 1

        X = sanitize(X, feats)
        yhat = results[model_name]["model"].predict(X)

    else:                            # deterministic path
        crf  = 0.08 * (1 + 0.08) ** 20 / ((1 + 0.08) ** 20 - 1)
        kgH2 = df_v["CF"] * 8760 / eff_dynamic
        yhat = (
            cap_dynamic * crf / kgH2
          + cap_dynamic * 0.05 * (cap_dynamic / cap_ref) / kgH2
          + df_v["Electricity_$/kWh"] * eff_dynamic
          + df_v["Water_$kg"] + df_v["CO2_$kg"]
          + df_v["Transport_$kg"] + df_v["Storage_$kg"]
        )

    return pd.Series(yhat, index=df_v.index, dtype="float64")

# Pull data from filtered CSV
row_df = df.query("State == @state and Year == @year and Tech == @tech")

if row_df.empty:
    st.error("⚠️ No data for that State / Tech / Year. Pick a different year.")
    st.stop()                      # exits cleanly inside Streamlit
else:
    row_base = row_df.iloc[0]

# --- Inputs -----------------------------------------------------------------
ml_mode = (calculation_mode == "ML Model") and ml_available      # True when user picked “ML Model”
                                                                #   and at least one model trained

# (1)  Pick an ML regressor only when in ML mode
if ml_mode:
    model_name = st.sidebar.selectbox("Pick model", list(results.keys()))
    best       = results[model_name]["model"]
else:
    model_name = None   # keeps the API uniform later on

# (2)  Parameter source: DOE targets vs. Fix inputs across years
tech_info = {
    "ALKALINE": "https://www.energy.gov/eere/fuelcells/technical-targets-liquid-alkaline-electrolysis",
    "PEM":      "https://www.energy.gov/eere/fuelcells/technical-targets-proton-exchange-membrane-electrolysis",
    "SOEC":     "https://www.energy.gov/eere/fuelcells/technical-targets-high-temperature-electrolysis",
}

param_source = st.sidebar.radio(
    "Parameter source",
    ["DOE targets", "Fix inputs across years"],
    horizontal=True,
    help=(
        "**DOE targets**: sliders load the official DOE trajectory when you switch year/tech, but can still be tweaked afterward.  \n"
        "**Fix inputs across years**: once you tweak a slider, its value stays the same across all year/tech selections."
    )
)
apply_doe = (param_source == "DOE targets")

# ► compute the official defaults for this Year/Tech
cap_official  = adjust_capex_by_year(tech, year)
eff_official  = get_efficiency_by_year(tech, year)
elec_official = row_base["Electricity_$/kWh"]

# ► initialize session_state on first run
if "cap" not in st.session_state:
    st.session_state.cap = cap_official
if "eta" not in st.session_state:
    st.session_state.eta = eff_official
if "elec_price" not in st.session_state:
    st.session_state.elec_price = elec_official
if "last_year" not in st.session_state:
    st.session_state.last_year = year
if "last_tech" not in st.session_state:
    st.session_state.last_tech = tech

# ► if in DOE mode and year/tech changed, reset to official defaults
if apply_doe and (
    st.session_state.last_year != year
    or st.session_state.last_tech != tech
):
    st.session_state.cap        = cap_official
    st.session_state.eta        = eff_official
    st.session_state.elec_price = elec_official

# ► remember current year/tech
st.session_state.last_year = year
st.session_state.last_tech = tech

# ► dynamic slider ranges include tech bounds, DOE defaults & manual input
slider_cap_min = int(min(cap_min, cap_official, st.session_state.cap))
slider_cap_max = int(max(cap_max, cap_official, st.session_state.cap))

slider_eff_min = float(min(eff_min, eff_official, st.session_state.eta))
slider_eff_max = float(max(eff_max, eff_official, st.session_state.eta))

# ► sliders bound to session_state (no explicit default argument)
cap_dynamic = st.sidebar.slider(
    "CAPEX ($/kW)",
    slider_cap_min, slider_cap_max,
    step=1,
    key="cap"
)
eff_dynamic = st.sidebar.slider(
    "Efficiency (kWh/kg)",
    slider_eff_min, slider_eff_max,
    step=0.1,
    key="eta"
)
elec = st.sidebar.slider(
    "Electricity price ($/kWh)",
    0.01, 0.30,
    step=0.01,
    key="elec_price"
)

st.sidebar.caption(
    f"CAPEX range: {slider_cap_min}–{slider_cap_max} | "
    f"Efficiency range: {slider_eff_min:.1f}–{slider_eff_max:.1f}"
)



# --- Predict -----------------------------------------------------------------
row = {
    "CF":                 row_base["CF"],
    "CAPEX_$/kW":         cap_dynamic,
    "Efficiency_kWh_kg":  eff_dynamic,
    "Electricity_$/kWh":  elec,
    "Water_$kg":          row_base["Water_$kg"],
    "CO2_$kg":            row_base["CO2_$kg"],
    "Transport_$kg":      row_base["Transport_$kg"],
    "Storage_$kg":        row_base["Storage_$kg"],
    "Year_val":           year,
    "Year_val_squared":   year ** 2,
}

if ml_mode:                                                    # ▸ ML path
    # 1️⃣ add one-hot tech flags missing from the single-row dict
    for col in results[model_name]["features"]:
        if col.startswith("Tech_"):
            row[col] = 1 if col == f"Tech_{tech}" else 0

    # 2️⃣ build DataFrame with *exactly* the columns the model expects
    df_in = pd.DataFrame([row])
    for col in results[model_name]["features"]:
        if col not in df_in:
            df_in[col] = 0
    df_in = df_in[results[model_name]["features"]]

    # 3️⃣ last-mile sanitising (NaNs, ±inf)
    df_in = sanitize(df_in, results[model_name]["features"])

    pred = float(best.predict(df_in)[0])

else:                                                           # ▸ Baseline TEA path
    def true_lcoh(cap, eff, cf, elec_p, wc, cs, tc, sc,
                  cap_ref=500, disc=0.08):
        """Analytical equation for deterministic TEA."""
        crf  = disc * (1 + disc) ** 20 / ((1 + disc) ** 20 - 1)
        kgH2 = cf * 8760 / eff
        return round(
            cap * crf / kgH2 +
            cap * 0.05 * (cap / cap_ref) / kgH2 +
            elec_p * eff +
            wc + cs + tc + sc,
            4,
        )

    pred = true_lcoh(
        cap_dynamic, eff_dynamic, row["CF"], elec,
        row["Water_$kg"], row["CO2_$kg"],
        row["Transport_$kg"], row["Storage_$kg"],
        cap_ref=cap_ref,
    )
# ---------------------------------------------------------------------------

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

# ─────────────────────────────────────────────────────────────────────────
# TAB 2  –  Feature Importance & Trends
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## Feature Importance & Trends")

    # Plotly config – keep camera button, hide logo
    _cfg = {"displaylogo": False}

    # helper → white canvas + ALL text in black
    def _white_canvas(fig):
        # base canvas & global font
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            title_text="",
        )
        # axes
        fig.update_xaxes(color="black", gridcolor="lightgray")
        fig.update_yaxes(color="black", gridcolor="lightgray")
        # colour-bar
        fig.update_coloraxes(
            colorbar_tickcolor="black",
            colorbar_title_font_color="black",
        )
        # any annotations (rare here)
        for ann in fig.layout.annotations or []:
            ann.font.color = "black"
        return fig

    # ───────────── Feature-importance bar (ML mode only) ────────────────
    if ml_mode and "best" in globals():
        fi = getattr(best, "feature_importances_", None)
        if fi is None and hasattr(best, "coef_"):
            fi = np.abs(best.coef_)

        if fi is not None:
            cols = results[model_name]["features"]
            fi_s = pd.Series(np.ravel(fi), index=cols).sort_values()

            fig_bar = px.bar(
                fi_s.values,
                y=fi_s.index,
                orientation="h",
                color=fi_s.values,
                color_continuous_scale="Viridis",
                template="plotly_dark",  # dark bars, we’ll recolour text
            )
            fig_bar.update_coloraxes(colorbar_title=None)
            fig_bar.update_layout(xaxis_title="Importance",
                                  yaxis_title="Feature")
            _white_canvas(fig_bar)                           # ★ text → black
            st.plotly_chart(fig_bar, use_container_width=True, config=_cfg)
        else:
            st.info("This estimator doesn’t expose feature importances.")
    else:
        st.info("Feature importance is available only in **ML Model** mode.")

    # ─────────────────────────────────── SHAP section ────────────────────
    if ml_mode and "best" in globals():
        shap_vals = results[model_name]["shap_values"]
        explainer = results[model_name]["explainer"]

        if shap_vals is not None and explainer is not None:
            import shap, matplotlib.pyplot as plt
            shap_vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
            shap_vals = np.asarray(shap_vals, dtype=np.float64)

            n_plot         = min(len(shap_vals), len(X_train))
            shap_vals_plot = shap_vals[:n_plot]
            X_train_plot   = X_train.iloc[:n_plot]

            pretty_names = [
                n.replace("Year_val_squared", "Year²")
                 .replace("Year_val", "Year")
                 .replace("_$/kW",  " $/kW")
                 .replace("_$/kWh", " $/kWh")
                 .replace("_kWh_kg"," kWh/kg")
                 .replace("_kg",    " (kg)")
                 .replace("_$",     " ($)")
                 .replace("Tech_",  "Tech: ")
                for n in results[model_name]["features"]
            ]

            st.subheader("SHAP summary (global importance)")
            plt.figure()
            shap.summary_plot(shap_vals_plot,
                              X_train_plot,
                              feature_names=pretty_names,
                              show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

            with st.expander("Explain a single training row"):
                idx = st.number_input("Row index inside training set",
                                      0, n_plot - 1, 0)
                force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_vals_plot[idx, :],
                    X_train_plot.iloc[idx, :],
                    feature_names=pretty_names,
                    matplotlib=False,
                )
                st_shap(force_plot, height=300)
        else:
            st.info("SHAP not available for this estimator – showing fallback importances.")

    # ───────────────────────────── Trend heat-map ────────────────────────
    pivot = (
        df.groupby(["Tech", "Year"])["LCOH_$kg"]
          .mean()
          .reset_index()
          .pivot(index="Tech", columns="Year", values="LCOH_$kg")
          .sort_index(axis=1)
    )

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        template="plotly_dark",
        labels={"y": "Tech", "x": "Year", "color": "LCOH ($/kg)"},
    )
    _white_canvas(fig_hm)                               # ★ text → black
    st.plotly_chart(fig_hm, use_container_width=True, config=_cfg)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3  –  Maps
# ─────────────────────────────────────────────────────────────────────────
with tabs[3]:

    # Plotly config – keep camera icon; hide logo
    _cfg = {"displaylogo": False}

    # 1) GeoDF restricted to the lower-48 (full state names)
    LOWER48 = {
        'Alabama','Arizona','Arkansas','California','Colorado','Connecticut',
        'Delaware','Florida','Georgia','Idaho','Illinois','Indiana','Iowa',
        'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts',
        'Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska',
        'Nevada','New Hampshire','New Jersey','New Mexico','New York',
        'North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania',
        'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah',
        'Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming'
    }
    gdf48 = gdf[gdf["State"].isin(LOWER48)].copy()

    # 2) Styling helper → white canvas + WHITE base land/lakes/ocean + black text
    def _white(fig):
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            title_text="",
        )
        fig.update_coloraxes(
            colorbar_tickcolor="black",
            colorbar_title_font_color="black",
        )
        fig.update_geos(
            bgcolor="white",      # page/canvas behind the map
            showland=True, landcolor="white",
            showlakes=True, lakecolor="white",
            showocean=True, oceancolor="white",
        )
        return fig

    # ──────────────────────────── ML-MODEL VIEW ─────────────────────────
    if calculation_mode == "ML Model":

        st.markdown(f"### {state}: LCOH = ${pred:.2f}/kg")

        # (A) single-state highlight map
        map_df = gdf48.copy()
        map_df["Highlight"] = map_df["State"].eq(state)

        fig_sel = (
            px.choropleth(
                map_df, geojson=map_df.geometry, locations=map_df.index,
                color="Highlight",
                color_discrete_map={True: "yellow", False: "#cccccc"},
                hover_name="State", hover_data={},
                template="plotly_dark",
            )
            .update_traces(
                marker_line_color="white",
                hovertemplate="<b>%{hovertext}</b><extra></extra>",
                showlegend=False,
            )
            .update_layout(
                geo=dict(scope="usa", visible=False),  # hides base outline here
                margin=dict(l=0, r=0, t=0, b=0),
            )
        )
        _white(fig_sel)
        st.plotly_chart(fig_sel, use_container_width=True, config=_cfg)

        # (B) optional 48-state choropleth
        if st.checkbox("Show ML-predicted LCOH for all states", value=False):
            df_sub = (
                df.query("Tech == @tech & Year == @year & State in @LOWER48")
                  .set_index("State")
            )
            preds48 = _predict48(df_sub)

            ml_map_df = (
                gdf48.merge(
                    preds48.rename("LCOH ($/kg)"),
                    left_on="State", right_index=True, how="left",
                )
                .dropna(subset=["LCOH ($/kg)"])     # drop any no-data shapes
            )

            fig_ml = (
                px.choropleth(
                    ml_map_df, geojson=ml_map_df.geometry,
                    locations=ml_map_df.index,
                    color="LCOH ($/kg)",
                    color_continuous_scale="Viridis",
                    range_color=(2, 14),
                    hover_name="State",
                    hover_data={"LCOH ($/kg)":":.2f"},
                    template="plotly_dark",
                )
                .update_traces(
                    marker_line_color="white",
                    hovertemplate="<b>%{hovertext}</b>"
                                  "<br>LCOH: %{z:.2f} $/kg<extra></extra>",
                )
                .update_layout(
                    geo=dict(scope="usa"),  # base is shown here
                    margin=dict(l=0, r=0, t=0, b=0),
                )
            )
            _white(fig_ml)  # ← turns land/lakes/ocean white (no black AK/HI/lakes)
            st.plotly_chart(fig_ml, use_container_width=True, config=_cfg)

    # ───────────────────────── BASELINE-TEA VIEW ─────────────────────────
    elif calculation_mode == "Baseline TEA":

        st.subheader(f"{tech} LCOH Map – {year}")

        files = {
            "ALKALINE": ("USAALKALINE.csv", "LCOH_Alk_$kg"),
            "PEM":      ("USAPEM.csv",      "LCOH_PEM_$kg"),
            "SOEC":     ("USASOEC.csv",     "LCOH_SOEC_$kg"),
        }
        csv_file, col_lcoh = files[tech]

        try:
            df_tech = pd.read_csv(csv_file)
            df_year = df_tech[df_tech["Year"] == year]

            if df_year.empty:
                st.warning("No data for that year/tech.")
            else:
                df_year = df_year.rename(columns={col_lcoh: "LCOH ($/kg)"})
                df_year["LCOH ($/kg)"] = df_year["LCOH ($/kg)"].round(2)

                merged = (
                    gdf48.merge(df_year[["State", "LCOH ($/kg)"]], on="State", how="left")
                         .dropna(subset=["LCOH ($/kg)"])
                )

                fig_te = (
                    px.choropleth(
                        merged, geojson=merged.geometry, locations=merged.index,
                        color="LCOH ($/kg)",
                        color_continuous_scale="Viridis",
                        template="plotly_dark",
                    )
                    .update_traces(
                        marker_line_color="white",
                        hovertemplate="<b>%{hovertext}</b>"
                                      "<br>LCOH: %{z:.2f} $/kg<extra></extra>",
                    )
                    .update_layout(
                        geo=dict(scope="usa"),
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                )
                fig_te.update_coloraxes(colorbar_tickformat=".2f")
                _white(fig_te)   # ← whiten base map water/land too
                st.plotly_chart(fig_te, use_container_width=True, config=_cfg)

        except FileNotFoundError:
            st.error(f"Missing data file: `{csv_file}`")

    # ───────────────────────────── default message ───────────────────────
    else:
        st.info("🌐 Map is available only in ML-Model or Baseline-TEA mode.")


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

# ─────────────────────────────────────────────────────────────────────────
# TAB 6  –  Download Center
# ─────────────────────────────────────────────────────────────────────────
with tabs[6]:
    import io, concurrent.futures, hashlib, zipfile, unicodedata
    from datetime import datetime
    from PyPDF2 import PdfReader, PdfWriter
    from fpdf import FPDF

    st.subheader("Download Center")

    # ---------- helper: ASCII-safe text -----------------------------------
    def ascii_safe(txt: str) -> str:
        return (unicodedata
                .normalize("NFKD", txt)
                .encode("latin1", "ignore")
                .decode("latin1")
                .replace("–", "-").replace("—", "-")
                .replace("’", "'").replace("“", '"').replace("”", '"'))

    # ---------- 48-state slice (matches current Tech & Year) --------------
    lower48 = gdf["State"].tolist()
    df_sub = (
        df.query("Tech == @tech & Year == @year & State in @lower48")
          .set_index("State")
    )

    preds48 = _predict48(df_sub)   # ← global cached helper

    # ---------- PDF helpers ----------------------------------------------
    def make_summary_page() -> bytes:
        pdf = FPDF(); pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "LCOH Scenario - Inputs", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(
            0, 10,
            ascii_safe(
                f"Mode: {calculation_mode}\n"
                f"Technology: {tech}\n"
                f"Year: {year}\n"
                f"CAPEX: ${cap_dynamic:,.0f}/kW\n"
                f"Efficiency: {eff_dynamic:.2f} kWh/kg\n"
                f"Electricity price: ${elec:.4f}/kWh"
            )
        )
        return pdf.output(dest="S").encode("latin1")

    def make_state_page(st_name, pred_val) -> bytes:
        pdf = FPDF(); pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, ascii_safe(f"{st_name} - Predicted LCOH"), ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, ascii_safe(f"LCOH: ${pred_val:,.2f}/kg"), ln=True)
        return pdf.output(dest="S").encode("latin1")

    # ---------- Render 48 state pages in parallel -------------------------
    def _build_one(st_name):
        return st_name, make_state_page(st_name, float(preds48[st_name]))

    with st.spinner("Rendering 48 state pages…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            pdf_state_pages = dict(ex.map(_build_one, lower48))

    if len({hashlib.md5(b).digest() for b in pdf_state_pages.values()}) != 48:
        st.error("Duplicate PDF pages detected – export aborted.")
        st.stop()

    summary_pdf = make_summary_page()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- Export options -------------------------------------------
    export_type = st.radio(
        "Choose export type",
        (
            "Scenario PDF",
            "ZIP (48 PDFs)",
            "Compact PDF (Summary + table)",
        ),
        horizontal=True,
    )

    # ---------- 1) Scenario PDF (selected state only) --------------------
    if export_type == "Scenario PDF":
        writer = PdfWriter()
        writer.append(PdfReader(io.BytesIO(summary_pdf)))
        writer.append(PdfReader(io.BytesIO(
            make_state_page(state, float(preds48[state]))
        )))
        buf = io.BytesIO(); writer.write(buf); buf.seek(0)
        st.download_button(
            f"Download {state} scenario",
            buf,
            f"{state}_{tech}_{year}_{ts}.pdf",
            "application/pdf",
        )

    # ---------- 2) ZIP with 48 single-page PDFs --------------------------
    elif export_type == "ZIP (48 PDFs)":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("00_Summary.pdf", summary_pdf)
            for s, b in pdf_state_pages.items():
                z.writestr(f"{s}.pdf", b)
        buf.seek(0)
        st.download_button(
            "Download ZIP with 48 PDFs",
            buf,
            f"AllStates_{tech}_{year}_{ts}.zip",
            "application/zip",
        )

    # ---------- 3) Compact PDF (summary + table) -------------------------
    elif export_type == "Compact PDF (Summary + table)":
        table_pdf = FPDF(); table_pdf.add_page()
        table_pdf.set_font("Arial", "B", 14)
        table_pdf.cell(0, 10, "All States - Predicted LCOH ($/kg)", ln=True)
        table_pdf.ln(4)
        table_pdf.set_font("Arial", "B", 12)
        table_pdf.cell(50, 8, "State", 1)
        table_pdf.cell(40, 8, "LCOH ($/kg)", 1)
        table_pdf.ln()
        table_pdf.set_font("Arial", "", 12)
        for s in lower48:
            table_pdf.cell(50, 8, ascii_safe(s), 1)
            table_pdf.cell(40, 8, f"{preds48[s]:.2f}", 1, ln=1)

        writer = PdfWriter()
        writer.append(PdfReader(io.BytesIO(summary_pdf)))
        writer.append(PdfReader(io.BytesIO(
            table_pdf.output(dest="S").encode("latin1")
        )))
        buf = io.BytesIO(); writer.write(buf); buf.seek(0)
        st.download_button(
            "Download compact PDF",
            buf,
            f"AllStatesCompact_{tech}_{year}_{ts}.pdf",
            "application/pdf",
        )
