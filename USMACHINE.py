# --- Standard libs ---
import os, sys, time, io, zipfile, warnings
from datetime import datetime

# --- Data / ML / viz ---
import pandas as pd, numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import matplotlib.pyplot as plt
from fpdf import FPDF
import shap
import streamlit as st

# --- PROJ fix (must run BEFORE importing geopandas) ---
from pyproj import datadir as _pyproj_datadir

def _ensure_proj_path():
    candidates = []
    # Conda-forge typical locations on Windows
    for root in filter(None, [os.environ.get("CONDA_PREFIX"), sys.prefix]):
        candidates += [
            os.path.join(root, "Library", "share", "proj"),  # conda-forge (Windows)
            os.path.join(root, "share", "proj"),             # other layouts
        ]
    # Fallback: pyproj-bundled data dir (wheels)
    pd_dir = _pyproj_datadir.get_data_dir()
    if pd_dir:
        candidates.append(pd_dir)

    for c in candidates:
        if c and os.path.exists(os.path.join(c, "proj.db")):
            os.environ["PROJ_LIB"] = c
            _pyproj_datadir.set_data_dir(c)
            return c
    return None

_ensure_proj_path()

# Now it's safe to import geopandas (it will find proj.db)
import geopandas as gpd

# Quiet the occasional network warning after we’ve set PROJ paths
warnings.filterwarnings(
    "ignore",
    message="pyproj unable to set PROJ database path",
    category=UserWarning,
    module="pyproj.network",
)

# --- Streamlit page setup (must be the first st.* call) ---
st.set_page_config(layout="centered",
                   page_title="Hydrogen LCOH Dashboard",
                   page_icon="⛽")

# Optional CSS theme
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #F5F5F5; }
    [data-testid="stSidebar"] { background-color: #161A23; color: #F0F0F0; }
    h1,h2,h3 { color: #0CF; }
    </style>
    """,
    unsafe_allow_html=True
)


 # ─── CSV helper ──────────────────────────────────────────────────────────
def read_quoted_csv(path, col_names):
    # First, try to read with the file's own header
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)
    # If the columns don’t match what we expect (or "State" is missing),
    # re-read assuming there is NO header and apply our expected names.
    if len(df.columns) != len(col_names) or "State" not in df.columns:
        df = pd.read_csv(
            path, quotechar='"', header=None, names=col_names, skipinitialspace=True
        )
    # Coerce numeric columns (skip first "State")
    for c in col_names[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ─── SHAP → Streamlit helper ─────────────────────────────────────────────
def st_shap(plot, height=300):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


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


@st.cache_data
def load_data():
    # ---------- 1) TARGET tables ----------
    base = r"C:\Users\Navar\OneDrive\Documentos\USA\TECHNO-ECONOMIC SIMULATION"
    if not os.path.exists(base):
        base = "."

    target_dfs = []
    for tech, csv_name in [
        ("ALKALINE", "USAALKALINE.csv"),
        ("PEM",      "USAPEM.csv"),
        ("SOEC",     "USASOEC.csv"),
    ]:
        fp = os.path.join(base, csv_name)
        if not os.path.exists(fp):
            st.error(f"Missing target file: {fp}")
            st.stop()
        d = pd.read_csv(fp)
        tgt = [c for c in d.columns if "LCOH" in c.upper()][0]  # normalize LCOH col
        d = d.rename(columns={tgt: "LCOH_$kg"})
        d["Tech"] = tech
        target_dfs.append(d)

    lcoh_df = pd.concat(target_dfs, ignore_index=True)

    # ---------- 2) FEATURE tables ----------
    elec = load_electricity_prices()
    # 🔧 align to the column name used everywhere else in the app
    elec = elec.rename(columns={"Electricity_$/kWh_fixed": "Electricity_$/kWh"})

    ghi = read_quoted_csv(
        "GHIUSA.csv",
        ["State", "GHI_min", "GHI_max", "GHI_mean", "GHI_median", "GHI_std"]
    )

    wind = read_quoted_csv(
        "WINDUSA.csv",
        ["State", "Wind_min", "Wind_max", "Wind_mean", "Wind_median", "Wind_std"]
    )

    grid = pd.read_csv("STATEGRIDDISTANCE.csv").rename(columns=lambda c: c.strip())
    if "State" not in grid.columns:
        for c in grid.columns:
            if c.lower().startswith("state"):
                grid = grid.rename(columns={c: "State"})
                break

    water = pd.read_csv("USWATER.csv").rename(columns=lambda c: c.strip())
    if "State" not in water.columns:
        for c in water.columns:
            if c.lower().startswith("state"):
                water = water.rename(columns={c: "State"})
                break
    water = water[[c for c in ["State", "runoff_km3", "runoff_mm"] if c in water.columns]]

    land_raw = pd.read_csv("USLAND.csv").rename(
        columns={"NAME": "State", "solar": "Solar_idx", "wind": "LandWind_idx", "suitability": "Suitability_idx"}
    )
    desired = ["State", "Solar_idx", "LandWind_idx", "Suitability_idx"]
    land = land_raw[[c for c in desired if c in land_raw.columns]]

    infra = (
        pd.read_excel("INFRAUSA.xlsx")
          .rename(columns={"Energy": "Infra_Energy", "Water": "Infra_Water", "Dams": "Infra_Dams"})
          [["State", "Infra_Energy", "Infra_Water", "Infra_Dams"]]
    )

    co2 = (
        pd.read_csv("USACO2.csv")
          .rename(columns={"2023 (million metric tons CO₂)": "CO2_2023_Mt"})
          [["State", "CO2_2023_Mt"]]
    )

    biomass = pd.read_csv("USABIOMASS.csv").rename(columns={"NAME": "State", "AGB_Mg_ha": "AGB_Mg_ha"})

    # ---------- 3) MERGE features ----------
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

    # --- Ensure we have a canonical electricity column: Electricity_$/kWh ---
    def _ensure_elec_col(df_):
        candidates = [
            "Electricity_$/kWh",
            "Electricity_$/kWh_fixed",
            "Electricity_price_$per_kWh",
            "Electricity_price_per_kWh",
            "Electricity_cents_kWh",
        ]
        found = None
        for c in candidates:
            if c in df_.columns:
                found = c
                break
        if found is None:
            # nothing found → create empty; downstream UI will show an error
            df_["Electricity_$/kWh"] = np.nan
            return df_

        if found == "Electricity_cents_kWh":
            df_["Electricity_$/kWh"] = pd.to_numeric(df_[found], errors="coerce") / 100.0
        else:
            df_["Electricity_$/kWh"] = pd.to_numeric(df_[found], errors="coerce")
        return df_

    df = _ensure_elec_col(df)

    # ---------- 5) SHAPES (robust to CRS / PROJ issues) ----------
    geo_url = (
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
        "master/data/geojson/us-states.json"
    )
    gdf = gpd.read_file(geo_url, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.rename(columns={"name": "State"}).query("State in @lower48")

    # ---------- 6) RANKFLIPS ----------
    rankflips = {}
    for tech, filename in [
        ("ALKALINE", "USAALKALINERANKFLIPS.csv"),
        ("PEM",      "USAPEMRANKFLIPS.csv"),
        ("SOEC",     "USASOECRANKFLIPS.csv"),
    ]:
        fp = os.path.join(base, filename)
        try:
            rf = pd.read_csv(fp)
            # clean headers
            rf.columns = [c.strip() for c in rf.columns]

            # 🔁 only if your CSVs use different names — replace LHS with your real headers
            rename_map = {
                "rank_pg_colname":   "rank_pg",
                "rank_low_colname":  "rank_low",
                "rank_med_colname":  "rank_med",
                "rank_high_colname": "rank_high",
            }
            rename_map = {k: v for k, v in rename_map.items() if k in rf.columns}
            if rename_map:
                rf = rf.rename(columns=rename_map)

            # types
            if "Year" in rf.columns:
                rf["Year"] = pd.to_numeric(rf["Year"], errors="coerce").astype("Int64")
            if "State" in rf.columns:
                rf["State"] = rf["State"].astype(str).str.strip()

        except FileNotFoundError:
            st.warning(f"Missing: {filename}")
            rf = pd.DataFrame()

        rankflips[tech] = rf
    return df, gdf, rankflips

@st.cache_resource(show_spinner=False)
def train_models(
    df: pd.DataFrame,
    *,
    learners=("OLS","Ridge","RandomForest","GBRegressor","XGBoost","LightGBM","CatBoost"),
    groups_col: str = "State",
    oof_folds: int = 5,
    do_holdout: bool = True,
    random_state: int = 42,
    show_title: bool = False,
    verbose_debug: bool = False,
    # ── NEW, but backward-compatible defaults ─────────────────────────────
    speed: str = "standard",        # "standard" | "fast" | "thorough"
    shap_mode: str = "off",         # "off" | "selected" | "all"
):
    """
    TRAIN MODELS (drop-in replacement)
    - Same outputs & defaults as before (standard profile, SHAP off).
    - Adds speed profiles: "fast" (~30–50% quicker), "thorough" (slower).
    - Adds SHAP throttle: off/selected(all tree models)/all.
    - Still uses log1p/expm1 target transform; same leakage guards.
    Returns (results_dict, X_float64).
    """
    import time, numpy as np, pandas as pd, shap
    from sklearn.model_selection import KFold, GroupKFold, train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor as TTR
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    # ---------- helpers ----------
    def _ttr(reg): return TTR(regressor=reg, func=np.log1p, inverse_func=np.expm1)

    def _clean_numeric_df(Xin: pd.DataFrame) -> pd.DataFrame:
        if Xin.empty: return Xin.copy()
        S = Xin.astype(str).replace(r"\[|\]", "", regex=True).apply(lambda s: s.str.strip())
        Xnum = (S.apply(pd.to_numeric, errors="coerce")
                  .replace([np.inf, -np.inf], np.nan))
        all_nan = [c for c in Xnum.columns if Xnum[c].isna().all()]
        if all_nan: Xnum = Xnum.drop(columns=all_nan)
        if Xnum.shape[1]: Xnum = Xnum.fillna(Xnum.median(numeric_only=True)).fillna(0.0)
        return Xnum.astype("float64")

    # optional libs
    try:
        from xgboost import XGBRegressor; _HAS_XGB = True
    except Exception:
        _HAS_XGB = False
    try:
        from lightgbm import LGBMRegressor; _HAS_LGBM = True
    except Exception:
        _HAS_LGBM = False
    try:
        from catboost import CatBoostRegressor; _HAS_CAT = True
    except Exception:
        _HAS_CAT = False

    # ---------- 1) Harmonize & de-leak ----------
    df_p = df.copy()
    ren = {}
    if "SEC_kWh_kg" in df_p: ren["SEC_kWh_kg"] = "Efficiency_kWh_kg"
    if "LCOH_plantgate_$kg" in df_p: ren["LCOH_plantgate_$kg"] = "LCOH_$kg"
    df_p = df_p.rename(columns=ren)

    leak_cols = []
    for c in df_p.columns:
        cl = c.lower()
        if c.startswith(("C_capex_","C_opex_","C_repl_","C_elec_")): leak_cols.append(c)
        elif cl.startswith("lcoh_") and c != "LCOH_$kg": leak_cols.append(c)
    if leak_cols: df_p = df_p.drop(columns=leak_cols, errors="ignore")

    # ---------- 2) Build X/y ----------
    if "Year" in df_p: df_p["Year_val"] = df_p["Year"]

    if "LCOH_$kg" not in df_p:
        st.sidebar.error("Missing target column 'LCOH_$kg'."); return {}, None
    y = pd.to_numeric(df_p["LCOH_$kg"], errors="coerce")

    drop_cols = [c for c in ["State","LCOH_$kg","Year"] if c in df_p]
    X = pd.get_dummies(
        df_p.drop(columns=drop_cols),
        columns=[c for c in ["Tech"] if c in df_p],
        drop_first=True
    )
    if "Year_val" in df_p:
        X["Year_val_squared"] = pd.to_numeric(df_p["Year_val"], errors="coerce")**2

    for core in ("Efficiency_kWh_kg","CAPEX_$/kW","Electricity_$/kWh"):
        if core not in X and core in df_p:
            X[core] = pd.to_numeric(df_p[core], errors="coerce")

    X = _clean_numeric_df(X)
    mask = np.isfinite(y); X, y = X.loc[mask], y.loc[mask]

    groups = None
    if (groups_col in df_p) and (groups_col not in drop_cols):
        groups = df_p.loc[mask, groups_col].astype(str).fillna("UNK")

    if len(X) < 5 or X.shape[1] == 0:
        st.sidebar.warning("Too few rows or zero features – ML disabled."); return {}, None

    # ---------- 3) Model zoo with speed profiles ----------
    # Baseline (STANDARD) hyperparameters — your current production settings
    rf_params   = dict(n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5,
                       max_features="sqrt", n_jobs=-1, random_state=random_state)
    gbr_params  = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=random_state)
    xgb_params  = dict(n_estimators=200, max_depth=6, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
                       reg_lambda=1.0, reg_alpha=0.0, n_jobs=-1, verbosity=0,
                       tree_method="hist", max_bin=256, random_state=random_state)
    lgbm_params = dict(n_estimators=300, max_depth=8, num_leaves=31, learning_rate=0.07, subsample=0.9,
                       colsample_bytree=0.9, n_jobs=-1, random_state=random_state)
    cat_params  = dict(iterations=400, depth=8, learning_rate=0.06, l2_leaf_reg=3.0, subsample=0.9, rsm=0.9,
                       loss_function="RMSE", verbose=0, random_state=random_state, thread_count=-1)

    if speed == "fast":
        # ~30–50% faster; empirically negligible accuracy loss
        rf_params.update(n_estimators=120, max_depth=8)
        gbr_params.update(n_estimators=160)
        xgb_params.update(n_estimators=140, max_depth=5, max_bin=128, learning_rate=0.08)
        lgbm_params.update(n_estimators=200, max_depth=6, num_leaves=25, learning_rate=0.08)
        cat_params.update(iterations=250, depth=7, learning_rate=0.08)
    elif speed == "thorough":
        # Slightly better accuracy; slower
        rf_params.update(n_estimators=600)
        gbr_params.update(n_estimators=500, learning_rate=0.04)
        xgb_params.update(n_estimators=500, max_depth=7, learning_rate=0.05)
        lgbm_params.update(n_estimators=600, num_leaves=48, learning_rate=0.05)
        cat_params.update(iterations=700, depth=8, learning_rate=0.05)

    all_defs = {
        "OLS":   Pipeline([("scaler", StandardScaler()), ("ttr", _ttr(LinearRegression()))]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ttr", _ttr(Ridge(alpha=1.0, random_state=random_state)))]),
        "RandomForest": Pipeline([("ttr", _ttr(RandomForestRegressor(**rf_params)))]),
        "GBRegressor":  Pipeline([("ttr", _ttr(GradientBoostingRegressor(**gbr_params)))]),
    }
    if _HAS_XGB:
        all_defs["XGBoost"] = Pipeline([("ttr", _ttr(XGBRegressor(**xgb_params)))])
    if _HAS_LGBM:
        all_defs["LightGBM"] = Pipeline([("ttr", _ttr(LGBMRegressor(**lgbm_params)))])
    if _HAS_CAT:
        all_defs["CatBoost"] = Pipeline([("ttr", _ttr(CatBoostRegressor(**cat_params)))])

    models = {k: v for k, v in all_defs.items() if k in learners}

    # ---------- 4) CV ----------
    if (groups is not None) and (len(set(groups)) > 1):
        splitter = GroupKFold(n_splits=max(2, min(oof_folds, len(set(groups)))))
        splits = list(splitter.split(X, y, groups=groups))
    else:
        splitter = KFold(n_splits=max(2, oof_folds), shuffle=True, random_state=random_state)
        splits = list(splitter.split(X, y))

    # ---------- 5) Fit & evaluate ----------
    results, t0 = {}, time.time()
    prog = st.sidebar.progress(0, text="Training models…") if show_title else None
    n_models = len(models)

    for m_i, (name, proto) in enumerate(models.items(), start=1):
        y_pred_oof = np.zeros(len(X), dtype=float)
        y_true_oof = y.to_numpy(dtype=float, copy=False)

        for tr_idx, va_idx in splits:
            est = clone(proto).fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_hat = np.asarray(est.predict(X.iloc[va_idx]), dtype=float)
            y_pred_oof[va_idx] = np.clip(y_hat, 0.0, None)

        r2_oof   = float(r2_score(y_true_oof, y_pred_oof))
        rmse_oof = float(np.sqrt(mean_squared_error(y_true_oof, y_pred_oof)))
        mae_oof  = float(mean_absolute_error(y_true_oof, y_pred_oof))

        holdout = {}
        if do_holdout:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)
            est_h = clone(proto).fit(X_tr, y_tr)
            y_hat = np.clip(np.asarray(est_h.predict(X_te), dtype=float), 0.0, None)
            holdout = {
                "holdout_r2":   float(r2_score(y_te, y_hat)),
                "holdout_rmse": float(np.sqrt(mean_squared_error(y_te, y_hat))),
                "holdout_mae":  float(mean_absolute_error(y_te, y_hat)),
                "y_true_holdout": y_te.to_numpy(dtype=float, copy=False).tolist(),
                "y_pred_holdout": y_hat.tolist(),
            }

        inference_model = clone(proto).fit(X, y)

        # SHAP (optional; small background to keep it light)
        explainer, shap_vals = None, None
        if shap_mode != "off":
            try:
                bg = X.to_numpy(dtype=np.float64, copy=False)[: min(200, len(X))]
                ttr_obj = inference_model.named_steps.get("ttr", None) if hasattr(inference_model, "named_steps") else None
                raw_est = getattr(ttr_obj, "regressor_", None) if ttr_obj is not None else inference_model
                want = (shap_mode == "all") or (shap_mode == "selected" and name in ("XGBoost","LightGBM","RandomForest","GBRegressor"))
                if want:
                    try:
                        explainer = shap.TreeExplainer(raw_est, feature_perturbation="interventional")
                        shap_raw  = explainer.shap_values(bg)
                    except Exception:
                        explainer = shap.Explainer(raw_est, bg); shap_raw = explainer(bg).values
                    shap_vals = shap_raw[0] if isinstance(shap_raw, list) else shap_raw
            except Exception:
                explainer, shap_vals = None, None

        results[name] = {
            "model_prototype": proto,
            "inference_model": inference_model,
            "model": inference_model,     # legacy alias
            "features": X.columns.tolist(),
            "oof_r2": r2_oof, "oof_rmse": rmse_oof, "oof_mae": mae_oof,
            "y_true_oof": y_true_oof.tolist(), "y_pred_oof": y_pred_oof.tolist(),
            **holdout, "explainer": explainer, "shap_values": shap_vals,
        }

        if prog: prog.progress(m_i / n_models, text=f"Training models… ({m_i}/{n_models})")

    if prog: prog.empty()
    st.sidebar.success(f"✅ Trained {len(results)} models in {time.time()-t0:.1f}s")
    return results, X.astype("float64")




# ⬇️ add this right after the load_data() function definition
df, gdf, rankflips = load_data()

# then you can do:
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
# AFTER
@st.cache_data(show_spinner=False)
def _predict48(df_v: pd.DataFrame, cache_key: str):
    """
    Return a Series of 48 LCOH predictions (index = state names).
    `cache_key` only exists to bust the cache when UI controls change.
    """
    # --- helper: use a column if it exists; otherwise a constant Series ---
    def _series_or_default(dfv, col, default=0.0):
        if col in dfv.columns:
            return pd.to_numeric(dfv[col], errors="coerce").fillna(default)
        return pd.Series(default, index=dfv.index, dtype="float64")

    if ml_mode:
        feats = results[model_name]["features"]

        # build an input frame with exactly the features the model expects
        X = pd.DataFrame(0, index=df_v.index, columns=feats)

        # core features (coerce to numeric, guard NaNs)
        X["CF"]                = pd.to_numeric(df_v["CF"], errors="coerce").fillna(0.0)
        X["CAPEX_$/kW"]        = cap_dynamic
        X["Efficiency_kWh_kg"] = eff_dynamic

        # electricity + adders (safe if columns are missing)
        X["Electricity_$/kWh"] = _series_or_default(df_v, "Electricity_$/kWh", elec)
        X["Water_$kg"]         = _series_or_default(df_v, "Water_$kg", 0.0)
        X["CO2_$kg"]           = _series_or_default(df_v, "CO2_$kg", 0.0)
        X["Transport_$kg"]     = _series_or_default(df_v, "Transport_$kg", 0.0)
        X["Storage_$kg"]       = _series_or_default(df_v, "Storage_$kg", 0.0)

        # time features
        X["Year_val"]          = year
        X["Year_val_squared"]  = year ** 2

        # one-hot tech flag if the model expects it
        tc_flag = f"Tech_{tech}"
        if tc_flag in feats:
            X[tc_flag] = 1

        # final sanitize to numeric float64 and correct column order
        X = sanitize(X, feats)

        yhat = results[model_name]["model"].predict(X)

    else:
        # baseline TEA path (vectorized)
        crf  = 0.08 * (1 + 0.08) ** 20 / ((1 + 0.08) ** 20 - 1)

        CF   = pd.to_numeric(df_v["CF"], errors="coerce").fillna(0.0)
        # guard division by zero
        kgH2 = CF * 8760.0 / max(float(eff_dynamic), 1e-9)

        elec_series = _series_or_default(df_v, "Electricity_$/kWh", elec)
        water_series = _series_or_default(df_v, "Water_$kg", 0.0)
        co2_series   = _series_or_default(df_v, "CO2_$kg", 0.0)
        trans_series = _series_or_default(df_v, "Transport_$kg", 0.0)
        stor_series  = _series_or_default(df_v, "Storage_$kg", 0.0)

        yhat = (
            cap_dynamic * crf / kgH2
          + cap_dynamic * 0.05 * (cap_dynamic / cap_ref) / kgH2
          + elec_series * eff_dynamic
          + water_series + co2_series + trans_series + stor_series
        )

    # return as a Series aligned to state index
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
        "**DOE targets**: sliders load the official DOE trajectory when you switch year/tech, but can still be tweaked afterward.\n"
        "**Fix inputs across years**: once you tweak a slider, its value stays the same across all year/tech selections."
    )
)
apply_doe = (param_source == "DOE targets")

# ► compute the official defaults for this Year/Tech
cap_official  = adjust_capex_by_year(tech, year)
eff_official  = get_efficiency_by_year(tech, year)

# Electricity: try exact row → then State+Year median → else fallback
elec_official = float(row_base.get("Electricity_$/kWh", np.nan))
if not np.isfinite(elec_official):
    try:
        med = pd.to_numeric(
            df.query("State == @state and Year == @year")["Electricity_$/kWh"],
            errors="coerce"
        ).median()
    except Exception:
        med = np.nan
    if np.isfinite(med):
        elec_official = float(med)

# ✅ Streamlit-safe default if still NaN
safe_elec_default = elec_official if np.isfinite(elec_official) else 0.08

# Single warning only (not duplicated)
if not np.isfinite(elec_official):
    st.warning("Electricity price is missing for this State/Year; using the slider value only.")

# ► initialize session_state on first run
if "cap" not in st.session_state:
    st.session_state.cap = cap_official
if "eta" not in st.session_state:
    st.session_state.eta = eff_official
if "elec_price" not in st.session_state:
    st.session_state.elec_price = safe_elec_default
if "last_year" not in st.session_state:
    st.session_state.last_year = year
if "last_tech" not in st.session_state:
    st.session_state.last_tech = tech

# ► if DOE mode and year/tech changed, reset to defaults
if apply_doe and (
    st.session_state.last_year != year
    or st.session_state.last_tech != tech
):
    st.session_state.cap        = cap_official
    st.session_state.eta        = eff_official
    st.session_state.elec_price = safe_elec_default

# ► remember current year/tech
st.session_state.last_year = year
st.session_state.last_tech = tech

# ► dynamic slider ranges include tech bounds, DOE defaults & manual input
slider_cap_min = int(min(cap_min, cap_official, st.session_state.cap))
slider_cap_max = int(max(cap_max, cap_official, st.session_state.cap))

slider_eff_min = float(min(eff_min, eff_official, st.session_state.eta))
slider_eff_max = float(max(eff_max, eff_official, st.session_state.eta))

# ► sliders bound to session_state (define each ONCE)
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

# 👉 Build a cache key that changes whenever the UI state changes
cache_key = (
    f"{calculation_mode}|{model_name}|{tech}|{year}|"
    f"{cap_dynamic:.4f}|{eff_dynamic:.4f}|{elec:.4f}"
)
# --- Predict -----------------------------------------------------------------
def _get_num_from_series(ser, key, default=0.0):
    """Safely pull a numeric value from a pandas.Series by key."""
    val = ser.get(key, np.nan)
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default

row = {
    "CF":                 float(row_base.get("CF", 0.0)) if np.isfinite(float(row_base.get("CF", np.nan))) else 0.0,
    "CAPEX_$/kW":         cap_dynamic,
    "Efficiency_kWh_kg":  eff_dynamic,
    "Electricity_$/kWh":  elec,
    "Water_$kg":          _get_num_from_series(row_base, "Water_$kg", 0.0),
    "CO2_$kg":            _get_num_from_series(row_base, "CO2_$kg", 0.0),
    "Transport_$kg":      _get_num_from_series(row_base, "Transport_$kg", 0.0),
    "Storage_$kg":        _get_num_from_series(row_base, "Storage_$kg", 0.0),
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
tab_labels = [
    "Prediction", "Stats", "Visuals", "Map",
    "State Comparison", "U.S. Summary", "Competitiveness", "Download"
]


# Initialize tab index in session state to preserve selection across reruns
if "active_tab_index" not in st.session_state:
    st.session_state.active_tab_index = 0

# Create tabs
tabs = st.tabs(tab_labels)

# Define this early so PDF generation can access tab state
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

# Utility: set active tab index
def set_active_tab(idx):
    st.session_state.active_tab_index = idx

# -------------------------------------------------------------------------
# 🧪 Tab 0: Prediction
with tabs[0]:

    st.markdown("## Scenario Prediction")

    # --- clamp & guard the already-computed `pred` -----------------------
    try:
        pred_val = float(pred)
    except Exception:
        pred_val = np.nan

    if not np.isfinite(pred_val):
        st.error("Prediction is unavailable (NaN/Inf). Check inputs or switch model.")
        pred_val = 0.0  # keep UI stable

    # hard floor at zero (prevents negatives from any ML path)
    pred_val = max(0.0, pred_val)

    st.metric("Predicted LCOH", f"${pred_val:.2f}/kg")

    # ── Inputs used (compact) ────────────────────────────────────────────
    with st.expander("Inputs Used", expanded=False):
        st.write({
            "Mode": calculation_mode,
            "Technology": tech,
            "Year": int(year),
            "State": state,
            "CAPEX ($/kW)": float(cap_dynamic),
            "Efficiency (kWh/kg)": float(eff_dynamic),
            "Electricity Price ($/kWh)": float(elec),
            "Capacity Factor": float(row.get("CF", 0.0)),
        })

    # ── Optional: deterministic breakdown only in Baseline TEA mode ─────
    if calculation_mode == "Baseline TEA":
        try:
            disc = 0.08
            N = 20
            crf = disc * (1 + disc) ** N / ((1 + disc) ** N - 1)

            cf = float(row.get("CF", 0.0))
            eff = max(1e-9, float(eff_dynamic))             # guard zero/neg
            kgH2 = cf * 8760.0 / eff

            cap = float(cap_dynamic)
            elec_p = float(elec)
            wc  = float(row.get("Water_$kg", 0.0))
            cs  = float(row.get("CO2_$kg", 0.0))
            tc  = float(row.get("Transport_$kg", 0.0))
            sc  = float(row.get("Storage_$kg", 0.0))

            if kgH2 <= 0 or not np.isfinite(kgH2):
                st.warning("Capacity factor / efficiency led to zero production; breakdown skipped.")
            else:
                capex_term = cap * crf / kgH2
                om_term    = cap * 0.05 * (cap / cap_ref) / kgH2
                elec_term  = elec_p * eff

                breakdown = {
                    "CAPEX term ($/kg)":        round(capex_term, 4),
                    "O&M scaling term ($/kg)":  round(om_term, 4),
                    "Electricity term ($/kg)":  round(elec_term, 4),
                    "Water ($/kg)":             round(wc, 4),
                    "CO₂ ($/kg)":               round(cs, 4),
                    "Transport ($/kg)":         round(tc, 4),
                    "Storage ($/kg)":           round(sc, 4),
                }
                # sum with safety floor (matches the metric value policy)
                lcoh_sum = sum(breakdown.values())
                lcoh_sum = max(0.0, float(lcoh_sum))

                st.markdown("**Cost Breakdown (Baseline TEA)**")
                st.write(breakdown)
                st.caption(f"Sum = ${lcoh_sum:.2f}/kg (clamped ≥ 0)")
        except Exception as e:
            st.warning(f"Could not compute TEA breakdown ({e}).")


# ─────────────────────────────────────────────────────────────────────────
# TAB 1 – Model Stats (OOF by default + optional Hold-out if available)
# ─────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## Model Stats")

    if ml_mode and isinstance(results, dict) and len(results) > 0:
        # If for some reason model_name isn't set or missing, pick the first available
        if (model_name is None) or (model_name not in results):
            model_name = next(iter(results.keys()))

        # Build a tidy metrics table from the results dict
        rows = []
        for k, v in results.items():
            row = {
                "Model": k,
                "R² (OOF)":  round(float(v.get("oof_r2", np.nan)), 3) if v.get("oof_r2", None) is not None else np.nan,
                "RMSE (OOF)": round(float(v.get("oof_rmse", np.nan)), 3) if v.get("oof_rmse", None) is not None else np.nan,
                "MAE (OOF)":  round(float(v.get("oof_mae", np.nan)), 3) if v.get("oof_mae", None) is not None else np.nan,
                # Hold-out columns (may be absent)
                "R² (Hold-out)":   round(float(v.get("holdout_r2",  np.nan)), 3) if v.get("holdout_r2",  None) is not None else np.nan,
                "RMSE (Hold-out)": round(float(v.get("holdout_rmse", np.nan)), 3) if v.get("holdout_rmse", None) is not None else np.nan,
                "MAE (Hold-out)":  round(float(v.get("holdout_mae",  np.nan)), 3) if v.get("holdout_mae",  None) is not None else np.nan,
            }
            rows.append(row)

        stats_df = pd.DataFrame(rows).set_index("Model")

        # Show full table
        st.dataframe(stats_df, use_container_width=True)

        # Highlight the currently selected model’s metrics
        sel = stats_df.loc[model_name]
        oof_str = f"R²={sel['R² (OOF)']}, RMSE={sel['RMSE (OOF)']}, MAE={sel['MAE (OOF)']}"
        if np.isfinite(sel["R² (Hold-out)"]):
            holdout_str = f" | Hold-out: R²={sel['R² (Hold-out)']}, RMSE={sel['RMSE (Hold-out)']}, MAE={sel['MAE (Hold-out)']}"
        else:
            holdout_str = " | Hold-out: n/a"

        st.success(f"**{model_name}** — {oof_str}{holdout_str}")

        # Small tip to remind what “OOF” means
        st.caption("OOF = out-of-fold predictions from cross-validation (group-aware when possible).")

    else:
        st.info("Model stats only apply in **ML Model** mode (after training at least one model).")


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 – Feature Importance, SHAP & Trends (clean styling + no baseline noise)
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## Feature Importance & Trends")
    _cfg = {"displaylogo": False}

    def light_plotly(fig, showgrid=False):
        fig.update_layout(template="none", paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(color="black"), title_text="")
        fig.update_xaxes(showgrid=showgrid, gridcolor="lightgray", zeroline=False,
                         tickfont=dict(color="black"), title_font=dict(color="black"), color="black")
        fig.update_yaxes(showgrid=showgrid, gridcolor="lightgray", zeroline=False,
                         tickfont=dict(color="black"), title_font=dict(color="black"), color="black")
        fig.update_coloraxes(colorbar=dict(tickfont=dict(color="black"),
                                           title=dict(font=dict(color="black"))))
        if fig.layout.annotations:
            for a in fig.layout.annotations:
                a.font.color = "black"
        return fig

    def pretty_label(name: str) -> str:
        repl = (name.replace("Year_val_squared", "Year²")
                     .replace("Year_val", "Year")
                     .replace("_kWh_kg", " kWh/kg")
                     .replace("_$/kWh", " $/kWh")
                     .replace("_$/kW", " $/kW")
                     .replace("_$kg", " $/kg")
                     .replace("Tech_", "Tech: "))
        specials = {
            "AGB_Mg_ha": "AGB (Mg/ha)",
            "CO2_2023_Mt": "CO2 2023 (Mt)",
            "Electricity_$/kWh_fixed": "Electricity $/kWh (fixed)",
        }
        repl = specials.get(repl, repl)
        return repl.replace("_", " ")

    if ml_mode and results and (model_name in results):

        # ---------- get the trained model & features ----------
        inf_model = results[model_name]["inference_model"]           # ✅ fitted Pipeline or Estimator
        # If it's a Pipeline, the final estimator is under "est"; otherwise use the object directly
        final_est = getattr(inf_model, "named_steps", {}).get("est", inf_model)
        feats = results[model_name]["features"]

        # ===== Feature Importance (works for trees or linear) =====
        fi = getattr(final_est, "feature_importances_", None)
        if fi is None and hasattr(final_est, "coef_"):
            fi = np.ravel(np.abs(getattr(final_est, "coef_")))
        if fi is not None:
            fi_s = pd.Series(np.ravel(fi), index=feats).fillna(0.0).sort_values(ascending=False)

            view = st.radio("Feature list", ["Top 10", "Top (95% cumulative)", "All"],
                            horizontal=True, key="fi_view")
            if view == "Top 10":
                fi_show = fi_s.head(10).sort_values()
            elif view == "Top (95% cumulative)":
                cum = fi_s.cumsum() / (fi_s.sum() + 1e-12)
                k = int(np.searchsorted(cum.values, 0.95)) + 1
                fi_show = fi_s.head(max(1, k)).sort_values()
            else:
                fi_show = fi_s[fi_s > 0].sort_values()

            y_labels = [pretty_label(n) for n in fi_show.index]
            fig_bar = px.bar(x=fi_show.values, y=y_labels, orientation="h",
                             color=fi_show.values, color_continuous_scale="Viridis",
                             labels={"x": "Importance", "y": "Feature"}, template="none")
            fig_bar.update_layout(coloraxis_showscale=False, yaxis_title="Feature",
                                  margin=dict(l=220, r=10, t=10, b=40), width=720, height=520)
            fig_bar.update_yaxes(automargin=True, title_standoff=8, tickfont=dict(size=11))
            light_plotly(fig_bar, showgrid=False)
            st.plotly_chart(fig_bar, use_container_width=False, config=_cfg)

        # ===== SHAP (precomputed during training, if any) =====
        shap_vals = results[model_name].get("shap_values")
        explainer = results[model_name].get("explainer")

        if (shap_vals is not None) and (explainer is not None) and (X_train is not None) and (len(X_train) > 0):
            import shap as _shap
            shap_vals_arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
            shap_vals_arr = np.asarray(shap_vals_arr, dtype=np.float64)

            n_plot = min(len(shap_vals_arr), len(X_train))
            shap_vals_plot = shap_vals_arr[:n_plot]
            X_train_plot   = X_train.iloc[:n_plot][feats]

            pretty_names = [pretty_label(n) for n in feats]

            mean_abs = np.mean(np.abs(shap_vals_plot), axis=0)
            order = np.argsort(mean_abs)[::-1]

            view_s = st.radio("SHAP feature list", ["Top 10", "Top (95% cumulative)", "All"],
                              horizontal=True, key="shap_view")
            if view_s == "Top 10":
                idx_sel = order[:10]
            elif view_s == "Top (95% cumulative)":
                cum = np.cumsum(mean_abs[order]) / (mean_abs.sum() + 1e-12)
                k = int(np.searchsorted(cum, 0.95)) + 1
                idx_sel = order[:max(1, k)]
            else:
                idx_sel = order

            cols_sel  = [feats[i] for i in idx_sel]
            names_sel = [pretty_names[i] for i in idx_sel]
            shap_sel  = shap_vals_plot[:, idx_sel]
            X_sel     = X_train_plot[cols_sel]

            st.subheader("SHAP summary (global importance)")
            try:
                import matplotlib.pyplot as _plt
                fig, ax = _plt.subplots(figsize=(8, 6), dpi=150)
                _shap.summary_plot(
                    np.asarray(shap_sel, dtype=np.float64),
                    np.asarray(X_sel.values, dtype=np.float64),
                    feature_names=names_sel,
                    show=False, plot_size=(8, 6)
                )
                st.pyplot(fig, clear_figure=True)
                _plt.close(fig)
            except Exception as e:
                st.warning(f"SHAP summary plot unavailable ({e}). Showing mean |SHAP| instead.")
                mean_abs_sel = np.mean(np.abs(np.asarray(shap_sel, dtype=np.float64)), axis=0)
                df_bar = (pd.DataFrame({"Feature": names_sel, "Mean |SHAP|": mean_abs_sel})
                            .sort_values("Mean |SHAP|", ascending=True))
                fig_bar2 = px.bar(df_bar, x="Mean |SHAP|", y="Feature", orientation="h",
                                  template="none", labels={"Mean |SHAP|": "Importance", "Feature": ""})
                fig_bar2.update_layout(margin=dict(l=220, r=10, t=10, b=40))
                light_plotly(fig_bar2, showgrid=False)
                st.plotly_chart(fig_bar2, use_container_width=True, config=_cfg)

        # ===== Calibration & Error Diagnostics =====
        st.markdown("### Model Calibration & Error Diagnostics")
        # We have OOF stored; optionally let user compute random hold-out on demand.
        choice = st.radio("Prediction source",
                          ["Stored 5-fold OOF (no leakage)", "Random hold-out (80/20)"],
                          horizontal=True, key="diag_src")

        if choice.startswith("Stored"):
            y_true = np.array(results[model_name]["y_true_oof"], dtype=float)
            y_pred = np.array(results[model_name]["y_pred_oof"], dtype=float)
            note = "stored 5-fold OOF"
        else:
            # Quick ad-hoc hold-out using the fully-fitted prototype
            from sklearn.model_selection import train_test_split
            proto = results[model_name]["model_prototype"]
            X_full = X_train[feats] if set(feats).issubset(X_train.columns) else X_train
            y_full = df.loc[X_full.index, "LCOH_$kg"].to_numpy(dtype=float, copy=False)
            X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
            est = clone(proto).fit(X_tr, y_tr)
            y_true, y_pred = y_te, est.predict(X_te)
            note = "random 80/20 hold-out"

        # parity
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
        slope = float(lr.coef_[0]); intercept = float(lr.intercept_)
        r2_parity = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)

        df_par = pd.DataFrame({"Measured": y_true, "Predicted": y_pred})
        lim_min = float(min(df_par.min()) * 0.95)
        lim_max = float(max(df_par.max()) * 1.05)

        fig_par = px.scatter(df_par, x="Measured", y="Predicted", opacity=0.7, template="none")
        fig_par.add_shape(type="line", x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max,
                          line=dict(dash="dash", color="black"), xref="x", yref="y")
        fig_par.add_shape(type="line", x0=lim_min, y0=slope*lim_min+intercept,
                          x1=lim_max, y1=slope*lim_max+intercept,
                          line=dict(color="firebrick"), xref="x", yref="y")
        fig_par.update_layout(
            title=f"Parity plot ({model_name}, {note}) — ŷ = {intercept:.3f} + {slope:.3f}·y, R²={r2_parity:.3f}",
            xaxis_title="Measured LCOH ($/kg)", yaxis_title="Predicted LCOH ($/kg)",
            margin=dict(l=40, r=10, t=10, b=40)
        )
        light_plotly(fig_par, showgrid=False)
        fig_par.update_xaxes(range=[lim_min, lim_max]); fig_par.update_yaxes(range=[lim_min, lim_max])
        st.plotly_chart(fig_par, use_container_width=True, config=_cfg)

        resid = y_pred - y_true
        df_res = pd.DataFrame({"Predicted": y_pred, "Residual": resid})
        fig_res = px.scatter(df_res, x="Predicted", y="Residual", opacity=0.7, template="none")
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        fig_res.update_layout(title="Residuals vs Predicted",
                              xaxis_title="Predicted LCOH ($/kg)", yaxis_title="Residual ($/kg)",
                              margin=dict(l=40, r=10, t=10, b=40))
        light_plotly(fig_res, showgrid=False)
        st.plotly_chart(fig_res, use_container_width=True, config=_cfg)

        with st.expander("Residual QQ plot (optional)"):
            try:
                import matplotlib.pyplot as plt
                from scipy import stats as sstats
                plt.figure()
                sstats.probplot(resid, dist="norm", plot=plt)
                plt.title("QQ plot of residuals")
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception:
                pass

    else:
        st.info("Model visuals are available in **ML Model** mode.")


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
            colorbar_tickfont=dict(color="black"),
        )
        fig.update_geos(
            bgcolor="white",
            showland=True,  landcolor="white",
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
                geo=dict(scope="usa", visible=False),
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

            if df_sub.empty:
                st.warning("No data available for this Tech/Year across the lower-48.")
            else:
                # ✅ use cache_key so cache refreshes when UI changes
                preds48 = _predict48(df_sub, cache_key)

                ml_map_df = (
                    gdf48.merge(
                        preds48.rename("LCOH ($/kg)"),
                        left_on="State", right_index=True, how="left",
                    )
                    .dropna(subset=["LCOH ($/kg)"])
                )

                if ml_map_df.empty:
                    st.warning("Predictions unavailable for the selected configuration.")
                else:
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
                            geo=dict(scope="usa"),
                            margin=dict(l=0, r=0, t=0, b=0),
                        )
                    )
                    _white(fig_ml)
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

                if merged.empty:
                    st.warning("No mappable rows after merge.")
                else:
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
                    _white(fig_te)
                    st.plotly_chart(fig_te, use_container_width=True, config=_cfg)

        except FileNotFoundError:
            st.error(f"Missing data file: `{csv_file}`")

    # ───────────────────────────── default message ───────────────────────
    else:
        st.info("🌐 Map is available only in ML-Model or Baseline-TEA mode.")



# ─────────────────────────────────────────────────────────────────────────
# TAB 4 – State-wise LCOH Summary (Simulated Variations)
# ─────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("State-wise LCOH Summary (Simulated Variations)")

    from itertools import product

    variations = []

    # 25 independent variations around current sliders
    cap_values = np.array([cap_dynamic * f for f in [0.90, 0.95, 1.00, 1.05, 1.10]], dtype=float)
    eff_values = np.array([eff_dynamic * f for f in [0.95, 0.975, 1.00, 1.025, 1.05]], dtype=float)

    # --- Safe access to model + features (if in ML mode) ---
    features = []
    _best = None
    if ml_mode and isinstance(results, dict) and len(results) > 0:
        if (model_name is None) or (model_name not in results):
            model_name = next(iter(results.keys()))
        rslot = results[model_name]
        # accept either key ("model" legacy or "inference_model")
        _best = rslot.get("model", rslot.get("inference_model", None))
        features = rslot.get("features", [])

    # Helpers
    def _n(x, d=np.nan):
        try:
            v = float(x)
            return v if np.isfinite(v) else d
        except Exception:
            return d

    def _mean_or_default(df_, col, default=0.0):
        """Mean of numeric column if present, else default."""
        if col in df_.columns:
            val = pd.to_numeric(df_[col], errors="coerce").mean()
            return float(val) if np.isfinite(val) else float(default)
        return float(default)

    def _elec_fallback(subset_df):
        """
        Try exact values → mean(State,Year,Tech) → session slider → global slider.
        Guarantees a finite float result.
        """
        if "Electricity_$/kWh" in subset_df.columns:
            s = pd.to_numeric(subset_df["Electricity_$/kWh"], errors="coerce")
            if len(s) and np.isfinite(s.iloc[0]):
                return float(s.iloc[0])

        m = _mean_or_default(subset_df, "Electricity_$/kWh", np.nan)
        if np.isfinite(m):
            return m

        sess = _n(st.session_state.get("elec_price", np.nan))
        if np.isfinite(sess):
            return sess

        return _n(elec, 0.10)

    if "Electricity_$/kWh" not in df.columns:
        st.warning("Electricity_$/kWh column is missing; using slider fallback for all states.")

    for st_name in sorted(df.State.unique()):
        subset = df.query("State == @st_name & Year == @year & Tech == @tech")
        if subset.empty:
            continue

        # means are safer than single iloc in case of duplicates
        cf_val   = _mean_or_default(subset, "CF", 0.0)
        elec_val = _elec_fallback(subset)
        wc       = _mean_or_default(subset, "Water_$kg", 0.0)
        cs       = _mean_or_default(subset, "CO2_$kg", 0.0)
        tc       = _mean_or_default(subset, "Transport_$kg", 0.0)
        sc       = _mean_or_default(subset, "Storage_$kg", 0.0)

        # guard rails
        if not np.isfinite(cf_val) or cf_val <= 0:
            continue

        for c, e in product(cap_values, eff_values):
            e = float(e)
            if not np.isfinite(e) or e <= 0:
                continue

            row_sim = {
                "State": st_name,
                "CF": cf_val,
                "CAPEX_$/kW": float(c),
                "Efficiency_kWh_kg": e,
                "Electricity_$/kWh": float(elec_val),
                "Water_$kg": wc,
                "CO2_$kg": cs,
                "Transport_$kg": tc,
                "Storage_$kg": sc
            }

            if ml_mode and _best is not None and features:
                # add temporal + tech one-hots expected by the model
                row_sim.update({"Year_val": year, "Year_val_squared": year ** 2})
                for col in features:
                    if col.startswith("Tech_"):
                        row_sim[col] = 1 if col == f"Tech_{tech}" else 0

                # exact feature order + numeric dtype + no NaNs/Infs
                df_row = pd.DataFrame([row_sim])
                for col in features:
                    if col not in df_row:
                        df_row[col] = 0
                df_row = (df_row[features]
                          .apply(pd.to_numeric, errors="coerce")
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0.0)
                          .astype("float64"))
                try:
                    pred_val = float(_best.predict(df_row)[0])
                    # clip to non-negative and a sane upper bound for visual stability
                    row_sim["LCOH ($/kg)"] = float(np.clip(pred_val, 0.0, 60.0))
                except Exception:
                    # if the selected model can't score this row, skip gracefully
                    continue
            else:
                # deterministic TEA (non-negative by construction)
                disc = 0.08; N = 20
                crf  = disc * (1 + disc) ** N / ((1 + disc) ** N - 1)
                kgH2 = cf_val * 8760.0 / e
                if not np.isfinite(kgH2) or kgH2 <= 0:
                    continue
                capex_term = (c * crf) / kgH2
                om_term    = (c * 0.05 * (c / cap_ref)) / kgH2
                elec_term  = elec_val * e
                val = capex_term + om_term + elec_term + wc + cs + tc + sc
                row_sim["LCOH ($/kg)"] = max(0.0, float(val))

            if np.isfinite(row_sim["LCOH ($/kg)"]):
                variations.append(row_sim)

    if not variations:
        st.warning("No data available for the selected state/tech/year (or inputs led to invalid combinations).")
    else:
        df_var = pd.DataFrame(variations)

        summary_stats = (
            df_var.groupby("State")["LCOH ($/kg)"]
                 .agg(mean="mean", min="min", max="max",
                      std=lambda s: s.std(ddof=1))
                 .round(2)
        )

        # If std degenerates (can happen with tiny sample sizes), recompute explicitly
        if ("std" in summary_stats.columns) and (summary_stats["std"].nunique() <= 1):
            summary_stats["std"] = (
                df_var.groupby("State")["LCOH ($/kg)"]
                     .apply(lambda s: s.std(ddof=1)).round(3)
            )

        st.dataframe(
            summary_stats.style.format(precision=2).background_gradient(cmap="viridis"),
            use_container_width=True
        )



# ─────────────────────────────────────────────────────────────────────────
# TAB 5  –  U.S. Summary (National LCOH Overview)
# ─────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🇺🇸 National LCOH Overview")

    national_df = df[df.Tech == tech].copy()

    if national_df.empty:
        st.warning(f"No data available for {tech}.")
    else:
        # Build national_trend depending on mode
        if calculation_mode == "ML Model":
            national_trend = (
                national_df
                .groupby("Year", as_index=False)["LCOH_$kg"]
                .mean()
                .rename(columns={"LCOH_$kg": "Average LCOH ($/kg)"})
            )
        else:
            # Deterministic baseline with guards for DOE lookups and zero/NaN divisions
            years = sorted(national_df["Year"].dropna().unique())
            rows = []
            for y in years:
                subset = national_df[national_df["Year"] == y]
                if subset.empty:
                    continue

                cap_val = adjust_capex_by_year(tech, y) or cap_ref
                eff_val = get_efficiency_by_year(tech, y) or float(st.session_state.get("eta", 45.0))
                elec_val = float(subset["Electricity_$/kWh"].mean())
                cf_val   = float(subset["CF"].mean())

                wc = float(subset["Water_$kg"].mean())
                cs = float(subset["CO2_$kg"].mean())
                tc = float(subset["Transport_$kg"].mean())
                sc = float(subset["Storage_$kg"].mean())

                crf  = 0.08 * (1 + 0.08) ** 20 / ((1 + 0.08) ** 20 - 1)
                kgH2 = (cf_val * 8760.0) / eff_val if eff_val and cf_val else np.nan
                if not np.isfinite(kgH2):
                    lcoh = np.nan
                else:
                    lcoh = (
                        cap_val * crf / kgH2 +
                        cap_val * 0.05 * (cap_val / cap_ref) / kgH2 +
                        elec_val * eff_val +
                        wc + cs + tc + sc
                    )

                rows.append({"Year": int(y), "Average LCOH ($/kg)": np.round(lcoh, 2) if np.isfinite(lcoh) else np.nan})

            national_trend = pd.DataFrame(rows)

        # Clean & sort
        national_trend = (
            national_trend
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["Average LCOH ($/kg)"])
            .sort_values("Year")
        )
        if "Year" in national_trend.columns:
            national_trend["Year"] = national_trend["Year"].astype(int)

        # Guard: nothing to show
        if national_trend.empty:
            st.warning("No national trend data available for this configuration.")
        else:
            # Plot (white bg, black text, no title)
            fig_nat = px.line(
                national_trend,
                x="Year",
                y="Average LCOH ($/kg)",
                markers=True,
                template="none"
            )
            fig_nat.update_layout(
                title_text="",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                xaxis_title="Year",
                yaxis_title="LCOH ($/kg)",
                margin=dict(l=40, r=10, t=10, b=40),
            )
            fig_nat.update_xaxes(showgrid=False, zeroline=False, color="black",
                                 title_font=dict(color="black"), tickfont=dict(color="black"))
            fig_nat.update_yaxes(showgrid=False, zeroline=False, color="black",
                                 title_font=dict(color="black"), tickfont=dict(color="black"))

            fn_base = f"National_LCOH_{tech}_{int(national_trend['Year'].min())}-{int(national_trend['Year'].max())}"
            st.plotly_chart(
                fig_nat,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["toImage"],
                    "toImageButtonOptions": {"format": "png", "filename": fn_base, "scale": 3},
                },
            )

            # Table below the chart
            st.dataframe(national_trend, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 6  –  State Competitiveness & Ranking Dynamics
# ─────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("💹 State Competitiveness & Ranking Dynamics")

    # Guard: ensure rankflips exists and has this tech
    if isinstance(rankflips, dict) and (tech in rankflips) and (not rankflips[tech].empty):
        rf_df = rankflips[tech].copy()

        # Basic column normalization (robust to case or stray spaces)
        rf_df.columns = [c.strip() for c in rf_df.columns]

        # Filter by year selected in sidebar
        rf_year = rf_df[rf_df["Year"] == year]
        if not rf_year.empty and (state in set(rf_year["State"])):
            current_state = rf_year[rf_year["State"] == state].iloc[0]

            # ── Key metrics row ──
            st.markdown("### Your State Ranking")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Plant-Gate", f"#{int(current_state['rank_pg'])} of 48")
            with c2:
                st.metric("Low Cost",   f"#{int(current_state['rank_low'])} of 48")
            with c3:
                st.metric("Medium",     f"#{int(current_state['rank_med'])} of 48")
            with c4:
                st.metric("High Cost",  f"#{int(current_state['rank_high'])} of 48")

            # ── Stability badges ──
            st.markdown("### Ranking Stability")
            flip_status = {
                "Low":   "🔴 Rank shifted" if bool(current_state.get("flip_low", False))  else "🟢 Stable",
                "Medium":"🔴 Rank shifted" if bool(current_state.get("flip_med", False))  else "🟢 Stable",
                "High":  "🔴 Rank shifted" if bool(current_state.get("flip_high", False)) else "🟢 Stable",
            }
            cc = st.columns(3)
            for col, (name, status) in zip(cc, flip_status.items()):
                with col:
                    st.write(f"**{name}**: {status}")

            # ── Full ranking table ──
            st.markdown("### All 48 States Ranked by Plant-Gate LCOH")
            rank_table = rf_year[["State", "rank_pg", "rank_low", "rank_med", "rank_high"]].copy()
            rank_table.columns = ["State", "🏭 Plant-Gate", "💚 Low", "📊 Medium", "⚠️ High"]
            rank_table = rank_table.sort_values("🏭 Plant-Gate").reset_index(drop=True)
            rank_table.index = rank_table.index + 1
            st.dataframe(rank_table, use_container_width=True)

            # ── Trend over time for the chosen state ──
            if len(rf_df) > 1:
                st.markdown("### Ranking Trend Over Time")
                trend = rf_df[rf_df["State"] == state][["Year", "rank_pg"]].copy()
                if not trend.empty:
                    fig_trend = px.line(
                        trend, x="Year", y="rank_pg",
                        markers=True, title=f"{state} Ranking Over Time (Plant-Gate)",
                        labels={"rank_pg": "Rank (1=Best)", "Year": "Year"},
                        template="none"
                    )
                    fig_trend.update_yaxes(autorange="reversed")
                    fig_trend.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font=dict(color="black"),
                        margin=dict(l=40, r=10, t=40, b=40)
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning(f"No ranking data for {tech} in {year} (or state not found).")
    else:
        st.info("🔍 Ranking data not available for this technology.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 7  –  Download Center
# ─────────────────────────────────────────────────────────────────────────
with tabs[7]:
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

    # ---------- build/cache predictions for the 48-state slice ------------
    # Use only states present for current Tech/Year (avoids KeyErrors)
    df_sub = (
        df.query("Tech == @tech & Year == @year")
          .drop_duplicates(subset=["State"])
          .set_index("State")
    )

    if df_sub.empty:
        st.warning("No data available for the selected Technology and Year.")
        st.stop()

    # Ensure we only keep states that exist in the map GeoDataFrame
    # (gdf may already be lower-48 filtered in Tab 3; handle both cases)
    valid_states = set(gdf["State"]) if "State" in gdf.columns else set(df_sub.index)
    df_sub = df_sub.loc[df_sub.index.intersection(valid_states)].sort_index()

    if df_sub.empty:
        st.warning("No mappable states for the selected configuration.")
        st.stop()

    # ✅ use cache_key so cache refreshes when UI changes
    try:
        preds48 = _predict48(df_sub, cache_key)   # series indexed by state names
    except Exception as e:
        st.error(f"Could not compute predictions for export: {e}")
        st.stop()

    # Drop any states without a numeric prediction
    preds48 = preds48[preds48.replace([np.inf, -np.inf], np.nan).notna()]
    if preds48.empty:
        st.warning("Predictions are empty for the selected configuration.")
        st.stop()

    lower48 = preds48.index.tolist()  # only states we actually have predictions for

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

    def make_state_page(st_name: str, pred_val: float) -> bytes:
        pdf = FPDF(); pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, ascii_safe(f"{st_name} - Predicted LCOH"), ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, ascii_safe(f"LCOH: ${pred_val:,.2f}/kg"), ln=True)
        return pdf.output(dest="S").encode("latin1")

    # ---------- Render single-page PDFs in parallel -----------------------
    def _build_one(st_name):
        try:
            return st_name, make_state_page(st_name, float(preds48.loc[st_name]))
        except Exception:
            # If a state slipped through without a prediction, skip it gracefully
            return st_name, None

    with st.spinner("Rendering state pages…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            pairs = list(ex.map(_build_one, lower48))

    # Keep only successful pages
    pdf_state_pages = {s: b for (s, b) in pairs if b is not None}

    if not pdf_state_pages:
        st.error("No state pages could be rendered for export.")
        st.stop()

    # Optional: sanity check (uniqueness of pages)
    # If lengths mismatch, warn rather than abort hard
    if len(pdf_state_pages) != len(lower48):
        st.warning("Some states were skipped due to missing predictions during export.")

    summary_pdf = make_summary_page()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- Export options -------------------------------------------
    export_type = st.radio(
        "Choose export type",
        (
            "Scenario PDF (selected state)",
            "ZIP (all available states)",
            "Compact PDF (Summary + table)",
        ),
        horizontal=True,
    )

    # ---------- 1) Scenario PDF (selected state only) --------------------
    if export_type.startswith("Scenario PDF"):
        if state not in preds48.index:
            st.warning(f"{state} has no prediction in the current configuration.")
        else:
            writer = PdfWriter()
            writer.append(PdfReader(io.BytesIO(summary_pdf)))
            writer.append(PdfReader(io.BytesIO(
                make_state_page(state, float(preds48.loc[state]))
            )))
            buf = io.BytesIO(); writer.write(buf); buf.seek(0)
            st.download_button(
                f"Download {state} scenario",
                buf,
                f"{state}_{tech}_{year}_{ts}.pdf",
                "application/pdf",
            )

    # ---------- 2) ZIP with single-page PDFs for all available states ----
    elif export_type.startswith("ZIP"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("00_Summary.pdf", summary_pdf)
            for s in sorted(pdf_state_pages):
                z.writestr(f"{s}.pdf", pdf_state_pages[s])
        buf.seek(0)
        st.download_button(
            "Download ZIP with state PDFs",
            buf,
            f"AllStates_{tech}_{year}_{ts}.zip",
            "application/zip",
        )

    # ---------- 3) Compact PDF (summary + table) -------------------------
    else:
        table_pdf = FPDF(); table_pdf.add_page()
        table_pdf.set_font("Arial", "B", 14)
        table_pdf.cell(0, 10, "All States - Predicted LCOH ($/kg)", ln=True)
        table_pdf.ln(4)
        table_pdf.set_font("Arial", "B", 12)
        table_pdf.cell(50, 8, "State", 1)
        table_pdf.cell(40, 8, "LCOH ($/kg)", 1)
        table_pdf.ln()
        table_pdf.set_font("Arial", "", 12)
        for s in sorted(preds48.index):
            table_pdf.cell(50, 8, ascii_safe(s), 1)
            table_pdf.cell(40, 8, f"{float(preds48.loc[s]):.2f}", 1, ln=1)

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
