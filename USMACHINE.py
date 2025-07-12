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
        df.drop(columns=["State", "LCOH_$kg", "Electricity_$/kWh_fixed", "Year"]),
        columns=["Tech"],
        drop_first=True
    )
    X["Year_val"] = df["Year_val"]
    X["Year_val_squared"] = X["Year_val"] ** 2  # ✅ curvature feature for extrapolation

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

    st.sidebar.write(f"Training finished in {time.time() - t0:.1f}s")
    return results, X_tr

# Train
results, X_train = train_models(df)
if not results:
    st.error("Training skipped: not enough data!")
    st.stop()

# --- 🔧 Scenario Simulator Setup ---
st.sidebar.header("🔧 Scenario Simulator")

model_name = st.sidebar.selectbox("Pick model", list(results.keys()), index=list(results.keys()).index("XGBoost"))
best = results[model_name]["model"]
year_options = sorted(set(list(df.Year.unique()) + [2040, 2050, 2060]))
year = st.sidebar.selectbox("Year", year_options)
tech = st.sidebar.selectbox("Technology", sorted(df.Tech.unique()))
state = st.sidebar.selectbox("State", sorted(df.State.unique()))

# --- Tech-based ranges ---
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

st.sidebar.caption(f"CAPEX range: {cap_min}–{cap_max} | Efficiency range: {eff_min}–{eff_max}")

# --- Dynamic adjustment functions (defined BEFORE usage) ---
def adjust_capex_by_year(tech, year, original):
    t = {
        "ALKALINE": {2022:500,2026:250,2031:150,2040:100,2050:80,2060:60},
        "PEM":      {2022:1000,2026:250,2031:150,2040:100,2050:80,2060:60},
        "SOEC":     {2022:2500,2026:500,2031:200,2040:160,2050:120,2060:100}
    }.get(tech, {})
    years = sorted(t.keys())
    if year <= years[0]: return t[years[0]]
    elif year >= years[-1]: return t[years[-1]]
    for i in range(len(years)-1):
        if years[i] <= year <= years[i+1]:
            return round(np.interp(year, [years[i], years[i+1]], [t[years[i]], t[years[i+1]]]), 2)

def get_efficiency_by_year(tech, year):
    t = {
        "ALKALINE": {2022:55,2026:52,2031:48,2040:45,2050:43,2060:42},
        "PEM":      {2022:55,2026:51,2031:46,2040:43,2050:42,2060:41},
        "SOEC":     {2022:47,2026:44,2031:42,2040:41,2050:40,2060:38}
    }.get(tech, {})
    years = sorted(t.keys())
    if year <= years[0]: return t[years[0]]
    elif year >= years[-1]: return t[years[-1]]
    for i in range(len(years)-1):
        if years[i] <= year <= years[i+1]:
            return round(np.interp(year, [years[i], years[i+1]], [t[years[i]], t[years[i+1]]]), 2)

# --- Default values ---
default_cap = int((cap_min + cap_max) / 2)
default_eff = round((eff_min + eff_max) / 2, 1)
default_elec = float(round(df.loc[df.State == state, "Electricity_$/kWh_fixed"].iloc[0], 4))

# --- DOE Toggle + computed DOE values ---
auto_mode = st.sidebar.toggle("Use DOE projections?", value=True)
cap_doe   = adjust_capex_by_year(tech, year, default_cap)
eff_doe   = get_efficiency_by_year(tech, year)

# ✅ Convert and fallback if wrong type
cap_default  = float(cap_doe) if auto_mode else float(st.session_state.get("cap_val", default_cap))
eff_default  = float(eff_doe) if auto_mode else float(st.session_state.get("eff_val", default_eff))
elec_default = float(st.session_state.get("elec_val", default_elec))

# ✅ Clamp to avoid out-of-range errors
cap_default = max(min(cap_default, cap_max), cap_min)
eff_default = max(min(eff_default, eff_max), eff_min)

# ✅ Debug: check types and values
st.sidebar.write(f"cap_min={cap_min} cap_max={cap_max} cap_default={cap_default!r} (type={type(cap_default)})")
st.sidebar.write(f"eff_min={eff_min} eff_max={eff_max} eff_default={eff_default!r} (type={type(eff_default)})")


# --- Sliders ---
cap = st.sidebar.slider(
    "CAPEX ($/kW)",
    int(cap_min),
    int(cap_max),
    value=int(cap_default),
    step=1,  # 👈 FIXED! Now Streamlit is happy
    key="cap_val",
    disabled=auto_mode
)


eff_input = st.sidebar.slider(
    "Efficiency (kWh/kg)",
    float(eff_min),
    float(eff_max),
    value=float(eff_default[0]) if isinstance(eff_default, list) else float(eff_default),
    step=0.1,
    key="eff_val",
    disabled=auto_mode
)


elec = st.sidebar.slider("Electricity price ($/kWh)", 0.01, 0.30,
                         value=elec_default, key="elec_val")  # 👈 already floats, perfect


# --- Dynamic values for prediction ---
cap_dynamic = cap_doe if auto_mode else cap
eff_dynamic = eff_doe if auto_mode else eff_input

# --- Create prediction input row ---
row = {
    "CF": float(df.query("State==@state & Year==@year & Tech==@tech").CF.iloc[0]),
    "CAPEX_$/kW": cap_dynamic,
    "Efficiency_kWh_kg": eff_dynamic,
    "Electricity_$/kWh": elec,
    "Water_$kg": df["Water_$kg"].median(),
    "CO2_$kg": df["CO2_$kg"].median(),
    "Transport_$kg": df["Transport_$kg"].median(),
    "Storage_$kg": df["Storage_$kg"].median(),
    "Year_val": year,
    "Year_val_squared": year ** 2
}

for col in results[model_name]["features"]:
    if col.startswith("Tech_"):
        row[col] = 1 if col == f"Tech_{tech}" else 0

# --- Build model input and predict ---
def build_model_input(r, feat):
    df_in = pd.DataFrame([r])
    for c in feat:
        if c not in df_in.columns:
            df_in[c] = 0
    return df_in[feat]

features = results[model_name]["features"]
row_input = build_model_input(row, features)
pred = best.predict(row_input)[0]


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
    st.markdown("## Scenario Prediction")
    st.metric("Predicted LCOH", f"${pred:.2f}/kg")

    with st.expander("View Scenario Inputs"):
        st.write({
            "Technology": tech,
            "Year": year,
            "State": state,
            "CAPEX ($/kW)": cap_dynamic,
            "Efficiency (kWh/kg)": eff_dynamic,
            "Electricity Price ($/kWh)": elec,
            "Capacity Factor": row_input['CF'].values[0]
        })

with tabs[1]:
    st.markdown("## Model Evaluation Metrics")

    stats_df = pd.DataFrame({
        k: {
            "R²": v["r2"],
            "RMSE": v["rmse"],
            "MAE": v["mae"]
        } for k, v in results.items()
    }).T.round(3)

    st.dataframe(
        stats_df.style
        .highlight_max(axis=0, color="lightgreen")
        .highlight_min(axis=0, color="salmon")
        .format(precision=3)
        .background_gradient(cmap="viridis"),
        use_container_width=True
    )

    best_model_row = stats_df.loc[model_name]
    st.success(
        f"**{model_name}** Model Summary — R²: {best_model_row['R²']}, "
        f"RMSE: {best_model_row['RMSE']}, MAE: {best_model_row['MAE']}"
    )



with tabs[2]:
    fi = getattr(best, "feature_importances_", None)
    if fi is not None:
        cols = results[model_name]["features"]  # ensure consistent feature alignment
        fi_s = pd.Series(fi, index=cols).sort_values()
        fig = px.bar(
            fi_s.values, y=fi_s.index, orientation="h",
            color=fi_s.values, color_continuous_scale="Viridis",
            template="plotly_dark", title="Feature Importance"
        )
        fig.update_coloraxes(colorbar_title=None)
        fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

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

    dynamic_rows = []
    for st_name in sorted(df.State.unique()):
        try:
            subset = df.query("State == @st_name & Year == @year & Tech == @tech")
            cf_val = float(subset.CF.iloc[0])
        except:
            continue

        row = {
            "CF": cf_val,
            "CAPEX_$/kW": cap,
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

        for col in results[model_name]["features"]:
            if col.startswith("Year_"):
                row[col] = 1 if col == f"Year_{year}" else 0

        dynamic_rows.append(row)

    df_inputs = pd.DataFrame(dynamic_rows)

    def align_features(df_input, model_features):
        df_input = pd.get_dummies(df_input)
        for c in model_features:
            if c not in df_input.columns:
                df_input[c] = 0
        return df_input[model_features]

    df_pred = align_features(df_inputs.drop(columns=["State"]), results[model_name]["features"])
    df_inputs["LCOH ($/kg)"] = best.predict(df_pred)

    merged = gdf.merge(df_inputs[["State", "LCOH ($/kg)"]], on="State", how="left")

    fig_map = px.choropleth(
        merged, geojson=merged.geometry,
        locations=merged.index, color="LCOH ($/kg)",
        hover_name="State", color_continuous_scale="Viridis",
        template="plotly_dark",
        title=f"Predicted LCOH by State ({year}, {tech}, Model: {model_name})"
    )
    fig_map.update_geos(scope="usa")
    fig_map.update_coloraxes(colorbar_tickformat=".2f")

    st.plotly_chart(fig_map, use_container_width=True)


with tabs[4]:
    st.subheader("State-wise LCOH Summary (Simulated Variations)")

    # Simulated variations for CAPEX and Efficiency
    variations = []
    cap_values = [cap * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
    eff_values = [eff_dynamic * f for f in [0.95, 0.975, 1.0, 1.025, 1.05]]



    for st_name in sorted(df.State.unique()):
        try:
            subset = df.query("State == @st_name & Year == @year & Tech == @tech")
            if subset.empty:
                continue
            cf_val = float(subset["CF"].iloc[0])
            elec_val = float(subset["Electricity_$/kWh_fixed"].iloc[0])
        except (IndexError, ValueError):
            continue

        for c, e in zip(cap_values, eff_values):
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

            # ✅ BONUS: Inject all year one-hot columns based on model training features
            for col in results[model_name]["features"]:
                if col.startswith("Year_"):
                    row[col] = 1 if col == f"Year_{year}" else 0

            variations.append(row)

    df_var = pd.DataFrame(variations)

    # --- Align features before prediction ---
    def align_features(df, model_features):
        df = pd.get_dummies(df)
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        return df[model_features]

    features = results[model_name]["features"]
    df_pred = align_features(df_var.drop(columns=["State"]), features)
    df_var["LCOH ($/kg)"] = best.predict(df_pred)

    # --- Summary statistics
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

    # --- Export for download tab
    y_true = results[model_name]["y_true"]
    preds = results[model_name]["preds"]
    df_out = pd.DataFrame({
        "True": y_true,
        "Predicted": preds
    })
    df_out["Error"] = df_out["True"] - df_out["Predicted"]


with tabs[5]:
    st.subheader("🇺🇸 National LCOH Overview")

    # Filter only for selected technology
    national_df = df[df.Tech == tech]

    if national_df.empty:
        st.warning(f"No data available for {tech}.")
    else:
        national_trend = (
            national_df
            .groupby("Year")["LCOH_$kg"]
            .mean()
            .reset_index()
            .rename(columns={"LCOH_$kg": "Average LCOH ($/kg)"})
        )
        national_trend["Average LCOH ($/kg)"] = national_trend["Average LCOH ($/kg)"].round(2)

        # Plot national trend
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

    # --- Performance plot for PDF ---
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(y_true, label="True", linewidth=2)
    ax.plot(preds, label="Predicted", linestyle="--")
    ax.set_title(f"{model_name} – Prediction vs True")
    ax.legend()
    fig_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    # --- PDF Builder ---
    def generate_pdf(full=False):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "LCOH Model Report", ln=True)

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

        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Selected Scenario", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10,
    f"Technology: {tech}\n"
    f"Year: {year}\n"
    f"State: {state}\n"
    f"CAPEX: ${cap_dynamic}/kW\n"
    f"Efficiency: {eff_dynamic} kWh/kg\n"
    f"Electricity Price: ${elec}/kWh\n"
    f"Generated: {timestamp}"
)

        if full:
            # National Trend
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "National LCOH by Year", ln=True)
            pdf.set_font("Arial", "", 12)
            for _, row in national_trend.iterrows():
                pdf.cell(0, 10, f"{int(row['Year'])}: ${row['Average LCOH ($/kg)']:.2f}/kg", ln=True)

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

    # --- Output Download Handling ---
    if download_mode == "Scenario Report Only":
        pdf = generate_pdf(full=False)
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
        pdf = generate_pdf(full=True)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            z.writestr(f"LCOH_Full_Report_{timestamp}.pdf", pdf.output(dest="S").encode("latin1"))
            z.writestr("predictions.csv", df_out.to_csv(index=False))
            z.writestr("state_summary.csv", summary_stats.to_csv())
            z.writestr("national_trend.csv", national_trend.to_csv(index=False))
        zip_buffer.seek(0)
        st.download_button(
            "Download Full ZIP",
            data=zip_buffer,
            file_name=f"LCOH_Report_Bundle_{timestamp}.zip",
            mime="application/zip"
        )
