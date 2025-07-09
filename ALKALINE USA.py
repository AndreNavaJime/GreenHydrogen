# =============================================================================
#  ALKALINE USA.py  – Spyder-ready with decade-step projections
#  • Robust parsing of all CSV/XLSX inputs
#  • Automatic numeric coercion, NaN handling
#  • Uses solar, wind, land, water, infrastructure, CO₂ layers
#  • Projects LCOH for 2022, 2030, 2040, 2050, 2060
#  • Saves USAALKALINE.csv in the “MACHINE LEARNING” folder
# =============================================================================
# =============================================================================
#  ALKALINE USA.py  – Now with Transport + Storage Cost Integration
# =============================================================================
import pandas as pd, numpy as np, os, re

# ---------- 0. Paths ---------------------------------------------------------#
data_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\CSV TIF"
save_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\MACHINE LEARNING"
os.makedirs(save_dir, exist_ok=True)
os.chdir(data_dir)

# ---------- 1. Helper functions ---------------------------------------------#
def read_flexible(path, ncols, cols):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        tmp = df.iloc[:, 0].str.split(",", expand=True)
        if tmp.shape[1] < ncols:
            tmp = df.iloc[:, 0].str.split(r"\s+", expand=True)
        df = tmp
    df = df.iloc[:, :ncols]
    df.columns = cols
    return df

def clean_state(df, fname):
    cand = [c for c in df.columns if c.strip().lower() in ("state","name","state_name","st")]
    if cand:
        df.rename(columns={cand[0]: "State"}, inplace=True)
    else:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "State"}, inplace=True)
    df["State"] = df["State"].astype(str).str.strip()
    print(f"✔ {fname:15s} rows={len(df)}")
    return df

def lmerge(left, right, tag):
    out = left.merge(right, on="State", how="left")
    print(f"🔗 after {tag:<8s}: {len(out)} rows")
    return out

def force_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().sum():
            df[c] = df[c].fillna(df[c].median())
    return df

# ---------- 2. Load & coerce data -------------------------------------------#
elec = clean_state(
    pd.read_excel("ElectricityUSA.xlsx").rename(
        columns=lambda x: "Industrial_Price_C_per_kWh" if "Industrial" in x else x
    ),
    "ElectricityUSA.xlsx"
)
ghi = clean_state(read_flexible("GHIUSA.csv", 6,
       ["State","GHI_min","GHI_max","GHI_mean","GHI_median","GHI_std"]),
       "GHIUSA.csv")
wind = clean_state(read_flexible("WINDUSA.csv", 6,
       ["State","Wind_min","Wind_max","Wind_mean","Wind_median","Wind_std"]),
       "WINDUSA.csv")
land = clean_state(pd.read_csv("USLAND.csv").rename(
       columns={"solar":"SolarScore","suitability":"LandSuitability","wind":"WindScore"}),
       "USLAND.csv")
infra = clean_state(pd.read_excel("INFRAUSA.xlsx")[["State","Energy","Water","Dams"]],
       "INFRAUSA.xlsx")
water = clean_state(pd.read_csv("USWATER.csv").rename(
       columns={"area_km2":"Area_km2","runoff_km3":"Runoff_km3","runoff_mm":"Runoff_mm"}),
       "USWATER.csv")
biomass = clean_state(pd.read_csv("USABIOMASS.csv").rename(
         columns={"NAME":"State","AGB_Mg_ha":"Biomass_AGB_Mg_ha"})[["State","Biomass_AGB_Mg_ha"]],
         "USABIOMASS.csv")
co2 = clean_state(pd.read_csv("USACO2.csv"), "USACO2.csv")

ghi   = force_numeric(ghi, ghi.columns.drop("State"))
wind  = force_numeric(wind, wind.columns.drop("State"))
land  = force_numeric(land, ["SolarScore","LandSuitability","WindScore"])
water = force_numeric(water, ["Area_km2","Runoff_km3","Runoff_mm"])
infra = force_numeric(infra, ["Energy","Water","Dams"])
biomass = force_numeric(biomass, ["Biomass_AGB_Mg_ha"])
co2_cols = [c for c in co2.columns if re.fullmatch(r"\d{4}", str(c))]
co2 = force_numeric(co2, co2_cols)

# ---------- 2.1 CO₂ intensity -----------------------------------------------#
co2["CO2_avg_18_23"] = co2[co2_cols].loc[:, [c for c in co2_cols if 2018<=int(c)<=2023]].mean(axis=1, skipna=True)
co2["CO2_avg_18_23"] = co2["CO2_avg_18_23"].fillna(0)
co2["CO2_intensity_norm"] = co2["CO2_avg_18_23"] / co2["CO2_avg_18_23"].max()
co2 = co2[["State","CO2_intensity_norm"]]

# ---------- 2.2 Merge all ---------------------------------------------------#
df = elec
for frame, tag in [(ghi,"GHI"),(wind,"WIND"),(land,"LAND"),(infra,"INFRA"),
                   (water,"WATER"),(biomass,"BIOMASS"),(co2,"CO2")]:
    df = lmerge(df, frame, tag)
for c in df.select_dtypes(include="number").columns:
    df[c] = df[c].fillna(df[c].median())

# ---------- 3. Derived multipliers & costs ---------------------------------#
solar_min, solar_max = df["SolarScore"].min(), df["SolarScore"].max()
wind_min, wind_max   = df["Wind_mean"].min(), df["Wind_mean"].max()
df["solar_norm"]   = (df["SolarScore"] - solar_min)/(solar_max - solar_min)
df["wind_norm"]    = (df["Wind_mean"]  - wind_min )/(wind_max  - wind_min)
df["resource_idx"] = 0.6*df["solar_norm"] + 0.4*df["wind_norm"]
df["CF_state"]     = 0.6 + 0.3*df["resource_idx"]

df["capex_mult_land"]  = 1 + 0.25*(1 - df["LandSuitability"])
df["infra_norm"]       = (df["Energy"]+df["Water"]+df["Dams"])/12
df["capex_mult_infra"] = 1 - 0.15*df["infra_norm"]

df["water_cost_$kg"]    = 0.001*np.exp(-df["Runoff_mm"]/200)
df["co2_surcharge_$kg"] = 0.02*df["CO2_intensity_norm"]
df["water_cost_$kg"] = df["water_cost_$kg"].fillna(0.0)
df["co2_surcharge_$kg"] = df["co2_surcharge_$kg"].fillna(0.0)

# ---------- 3.1 Add Transport & Storage Costs -------------------------------#
df["Transport_km"] = 200
df["transport_cost_$kg"] = df["Transport_km"] * 0.0005
df["storage_cost_$kg"] = 0.50

# ---------- 4. Techno-economic parameters -----------------------------------#
disc = 0.08
def crf(life_yr):
    r = disc
    return r*(1+r)**life_yr / ((1+r)**life_yr - 1)

opex_ratio = 0.05
cap22, cap26, cap31 = 500, 250, 150
eff22, eff26, eff31 = 55, 52, 48
eff_floor = 45

def cap_eff(y):
    if y <= 2026:
        cap = np.interp(y, [2022,2026], [cap22,cap26])
        eff = np.interp(y, [2022,2026], [eff22,eff26])
    elif y <= 2031:
        cap = np.interp(y, [2026,2031], [cap26,cap31])
        eff = np.interp(y, [2026,2031], [eff26,eff31])
    else:
        cap = cap31*(0.96**((y-2031)/5))
        eff = max(eff_floor, eff31*(1-0.002*(y-2031)))
    return cap, eff

# ---------- 5. Projection (decade steps) ------------------------------------#
years = [2022, 2030, 2040, 2050, 2060]
records = []

for _, r in df.iterrows():
    state = r["State"]
    p0    = r["Industrial_Price_C_per_kWh"]/100
    cf    = r["CF_state"]
    cmL   = r["capex_mult_land"]
    cmI   = r["capex_mult_infra"]
    wc    = r["water_cost_$kg"]
    cs    = r["co2_surcharge_$kg"]
    tc    = r["transport_cost_$kg"]
    sc    = r["storage_cost_$kg"]

    for y in years:
        cap_nom, eff = cap_eff(y)
        cap = cap_nom * cmL * cmI
        p_e = p0 * (1 + (y-2022)/(2060-2022)*(-0.30))
        kgH2 = cf * 8760 / eff

        lcoh = (
            cap*crf(20)/kgH2 +
            cap*opex_ratio*(cap/cap22)/kgH2 +
            p_e*eff +
            wc + cs + tc + sc
        )

        records.append([
            state, y,
            round(cf,3),
            round(cap,2),
            round(eff,2),
            round(p_e,4),
            round(wc,4),
            round(cs,4),
            round(tc,4),
            round(sc,4),
            round(lcoh,4)
        ])

df_out = pd.DataFrame(records, columns=[
    "State","Year","CF","CAPEX_$/kW","Efficiency_kWh_kg",
    "Electricity_$/kWh","Water_$kg","CO2_$kg",
    "Transport_$kg","Storage_$kg","LCOH_Alk_$kg"
])

outfile = os.path.join(save_dir, "USAALKALINE.csv")
df_out.to_csv(outfile, index=False)

print(df_out.head(10))
print(f"\n✅ Saved → {outfile}")


