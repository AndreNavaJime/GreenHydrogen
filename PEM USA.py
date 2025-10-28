# =============================================================================
#  PEM USA.py – resource-aware deterministic LCOH model for PEM electrolysis
#  • Preserves Table S3 values (discount=0.08, life=20, OPEX=0.05)
#  • Robust CO₂ loader + guarded merge (no silent zeros)
#  • Plant-gate LCOH + Low/Med/High overlays (equal to baseline by default)
#  • Stack-replacement term supported but DISABLED by default
#  • Exports component terms + ranking-flip report
# =============================================================================
import os, re
import pandas as pd
import numpy as np

# ---------- 0. Paths ---------------------------------------------------------#
data_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\CSV TIF"
save_dir = r"C:\Users\Navar\OneDrive\Documentos\USA"   # per your request
os.makedirs(save_dir, exist_ok=True)
os.chdir(data_dir)

# ---------- 0.1 Global TEA toggles (neutral by default) ---------------------#
PLANT_LIFE_YEARS        = 20
DISCOUNT_RATE           = 0.08
OPEX_RATIO              = 0.05     # Table S3
OPEX_FACTOR             = 1.0      # set 0.5/1.0/1.5 for ±50% tests
WITH_REPLACEMENT        = False    # keep False to avoid new effects
STACK_LIFE_HOURS_PEM    = 0        # enable only if you turn WITH_REPLACEMENT=True
REPLACEMENT_COST_FRACT  = 0.0

# Delivery/Storage overlays — equal to baseline by default
SCENARIOS = {
    "Low":  {"per_km": 0.0005, "storage": 0.50},
    "Med":  {"per_km": 0.0005, "storage": 0.50},
    "High": {"per_km": 0.0005, "storage": 0.50},
}

# ---------- 1. CO₂ helpers (robust loader + guarded merge) ------------------#
USPS_TO_NAME = {
    'AL':'Alabama','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado','CT':'Connecticut',
    'DE':'Delaware','FL':'Florida','GA':'Georgia','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MA':'Massachusetts',
    'MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska',
    'NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York',
    'NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania',
    'RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah',
    'VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
}
def normalize_state_names(s):
    s = (str(s) or "").strip()
    if len(s) == 2 and s.upper() in USPS_TO_NAME:
        return USPS_TO_NAME[s.upper()]
    return s.replace("  "," ").strip().title()

def load_co2_robust(path_csv, tech_factor):
    co2 = pd.read_csv(path_csv, encoding="utf-8-sig")
    state_col = next((c for c in co2.columns if c.strip().lower() in
                      ("state","state_name","name","st","state_abbrev","statecode")), co2.columns[0])
    co2 = co2.rename(columns={state_col:"State"})
    co2["State"] = co2["State"].map(normalize_state_names)
    # find year columns (2018–2023)
    year_cols = []
    for c in co2.columns:
        cs = str(c).strip()
        if re.fullmatch(r"\d{4}", cs):
            year_cols.append(c)
    if not year_cols:
        new_cols = {}
        for c in co2.columns:
            cs = re.sub(r"\D", "", str(c))
            if re.fullmatch(r"\d{4}", cs):
                new_cols[c] = cs
        co2 = co2.rename(columns=new_cols)
        year_cols = [c for c in co2.columns if re.fullmatch(r"\d{4}", str(c))]
    year_1823 = [c for c in year_cols if 2018 <= int(str(c)) <= 2023]
    for c in year_1823:
        co2[c] = pd.to_numeric(co2[c], errors="coerce")
    # average & normalize
    mask_valid = co2[year_1823].notna().any(axis=1)
    co2 = co2[mask_valid].copy()
    co2["CO2_avg_18_23"] = co2[year_1823].mean(axis=1)
    maxv = co2["CO2_avg_18_23"].max(skipna=True)
    co2["CO2_intensity_norm"] = 0.0 if (pd.isna(maxv) or maxv == 0) else co2["CO2_avg_18_23"] / maxv
    co2["co2_surcharge_$kg"]  = co2["CO2_intensity_norm"] * float(tech_factor)
    return co2[["State","co2_surcharge_$kg"]].copy()

def merge_co2_or_raise(df_states, co2_df):
    out = df_states.merge(co2_df, on="State", how="left", validate="m:1")
    missing = sorted(set(out.loc[out["co2_surcharge_$kg"].isna(), "State"]))
    if missing:
        raise ValueError(f"CO2 merge miss for {len(missing)} states: {missing}")
    return out

# ---------- 2. Generic helpers ----------------------------------------------#
def read_flexible(path, ncols, cols):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        tmp = df.iloc[:, 0].str.split(",", expand=True)
        if tmp.shape[1] < ncols:
            tmp = df.iloc[:, 0].str.split(r"\s+", expand=True)
        df = tmp
    df = df.iloc[:, :ncols]
    df.columns = cols
    df = df[~df[cols[0]].str.lower().str.contains("state")]  # drop accidental header rows
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
    res = left.merge(right, on="State", how="left")
    print(f"🔗 after {tag:<8s}: {len(res)} rows")
    return res

def force_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().sum():
            df[c] = df[c].fillna(df[c].median())
    return df

def crf(life_yr, r=DISCOUNT_RATE):
    r = float(r)
    return r*(1+r)**life_yr / ((1+r)**life_yr - 1)

def kg_per_kWyr(capacity_factor, sec_kWh_per_kg):
    return (capacity_factor * 8760.0) / max(sec_kWh_per_kg, 1e-12)

def pv_stack_repl_usd_per_kW(stack_life_hours, annual_hours, plant_life_years,
                             discount_rate, replacement_cost_fraction, base_capex_per_kW):
    if (replacement_cost_fraction <= 0) or (stack_life_hours <= 0):
        return 0.0
    hours_total = annual_hours * plant_life_years
    n_repl = int(np.floor(hours_total / stack_life_hours))
    pv = 0.0
    for i in range(1, n_repl + 1):
        year_i = (i * stack_life_hours) / max(annual_hours, 1e-12)
        cost_i = replacement_cost_fraction * base_capex_per_kW
        pv += cost_i / ((1 + discount_rate) ** year_i)
    return pv

def usd_per_kW_to_usd_per_kg(usd_per_kW, capacity_factor, sec_kWh_per_kg):
    kgs = kg_per_kWyr(capacity_factor, sec_kWh_per_kg)
    return usd_per_kW / max(kgs, 1e-12)

# ---------- 3. Load & coerce data -------------------------------------------#
elec = clean_state(
    pd.read_excel("ElectricityUSA.xlsx").rename(
        columns=lambda x: "Industrial_Price_C_per_kWh" if "Industrial" in x else x
    ), "ElectricityUSA.xlsx"
)
ghi   = clean_state(read_flexible("GHIUSA.csv", 6,
        ["State","GHI_min","GHI_max","GHI_mean","GHI_median","GHI_std"]), "GHIUSA.csv")
wind  = clean_state(read_flexible("WINDUSA.csv", 6,
        ["State","Wind_min","Wind_max","Wind_mean","Wind_median","Wind_std"]), "WINDUSA.csv")

land = clean_state(pd.read_csv("USLAND.csv"), "USLAND.csv")
land.columns = land.columns.str.strip()
infra = clean_state(pd.read_excel("INFRAUSA.xlsx")[["State","Energy","Water","Dams"]],
        "INFRAUSA.xlsx")

# Water
_ = pd.read_csv("USWATER.csv", header=None)  # preview
water = clean_state(pd.read_csv("USWATER.csv"), "USWATER.csv")
water.columns = water.columns.str.strip().str.lower()
water.rename(columns={
    "state":"State","area_km2":"Area_km2","runoff_km3":"Runoff_Km3","runoff_mm":"Runoff_mm"
}, inplace=True)

biomass = clean_state(pd.read_csv("USABIOMASS.csv").rename(
    columns={"NAME":"State","AGB_Mg_ha":"Biomass_AGB_Mg_ha"}
)[["State","Biomass_AGB_Mg_ha"]], "USABIOMASS.csv")

# Grid distance
grid = clean_state(read_flexible("STATEGRIDDISTANCE.csv", 2, ["State","distance_to_grid_km"]),
                   "STATEGRIDDISTANCE.csv")
grid = grid[grid["State"].str.lower() != "state_name"]

# Numeric coercion
grid   = force_numeric(grid, ["distance_to_grid_km"])
ghi    = force_numeric(ghi, ghi.columns.drop("State"))
wind   = force_numeric(wind, wind.columns.drop("State"))
land   = force_numeric(land, ["suitability"])
water  = force_numeric(water, ["Runoff_Km3"])
infra  = force_numeric(infra, ["Energy","Water","Dams"])
biomass= force_numeric(biomass, ["Biomass_AGB_Mg_ha"])

# ---------- 3.1 Merge non-CO₂ layers ----------------------------------------#
df = elec
for frame, tag in [(ghi,"GHI"),(wind,"WIND"),(land,"LAND"),(infra,"INFRA"),
                   (water,"WATER"),(biomass,"BIOMASS"),(grid,"GRID")]:
    df = lmerge(df, frame, tag)

# ---------- 3.2 CO₂ load (PEM factor = 0.05) + guarded merge ----------------#
co2_df = load_co2_robust(os.path.join(data_dir, "USACO2.csv"), tech_factor=0.05)
df = merge_co2_or_raise(df, co2_df)
assert df["co2_surcharge_$kg"].notna().all(), "CO2 merge introduced NaNs."
assert df["co2_surcharge_$kg"].max() > 0, "CO2 surcharge all zeros—check USACO2.csv content."

# ---- Contiguous 48 only -----------------------------------------------------#
contig_48 = {
    'Alabama','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Idaho',
    'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico',
    'New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
    'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington',
    'West Virginia','Wisconsin','Wyoming'
}
df = df[df["State"].isin(contig_48)].copy()
assert len(df) == 48, "Row count drifted—check input files."
print(f"🗂  Filtered to {len(df)} contiguous states")

# ---------- 4. Derived multipliers & proxies --------------------------------#
# CAPEX multipliers
df["capex_mult_land"]  = 1 + (1 - df["suitability"]) * 0.25
df["capex_mult_infra"] = 1 + (1 - df["Energy"] / df["Energy"].max()) * 0.25

# Water proxy (PEM cap = 0.002 USD/kg)
df["Runoff_Km3"]     = df["Runoff_Km3"].replace(0, np.nan)
df["water_cost_$kg"] = (1 / df["Runoff_Km3"])
df["water_cost_$kg"] = df["water_cost_$kg"] / df["water_cost_$kg"].max() * 0.002
df["water_cost_$kg"] = df["water_cost_$kg"].fillna(df["water_cost_$kg"].median())

# Resource index → CF
ghi_min, ghi_max   = df["GHI_mean"].min(), df["GHI_mean"].max()
wind_min, wind_max = df["Wind_mean"].min(), df["Wind_mean"].max()
df["solar_norm"]   = (df["GHI_mean"] - ghi_min) / (ghi_max - ghi_min)
df["wind_norm"]    = (df["Wind_mean"] - wind_min) / (wind_max - wind_min)
df["resource_idx"] = 0.6 * df["solar_norm"] + 0.4 * df["wind_norm"]
df["CF_state"]     = 0.6 + 0.3 * df["resource_idx"]

# Transport distance (km) for overlays
df["Transport_km"] = df["distance_to_grid_km"].fillna(200)

# ---------- 5. PEM trajectories (Table S3) ----------------------------------#
cap22, cap26, cap31 = 1000, 250, 150   # $/kW
sec22, sec26, sec31 = 55, 51, 46       # kWh/kg
sec_floor = 43

def cap_sec(year):
    if year <= 2026:
        cap = np.interp(year, [2022,2026], [cap22,cap26])
        sec = np.interp(year, [2022,2026], [sec22,sec26])
    elif year <= 2031:
        cap = np.interp(year, [2026,2031], [cap26,cap31])
        sec = np.interp(year, [2026,2031], [sec26,sec31])
    else:
        cap = cap31 * (0.96**((year-2031)/5))
        sec = max(sec_floor, sec31 * (1 - 0.002 * (year-2031)))
    return cap, sec

# ---------- 6. Projection ----------------------------------------------------#
years   = [2022, 2030, 2040, 2050, 2060]
rows    = []

for _, r in df.iterrows():
    state = r["State"]
    p0    = r["Industrial_Price_C_per_kWh"] / 100.0   # $/kWh
    cf    = float(r["CF_state"])
    cmL   = float(r["capex_mult_land"])
    cmI   = float(r["capex_mult_infra"])
    wc    = float(r["water_cost_$kg"])
    cs    = float(r["co2_surcharge_$kg"])
    dist  = float(r["Transport_km"])

    for y in years:
        cap_nom, sec = cap_sec(y)
        cap = cap_nom * cmL * cmI
        p_e = p0 * (1 + (y - 2022) / (2060 - 2022) * (-0.30))  # −30% linear path

        if any(pd.isna(v) for v in [cap, sec, cf]) or sec <= 0 or cf <= 0:
            print(f"⚠️ SKIPPED {state}-{y}: cap={cap}, SEC={sec}, cf={cf}")
            rows.append([state,y,cf,cap,sec,p_e,wc,cs,dist,
                         np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            continue

        kgH2       = kg_per_kWyr(cf, sec)                 # kg/kW·yr
        capex_term = cap * crf(PLANT_LIFE_YEARS) / kgH2
        opex_term  = (cap * OPEX_RATIO * OPEX_FACTOR) / kgH2
        elec_term  = p_e * sec

        if WITH_REPLACEMENT:
            annual_hours = cf * 8760.0
            pv_repl_kW = pv_stack_repl_usd_per_kW(
                STACK_LIFE_HOURS_PEM, annual_hours, PLANT_LIFE_YEARS,
                DISCOUNT_RATE, REPLACEMENT_COST_FRACT, cap_nom
            )
            repl_term = usd_per_kW_to_usd_per_kg(pv_repl_kW, cf, sec)
        else:
            repl_term = 0.0

        lcoh_pg = capex_term + opex_term + repl_term + elec_term + wc + cs

        overlay = {sc: dist * pars["per_km"] + pars["storage"] for sc, pars in SCENARIOS.items()}

        rows.append([
            state, y, round(cf,3), round(cap,2), round(sec,2), round(p_e,4),
            round(wc,4), round(cs,4), round(dist,1),
            round(capex_term,4), round(opex_term,4), round(repl_term,4), round(elec_term,4),
            round(lcoh_pg,4),
            round(lcoh_pg + overlay["Low"], 4),
            round(lcoh_pg + overlay["Med"], 4),
            round(lcoh_pg + overlay["High"],4),
        ])

# ---------- 7. Outputs ------------------------------------------------------#
cols = [
    "State","Year","CF","CAPEX_$/kW","SEC_kWh_kg","Electricity_$/kWh",
    "Water_$kg","CO2_$kg","Transport_km",
    "C_capex_usd_kg","C_opex_usd_kg","C_repl_usd_kg","C_elec_usd_kg",
    "LCOH_plantgate_$kg","LCOH_Low_$kg","LCOH_Med_$kg","LCOH_High_$kg"
]
df_out = pd.DataFrame(rows, columns=cols)

outfile = os.path.join(save_dir, "USAPEM.csv")
df_out.to_csv(outfile, index=False)
print(df_out.head(10))
print(f"\n✅ Saved → {outfile}")

# Ranking flips (will be empty until overlays differ)
flip_rows = []
for y in years:
    tmp = df_out[df_out["Year"] == y].copy()
    if tmp.empty: 
        continue
    tmp["rank_pg"]   = tmp["LCOH_plantgate_$kg"].rank(method="min")
    tmp["rank_low"]  = tmp["LCOH_Low_$kg"].rank(method="min")
    tmp["rank_med"]  = tmp["LCOH_Med_$kg"].rank(method="min")
    tmp["rank_high"] = tmp["LCOH_High_$kg"].rank(method="min")
    tmp["flip_low"]  = (tmp["rank_low"]  != tmp["rank_pg"])
    tmp["flip_med"]  = (tmp["rank_med"]  != tmp["rank_pg"])
    tmp["flip_high"] = (tmp["rank_high"] != tmp["rank_pg"])
    flip_rows.append(tmp[["State","Year","rank_pg","rank_low","rank_med","rank_high",
                          "flip_low","flip_med","flip_high"]])

if flip_rows:
    flips = pd.concat(flip_rows, ignore_index=True)
    flips_file = os.path.join(save_dir, "USAPEM_DELIVERY_RANK_FLIPS.csv")
    flips.to_csv(flips_file, index=False)
    print(f"📝 Delivery/storage ranking flips saved → {flips_file}")

overlay_file = os.path.join(save_dir, "USAPEM_OVERLAYS.csv")
df_out[["State","Year","LCOH_plantgate_$kg","LCOH_Low_$kg","LCOH_Med_$kg","LCOH_High_$kg"]].to_csv(overlay_file, index=False)
print(f"🧮 Overlays saved → {overlay_file}")
