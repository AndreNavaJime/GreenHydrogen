# =============================================================================
#  SOEC USA.py – Spyder-ready with decade-step projections
#  Adds:
#   (1) Electricity price scenarios (conservative/moderate/aggressive)
#   (2) Transparent price calc (no hardcoded -30%)
#   (3) Scenario header print (audit trail)
#   (4) CF diagnostic print (Eq. 7 sanity check)
#   (5) Scenario-tagged output filenames (no overwrites)
#   (6) CAPEX scenarios: baseline (0.96) and slower (0.98) from Table S4
#   (7) Rank-flip & overlay exports also scenario-tagged
#  Keeps: Stack replacement, delivery overlays, all other TEA logic unchanged
# =============================================================================
import os, re
import pandas as pd
import numpy as np

# ---------- 0. Paths ---------------------------------------------------------#
data_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\CSV TIF"
save_dir = r"C:\Users\Navar\OneDrive\Documentos\USA"
os.makedirs(save_dir, exist_ok=True)
os.chdir(data_dir)

# ---------- 0.1 TEA toggles --------------------------------------------------#
PLANT_LIFE_YEARS        = 20
DISCOUNT_RATE           = 0.08
OPEX_RATIO              = 0.05
OPEX_FACTOR             = 1.0
WITH_REPLACEMENT        = False
STACK_LIFE_HOURS_SOEC   = 0
REPLACEMENT_COST_FRACT  = 0.0

# Delivery/Storage overlays (per Supplementary Table 3)
SCENARIOS = {
    "Low":  {"per_km": 0.00025, "storage": 0.20},
    "Med":  {"per_km": 0.00050, "storage": 0.50},
    "High": {"per_km": 0.00100, "storage": 0.90},
}

print("Overlay sanity @ 10 km (Low/Med/High):")
for k, v in SCENARIOS.items():
    print(f"  {k}: {10.0*v['per_km'] + v['storage']:.5f} $/kg")
# Expect ~0.20250 / 0.50500 / 0.91000


# ========== Electricity price scenarios ==========
ELECTRICITY_PRICE_SCENARIOS = {
    'conservative': {'decline_pct': 0.10, 'description': 'Slow grid decarbonization'},
    'moderate':     {'decline_pct': 0.30, 'description': 'Baseline electricity price decline (Table S3 default)'},
    'aggressive':   {'decline_pct': 0.50, 'description': 'Aggressive grid decarbonization + VRE cost learning'},
}
price_scenario = 'moderate'   # 'conservative' | 'moderate' | 'aggressive'

# ========== SOEC CAPEX scenarios (Table S4) ==========
CAPEX_SCENARIOS = {
    'baseline': {'decay_rate': 0.96, 'description': 'Baseline learning: 0.96 decay post-2031 (Table S4)'},
    'slower':   {'decay_rate': 0.98, 'description': 'Slower learning: 0.98 decay post-2031 (Table S4)'},
}
capex_scenario = 'baseline'   # 'baseline' | 'slower'

# ---------- 1) CO₂ helpers ---------------------------------------------------#
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
    # find 2018–2023 columns (salvage messy headers)
    year_cols = [c for c in co2.columns if re.fullmatch(r"\d{4}", str(c).strip())]
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

# ---------- 2) Generic helpers ----------------------------------------------#
def read_flexible(path, ncols, cols):
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        tmp = df.iloc[:, 0].str.split(",", expand=True)
        if tmp.shape[1] < ncols:
            tmp = df.iloc[:, 0].str.split(r"\s+", expand=True)
        df = tmp
    df = df.iloc[:, :ncols]
    df.columns = cols
    df = df[~df[cols[0]].str.lower().str.contains("state")]
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

# ---------- 3) Load & coerce layers -----------------------------------------#
elec = clean_state(
    pd.read_excel("ElectricityUSA.xlsx").rename(
        columns=lambda x: "Industrial_Price_C_per_kWh" if "Industrial" in x else x
    ), "ElectricityUSA.xlsx"
)
ghi  = clean_state(read_flexible("GHIUSA.csv", 6,
        ["State","GHI_min","GHI_max","GHI_mean","GHI_median","GHI_std"]), "GHIUSA.csv")
wind = clean_state(read_flexible("WINDUSA.csv", 6,
        ["State","Wind_min","Wind_max","Wind_mean","Wind_median","Wind_std"]), "WINDUSA.csv")

land = clean_state(pd.read_csv("USLAND.csv"), "USLAND.csv")
land.columns = land.columns.str.strip()
infra = clean_state(pd.read_excel("INFRAUSA.xlsx")[["State","Energy","Water","Dams"]],
        "INFRAUSA.xlsx")

_ = pd.read_csv("USWATER.csv", header=None)
water = clean_state(pd.read_csv("USWATER.csv"), "USWATER.csv")
water.columns = water.columns.str.strip().str.lower()
water.rename(columns={"state":"State","area_km2":"Area_km2","runoff_km3":"Runoff_Km3","runoff_mm":"Runoff_mm"}, inplace=True)

biomass = clean_state(pd.read_csv("USABIOMASS.csv").rename(
    columns={"NAME":"State","AGB_Mg_ha":"Biomass_AGB_Mg_ha"}
)[["State","Biomass_AGB_Mg_ha"]], "USABIOMASS.csv")

grid = clean_state(read_flexible("STATEGRIDDISTANCE.csv", 2, ["State","distance_to_grid_km"]),
                   "STATEGRIDDISTANCE.csv")
grid = grid[grid["State"].str.lower() != "state_name"]

ghi    = force_numeric(ghi, ghi.columns.drop("State"))
wind   = force_numeric(wind, wind.columns.drop("State"))
land   = force_numeric(land, ["suitability"])
water  = force_numeric(water, ["Runoff_Km3"])
infra  = force_numeric(infra, ["Energy","Water","Dams"])
biomass= force_numeric(biomass, ["Biomass_AGB_Mg_ha"])
grid   = force_numeric(grid, ["distance_to_grid_km"])

# ---------- 3.1 Merge non-CO₂ layers ----------------------------------------#
df = elec
for frame, tag in [(ghi,"GHI"),(wind,"WIND"),(land,"LAND"),(infra,"INFRA"),
                   (water,"WATER"),(biomass,"BIOMASS"),(grid,"GRID")]:
    df = lmerge(df, frame, tag)

# ---------- 3.2 CO₂ load (SOEC factor = 0.05) --------------------------------
co2_df = load_co2_robust(os.path.join(data_dir, "USACO2.csv"), tech_factor=0.05)
df     = merge_co2_or_raise(df, co2_df)
assert df["co2_surcharge_$kg"].notna().all(), "CO2 merge introduced NaNs."
assert df["co2_surcharge_$kg"].max() > 0, "CO2 surcharge all zeros—check USACO2.csv."

# Keep only contiguous 48
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

# ---------- 4) Derived multipliers & proxies --------------------------------#
df["capex_mult_land"]  = 1 + (1 - df["suitability"]) * 0.25
df["capex_mult_infra"] = 1 + (1 - df["Energy"] / df["Energy"].max()) * 0.25

# Water proxy (SOEC cap = 0.002 USD/kg)
df["Runoff_Km3"]     = df["Runoff_Km3"].replace(0, np.nan)
df["water_cost_$kg"] = (1 / df["Runoff_Km3"])
df["water_cost_$kg"] = df["water_cost_$kg"] / (df["water_cost_$kg"].max() + 1e-12) * 0.002
df["water_cost_$kg"] = df["water_cost_$kg"].fillna(df["water_cost_$kg"].median())

# Resource index → CF in [0.6, 0.9]
ghi_min, ghi_max   = df["GHI_mean"].min(), df["GHI_mean"].max()
wind_min, wind_max = df["Wind_mean"].min(), df["Wind_mean"].max()
df["solar_norm"]   = (df["GHI_mean"] - ghi_min) / (ghi_max - ghi_min + 1e-12)
df["wind_norm"]    = (df["Wind_mean"] - wind_min) / (wind_max - wind_min + 1e-12)
df["resource_idx"] = 0.6 * df["solar_norm"] + 0.4 * df["wind_norm"]
df["CF_state"]     = 0.6 + 0.3 * df["resource_idx"]

# ========== CF DIAGNOSTIC (Eq. 7 Validation) ==========
print("\n" + "="*70)
print("CAPACITY FACTOR DIAGNOSTIC (Equation 7)")
print("="*70)
print("Formula: CF_i = 0.6 + 0.3 × R_i, where R_i = 0.6·Solar_norm + 0.4·Wind_norm")
print(f"CF_state range: {df['CF_state'].min():.4f} – {df['CF_state'].max():.4f}")
print(f"CF_state mean:  {df['CF_state'].mean():.4f}")
print("Expected plant CF ranges (screening-grade): ~0.15–0.55 (unfirmed), higher if firmed.")
low_out  = df.loc[df['CF_state'] < 0.15, 'State'].tolist()
high_out = df.loc[df['CF_state'] > 0.55, 'State'].tolist()
if low_out:  print(f"⚠️  Low CF states (<0.15): {low_out}")
if high_out: print(f"⚠️  High CF states (>0.55): {high_out}")
if not low_out and not high_out:
    print("✅ All CF values within 0.15–0.55 screening range (or appropriately firmed).")
print("="*70 + "\n")
# ========== END CF DIAGNOSTIC ==========

# Transport distance
df["Transport_km"] = df["distance_to_grid_km"].fillna(200)

# ---------- 5) SOEC trajectories (Table S4/S3) ------------------------------#
def cap_sec(year, capex_scenario='baseline'):
    """
    Returns (CAPEX $/kW, SEC kWh/kg) from Table S4 trajectories.
    capex_scenario: 'baseline' (0.96 decay) or 'slower' (0.98 decay) post-2031.
    """
    decay_rate = CAPEX_SCENARIOS[capex_scenario]['decay_rate']

    cap22, cap26, cap31 = 2500, 500, 200   # $/kW
    sec22, sec26, sec31 = 38, 36, 35       # kWh/kg
    sec_floor = 34

    if year <= 2026:
        cap = np.interp(year, [2022,2026], [cap22,cap26])
        sec = np.interp(year, [2022,2026], [sec22,sec26])
    elif year <= 2031:
        cap = np.interp(year, [2026,2031], [cap26,cap31])
        sec = np.interp(year, [2026,2031], [sec26,sec31])
    else:
        cap = cap31 * (decay_rate**((year-2031)/5))
        sec = max(sec_floor, sec31 * (1 - 0.002 * (year-2031)))
    return cap, sec

# ---------- 6) Projection ----------------------------------------------------#
years = [2022, 2030, 2040, 2050, 2060]
rows  = []

# Scenario headers
print("\n" + "="*70)
print("SOEC TECHNO-ECONOMIC ASSESSMENT — ACTIVE SCENARIOS")
print("="*70)
print(f"Electricity Price Scenario: '{price_scenario}'")
print(f"  Description : {ELECTRICITY_PRICE_SCENARIOS[price_scenario]['description']}")
print(f"  Decline Rate: {ELECTRICITY_PRICE_SCENARIOS[price_scenario]['decline_pct']*100:.0f}% from 2022→2060 (linear)")
print(f"\nSOEC CAPEX Scenario: '{capex_scenario}'")
print(f"  Description: {CAPEX_SCENARIOS[capex_scenario]['description']}")
cap2060_preview = 200 * (CAPEX_SCENARIOS[capex_scenario]['decay_rate']**((2060-2031)/5))
print(f"  2060 CAPEX preview: {cap2060_preview:.1f} $/kW")
try:
    p_ok_2022 = float(elec.loc[elec['State'].eq('Oklahoma'), 'Industrial_Price_C_per_kWh'].iloc[0]) / 100.0
    d = ELECTRICITY_PRICE_SCENARIOS[price_scenario]['decline_pct']
    p_ok_2060 = p_ok_2022 * (1 - d)
    print(f"\nOklahoma example: 2022 ${p_ok_2022:.4f}/kWh → 2060 ${p_ok_2060:.4f}/kWh (decline ${p_ok_2022 - p_ok_2060:.4f}/kWh)")
except Exception:
    print("Oklahoma price preview not available (check ElectricityUSA.xlsx).")
print("="*70 + "\n")

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
        cap_nom, sec = cap_sec(year=y, capex_scenario=capex_scenario)
        cap = cap_nom * cmL * cmI

        # Scenario-based electricity path
        decline_rate = ELECTRICITY_PRICE_SCENARIOS[price_scenario]['decline_pct']
        p_e = p0 * (1 - decline_rate * (y - 2022) / (2060 - 2022))

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
                STACK_LIFE_HOURS_SOEC, annual_hours, PLANT_LIFE_YEARS,
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

# ---------- 7) Outputs -------------------------------------------------------#
cols = [
    "State","Year","CF","CAPEX_$/kW","SEC_kWh_kg","Electricity_$/kWh",
    "Water_$kg","CO2_$kg","Transport_km",
    "C_capex_usd_kg","C_opex_usd_kg","C_repl_usd_kg","C_elec_usd_kg",
    "LCOH_plantgate_$kg","LCOH_Low_$kg","LCOH_Med_$kg","LCOH_High_$kg"
]
df_out = pd.DataFrame(rows, columns=cols)

# Scenario-tagged filenames (price + capex)
outfile_main = os.path.join(save_dir, f"USASOEC_price_{price_scenario}_capex_{capex_scenario}.csv")
overlay_file = os.path.join(save_dir, f"USASOEC_price_{price_scenario}_capex_{capex_scenario}_OVERLAYS.csv")
flips_file   = os.path.join(save_dir, f"USASOEC_price_{price_scenario}_capex_{capex_scenario}_RANK_FLIPS.csv")

df_out.to_csv(outfile_main, index=False)
print(df_out.head(10))
print(f"\n✅ Saved → {outfile_main}")

# Ranking flips vs plant-gate
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
    flips.to_csv(flips_file, index=False)
    print(f"📝 Delivery/storage ranking flips saved → {flips_file}")

# Compact overlay summary
df_out[["State","Year","LCOH_plantgate_$kg","LCOH_Low_$kg","LCOH_Med_$kg","LCOH_High_$kg"]].to_csv(overlay_file, index=False)
print(f"🧮 Overlays saved → {overlay_file}")
