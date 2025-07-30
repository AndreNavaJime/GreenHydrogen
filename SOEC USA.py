# =============================================================================
#  SOEC USA.py  – Decade-step LCOH model for high-temperature
#              electrolysis (SOEC); mirrors Alkaline and PEM scripts.
#              Projects for 2022, 2030, 2040, 2050, 2060.
# =============================================================================
# =============================================================================
#  SOEC USA.py – Now with Transport + Storage Cost Integration (2022–2060)
# =============================================================================
import pandas as pd, numpy as np, os, re

# ---------- 0. Paths ---------------------------------------------------------#
data_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\CSV TIF"
save_dir = r"C:\Users\Navar\OneDrive\Documentos\USA\MACHINE LEARNING"
os.makedirs(save_dir, exist_ok=True)
os.chdir(data_dir)

# ---------- 1. Helpers --------------------------------------------------------#
def read_flexible(path, ncols, cols):
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 1:
        tmp = df.iloc[:, 0].str.split(",", expand=True)
        if tmp.shape[1] < ncols:
            tmp = df.iloc[:, 0].str.split(r"\s+", expand=True)
        df = tmp

    df = df.iloc[:, :ncols]
    df.columns = cols

    # --- NEW: drop any accidental header rows inside the file ---
    df = df[~df[cols[0]].str.lower().str.contains("state")]

    return df


def clean_state(df, fname):
    cand = [c for c in df.columns if c.strip().lower() in ("state","name","state_name","st")]
    if cand:
        df.rename(columns={cand[0]:"State"}, inplace=True)
    else:
        df.reset_index(inplace=True)
        df.rename(columns={"index":"State"}, inplace=True)
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

# ---------- 2. Load & coerce layers -----------------------------------------#
elec = clean_state(
    pd.read_excel("ElectricityUSA.xlsx").rename(
        columns=lambda x: "Industrial_Price_C_per_kWh" if "Industrial" in x else x
    ), "ElectricityUSA.xlsx"
)
ghi = clean_state(read_flexible("GHIUSA.csv", 6,
    ["State","GHI_min","GHI_max","GHI_mean","GHI_median","GHI_std"]), "GHIUSA.csv")
wind = clean_state(read_flexible("WINDUSA.csv", 6,
    ["State","Wind_min","Wind_max","Wind_mean","Wind_median","Wind_std"]), "WINDUSA.csv")

land = clean_state(pd.read_csv("USLAND.csv"), "USLAND.csv")
land.columns = land.columns.str.strip()  # 🔧 STRIP WHITESPACE
print("USLAND columns:", land.columns.tolist())  # 

infra = clean_state(
    pd.read_excel("INFRAUSA.xlsx")[ ["State","Energy","Water","Dams"] ],
    "INFRAUSA.xlsx"
)
# --- Inspect raw water file ---
raw_water = pd.read_csv("USWATER.csv", header=None)
print("RAW water preview:")
print(raw_water.head(5))

# Re-read correctly
water = clean_state(pd.read_csv("USWATER.csv"), "USWATER.csv")
print("Columns before renaming:", water.columns.tolist())

# Ensure consistent format
water.columns = water.columns.str.strip().str.lower()
water.rename(columns={
    "state": "State",              # ✅ ADD THIS LINE
    "area_km2": "Area_km2",
    "runoff_km3": "Runoff_Km3",
    "runoff_mm": "Runoff_mm"
}, inplace=True)
print("Columns after renaming:", water.columns.tolist())


biomass = clean_state(
    pd.read_csv("USABIOMASS.csv").rename(
        columns={"NAME":"State","AGB_Mg_ha":"Biomass_AGB_Mg_ha"}
    )[["State","Biomass_AGB_Mg_ha"]], "USABIOMASS.csv"
)
co2 = clean_state(pd.read_csv("USACO2.csv"), "USACO2.csv")

# ---------- Load Grid Distance File ----------------------------------------#
grid = clean_state(
    read_flexible("STATEGRIDDISTANCE.csv",        # use the same helper you wrote
                  2,                              # expect 2 columns
                  ["State", "distance_to_grid_km"]),
    "STATEGRIDDISTANCE.csv"
)

# the first row is the quoted header inside the file itself → drop it
grid = grid[grid["State"].str.lower() != "state_name"]

grid = force_numeric(grid, ["distance_to_grid_km"])



ghi    = force_numeric(ghi, ghi.columns.drop("State"))
wind   = force_numeric(wind, wind.columns.drop("State"))
land = force_numeric(land, ["suitability"])
water = force_numeric(water, ["Runoff_Km3"])
# --- Estimate water cost ($/kg H2) based on inverse runoff ---
# Assume higher runoff = lower cost; use normalized inverse
water["water_cost_$kg"] = (1 / water["Runoff_Km3"])
water["water_cost_$kg"] = water["water_cost_$kg"] / water["water_cost_$kg"].max() * 0.002

infra  = force_numeric(infra, ["Energy","Water","Dams"])
biomass= force_numeric(biomass, ["Biomass_AGB_Mg_ha"])
co2cols= [c for c in co2.columns if re.fullmatch(r"\d{4}", str(c))]
co2    = force_numeric(co2, co2cols)

# ---------- CO₂ intensity ---------------------------------------------------
co2["CO2_avg_18_23"]   = co2[co2cols].loc[:, [c for c in co2cols if 2018<=int(c)<=2023]].mean(axis=1, skipna=True)
co2["CO2_avg_18_23"]   = co2["CO2_avg_18_23"].fillna(0)
co2["CO2_intensity_norm"] = co2["CO2_avg_18_23"] / co2["CO2_avg_18_23"].max()

co2["co2_surcharge_$kg"] = co2["CO2_intensity_norm"] * 0.05     # adjust factor if desired
co2 = co2[["State", "co2_surcharge_$kg"]]                       # keep only needed cols


# ---------- 2.2 Merge all ---------------------------------------------------#
df = elec
for frame, tag in [(ghi,"GHI"),(wind,"WIND"),(land,"LAND"),(infra,"INFRA"),
                   (water,"WATER"),(biomass,"BIOMASS"),(co2,"CO2"),(grid,"GRID")]:
    df = lmerge(df, frame, tag)
    
# ---- KEEP ONLY THE 48 CONTIGUOUS STATES -----------------------------------#
contig_48 = {
    'Alabama','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
    'Florida','Georgia','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
    'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota',
    'Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire',
    'New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio',
    'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
    'South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington',
    'West Virginia','Wisconsin','Wyoming'
}
df = df[df["State"].isin(contig_48)].copy()   # now rows = 48, guaranteed
assert len(df) == 48, "Row count drifted—check input files."   # ← NEW LINE
print(f"🗂  Filtered to {len(df)} contiguous states")

# --- Add CAPEX multipliers for land and infrastructure ---
df["capex_mult_land"] = 1 + (1 - df["suitability"]) * 0.25
df["capex_mult_infra"] = 1 + (1 - df["Energy"] / df["Energy"].max()) * 0.25


# --- Add CO₂ surcharge (placeholder or computed) ---------------------------
if "co2_surcharge_$kg" not in df.columns:
    df["co2_surcharge_$kg"] = 0.0          # create column if it doesn't exist

df["co2_surcharge_$kg"] = df["co2_surcharge_$kg"].fillna(0.0)


# ---------- 3. Resource multipliers ----------------------------------------#
ghi_min, ghi_max = df["GHI_mean"].min(), df["GHI_mean"].max()
wind_min, wind_max = df["Wind_mean"].min(), df["Wind_mean"].max()

df["solar_norm"] = (df["GHI_mean"] - ghi_min) / (ghi_max - ghi_min)
df["wind_norm"] = (df["Wind_mean"] - wind_min) / (wind_max - wind_min)

df["resource_idx"] = 0.6 * df["solar_norm"] + 0.4 * df["wind_norm"]
df["CF_state"] = 0.6 + 0.3 * df["resource_idx"]


# ---------- 3.1 Add Transport & Storage Costs -----------------------------#
df["Transport_km"] = df["distance_to_grid_km"]  # Use real grid distance
df["Transport_km"] = df["Transport_km"].fillna(200)  # Fallback for missing values
df["transport_cost_$kg"] = df["Transport_km"] * 0.0005
df["storage_cost_$kg"] = 0.50


# ---------- 4. SOEC DOE parameters -----------------------------------------#
disc, life   = 0.08, 20
crf = (disc*(1+disc)**life)/((1+disc)**life - 1)
opex_ratio  = 0.05

cap22, cap26, cap31 = 2500, 500, 200    # $/kW
eff22, eff26, eff31 = 38, 36, 35        # kWh/kg (electrical)
eff_floor          = 34

def cap_eff(year):
    if year <= 2026:
        cap = np.interp(year, [2022,2026], [cap22,cap26])
        eff = np.interp(year, [2022,2026], [eff22,eff26])
    elif year <= 2031:
        cap = np.interp(year, [2026,2031], [cap26,cap31])
        eff = np.interp(year, [2026,2031], [eff26,eff31])
    else:
        cap = cap31 * (0.96**((year-2031)/5))
        eff = max(eff_floor, eff31*(1-0.002*(year-2031)))
    return cap, eff

# ---------- 5. Projection --------------------------------------------------#
years = [2022, 2030, 2040, 2050, 2060]
rows  = []

for _, r in df.iterrows():
    s, p0, cf, cmL, cmI, wc, cs, tc, sc = (
        r["State"],
        r["Industrial_Price_C_per_kWh"] / 100,   # convert ¢/kWh → $/kWh
        r["CF_state"], r["capex_mult_land"], r["capex_mult_infra"],
        r["water_cost_$kg"], r["co2_surcharge_$kg"],
        r["transport_cost_$kg"], r["storage_cost_$kg"]
    )

    for y in years:
        cap_nom, eff = cap_eff(y)
        cap  = cap_nom * cmL * cmI
        p_e  = p0 * (1 + (y - 2022) / (2060 - 2022) * (-0.30))
        kgH2 = cf * 8760 / eff
        opex_term = cap * opex_ratio / kgH2               # corrected OPEX

        lcoh = (
            cap * crf / kgH2 +            # CAPEX (annuitised)
            opex_term        +            # fixed OPEX
            p_e * eff         +           # electricity
            wc + cs + tc + sc             # water, CO₂, transport, storage
        )

        rows.append([s, y, cf, cap, eff, p_e, wc, cs, tc, sc, lcoh])

# ----------------------------- 6. Export ------------------------------------#
df_out = pd.DataFrame(rows, columns=[
    "State", "Year", "CF", "CAPEX_$/kW", "Efficiency_kWh_kg",
    "Electricity_$/kWh", "Water_$kg", "CO2_$kg",
    "Transport_$kg", "Storage_$kg", "LCOH_SOEC_$kg"
])

df_out = df_out.round(4)                    # round once, at export
outfile = os.path.join(save_dir, "USASOEC.csv")
df_out.to_csv(outfile, index=False)

print(df_out.head(10))
print(f"\n✅ Saved → {outfile}")


