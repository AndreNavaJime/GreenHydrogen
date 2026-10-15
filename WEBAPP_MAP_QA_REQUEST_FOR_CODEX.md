# Final map/build QA for revised-v3 web app

Work only on branch:

`revised-v3-webapp`

Do not merge to `main`.

## 1. Sync

```powershell
git fetch origin
git checkout revised-v3-webapp
git pull --ff-only origin revised-v3-webapp
```

## 2. Build

Run:

```powershell
npm install
npm run build
```

Report build PASS/FAIL and any exact error.

Do not change any scientific CSV.

## 3. Runtime smoke test

Run the app locally:

```powershell
npm run dev
```

Verify in a browser:

- home page loads;
- the deterministic cohort tabs work;
- the 48-state map renders;
- map clicks update the selected state;
- the state dropdown and year selector work;
- all three technology cards populate from `public/data/canonical/deterministic_state_year.csv`;
- CAPEX/OPEX/electricity/heat/replacement component bars render;
- DOE CAPEX-envelope card is available for 2022 and explicitly not applied to later cohorts;
- observed-2025 electricity scenario values populate;
- observed-2024 natural-gas scenario values populate;
- canonical CSV and manifest download links work;
- no legacy root CSV is requested by the browser;
- browser console has no uncaught error or hydration error.

Spot-check at least:

1. Alabama 2022:
   - deterministic winner AWE;
2. California 2022:
   - deterministic winner AWE;
   - observed-2025 electricity case winner O-SOEC;
3. Rhode Island 2022:
   - deterministic winner O-SOEC;
   - DOE CAPEX classification robust;
4. Texas 2030:
   - deterministic winner O-SOEC.

## 4. Map QA

Verify:

- exactly 48 state paths render;
- Rhode Island is clickable/selectable despite its small geometry;
- map colors match deterministic winners for the selected year;
- 2022 shows 47 AWE and 1 O-SOEC state;
- 2030 and 2040/2050/2060 show 48 O-SOEC states;
- no Alaska/Hawaii/DC/territory feature appears.

If the 5-km simplified Rhode Island polygon is too small to click reliably, add the smallest cartographic-only accessibility treatment needed (for example a larger transparent hit target or tiny callout) without changing scientific data or map classification.

## 5. Commit only if required

If build/runtime QA reveals a code-only issue, make the smallest possible fix and rerun the checks.

Commit only code/UI fixes or build metadata. Do not alter scientific CSVs or GeoJSON source features.

Push to `revised-v3-webapp`.

Report:

- final build result;
- runtime/browser QA result;
- any fixes made;
- final commit SHA;
- confirmation that scientific CSV hashes remain unchanged.

Do not merge to `main`.
