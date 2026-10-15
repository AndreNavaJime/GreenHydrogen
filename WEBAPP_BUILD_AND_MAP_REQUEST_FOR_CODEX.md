# Web-app build verification and optional map handoff for Codex

Work only on branch:

`revised-v3-webapp`

Do not merge to `main`.

## 1. Sync branch

From the local clone:

```powershell
git fetch origin
git checkout revised-v3-webapp
git pull --ff-only origin revised-v3-webapp
```

## 2. Build verification

Run:

```powershell
npm install
npm run build
```

Requirements:

- do not change any scientific CSV under `public/data/canonical/`;
- report any TypeScript, Next.js, hydration, or import error exactly;
- if a code-only fix is required for build success, make the smallest possible fix;
- do not change scientific values;
- retain `USMACHINE.py` and legacy files unchanged.

If `npm install` creates `package-lock.json`, retain it.

## 3. Canonical-data checks

Confirm:

- `deterministic_state_year.csv`: 720 data rows;
- 48 unique states;
- 3 technologies: AWE, PEM, O-SOEC;
- 5 stored year labels: 2022, 2030, 2040, 2050, 2060;
- `state_rank_robustness_2022.csv`: 48 rows;
- all files listed in `MIGRATION_MANIFEST.json` match their recorded SHA-256 hashes;
- the app loads data only from `public/data/canonical/`, not legacy root CSVs.

## 4. Create a simplified 48-state GeoJSON for the web map

Use the existing Census/TIGER state shapefile already present in this repository/local clone:

`SHAPEFILE/tl_2022_us_state.shp`

with its companion DBF/SHX/PRJ files.

Create:

`public/data/us_states_contiguous_48.geojson`

Requirements:

- derive the accepted state-name set directly from `public/data/canonical/deterministic_state_year.csv`;
- retain exactly those 48 state features, excluding Alaska, Hawaii, District of Columbia, Puerto Rico, and territories;
- preserve only minimal properties needed by the app, preferably `NAME` and `STUSPS`;
- output CRS EPSG:4326;
- simplify geometry for browser use without materially changing state topology;
- recommended method: project to EPSG:5070, simplify with topology preserved, then return to EPSG:4326;
- choose a tolerance that keeps the final GeoJSON reasonably lightweight while visually faithful;
- verify exactly 48 features and exact 1:1 name matching with the canonical state set;
- record source and simplification settings in `public/data/GEOJSON_PROVENANCE.md`.

Do not derive any scientific value from the shapefile. It is cartographic only.

## 5. Commit and push

Commit only:

- any minimal build-fix files if required;
- `package-lock.json` if generated;
- `public/data/us_states_contiguous_48.geojson`;
- `public/data/GEOJSON_PROVENANCE.md`;
- optional concise build-QA note.

Push to `revised-v3-webapp`.

Report:

- build PASS/FAIL;
- commit SHA;
- GeoJSON feature count and file size;
- simplification tolerance and CRS workflow;
- exact state-name validation result;
- any code changes required to make the build pass.

Do not merge to `main`.
