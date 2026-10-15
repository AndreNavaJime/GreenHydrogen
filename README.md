# U.S. Electrolysis Cost Explorer

This repository is transitioning from the historical Streamlit/ML dashboard to a lighter web application aligned with the **revised deterministic v3** scientific analysis.

## Active revised interface

The new app is a Next.js interface for the accepted manuscript architecture:

- grid-connected, plant-gate LCOH;
- 48 contiguous U.S. states;
- AWE (liquid alkaline), PEM, and O-SOEC;
- real 2022 USD;
- 8% real WACC;
- 20-year project life;
- 90% utilization;
- DOE technology assumptions;
- EIA state electricity and natural-gas price inputs;
- source-backed ranking robustness rather than probabilistic Monte Carlo;
- DOE 2024 published-model consistency benchmark.

The revised baseline **does not claim renewable matching, emissions intensity, or green-hydrogen certification**.

### Run the new app locally

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Current web-app data status

The revised branch now contains the migrated, hash-verified canonical data release under `public/data/canonical/`:

- 720-row deterministic v3 state/year/technology table;
- state technology-ordering tables;
- DOE 2022 installed-CAPEX envelope and break-even outputs;
- observed 2025 electricity-price sensitivity;
- observed 2024 natural-gas-price sensitivity;
- DOE 2024 benchmark comparison;
- public provenance manifest with SHA-256 hashes and source-relative paths.

The web app reads these accepted CSVs directly. It includes a full 48-state selector, technology cost decomposition, scenario comparisons, and direct CSV/manifest downloads. Legacy root CSVs are **not** used as substitutes.

See `data/README.md` and `public/data/canonical/README.md`.

## Historical Streamlit dashboard

`USMACHINE.py` and the root-level historical TEA/ML files are retained for provenance. They belong to the earlier analysis and are **not the active scientific interface for the revised manuscript**.

The historical implementation includes Streamlit, ML surrogate models, SHAP, legacy TEA CSVs, rank-flip tables, resource proxies, and other elements that were removed from the revised scientific contribution.

Do not use the historical dashboard outputs as revised-manuscript results.

## Scientific transition

The revised paper now centers on:

1. a physically consistent deterministic plant-gate TEA;
2. state-resolved technology ordering;
3. source-backed DOE CAPEX-envelope robustness;
4. observed electricity- and natural-gas-price robustness;
5. published DOE benchmark consistency.

Probabilistic Monte Carlo, ML/SHAP, Cambium, delivered-cost overlays, arbitrary capacity-factor equations, CO₂ surcharges, and land/infrastructure CAPEX multipliers are not part of the active revised analysis.

## Repository branch

The new application is being developed on:

`revised-v3-webapp`

The historical `main` branch remains available for provenance until the revised application is reviewed and merged.
