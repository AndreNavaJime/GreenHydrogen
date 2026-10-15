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

The first revised web-app version is seeded with manuscript-approved summary results:

- deterministic v3 state-distribution summaries;
- DOE 2022 installed-CAPEX envelope ranking robustness;
- observed 2025 electricity-price sensitivity;
- observed 2024 natural-gas-price sensitivity;
- DOE 2024 benchmark comparison.

The full 48-state explorer/map will be connected only after the accepted canonical v3 state and scenario files are imported. Legacy root CSVs must **not** be used as substitutes.

See `data/README.md`.

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
