# Revised v3 web-app data

The active web interface uses only values accepted in the revised manuscript workflow.

## Canonical browser data

The migrated source files now live under:

`public/data/canonical/`

They include:

- deterministic v3 state/year/technology results;
- technology-ordering outputs;
- DOE 2022 CAPEX-envelope robustness and break-even tables;
- observed 2025 electricity-price sensitivity;
- observed 2024 natural-gas-price sensitivity;
- DOE 2024 consistency benchmark;
- `MIGRATION_MANIFEST.json` with SHA-256 hashes and source-relative provenance.

The state explorer loads the canonical CSVs directly in the browser.

## Legacy data boundary

The historical root-level `USAALKALINE.csv`, `USAPEM.csv`, `USASOEC.csv`, rank-flip files, ML outputs and Streamlit pipeline belong to the rejected/pre-review analysis and are **not** used by the revised web app.

Do not substitute those files for the canonical revised-v3 release.
