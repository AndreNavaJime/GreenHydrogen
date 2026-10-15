# Revised v3 web-app data

The current web interface is seeded only with values already accepted in the revised manuscript workflow:

- deterministic v3 national state-distribution summaries;
- Step 5A DOE CAPEX-envelope ranking robustness;
- Step 5A observed-2025 electricity-price result;
- Step 5B observed-2024 natural-gas-price result;
- Step 7A DOE consistency benchmark.

The historical root-level `USAALKALINE.csv`, `USAPEM.csv`, `USASOEC.csv`, rank-flip files, ML outputs and Streamlit data pipeline belong to the rejected/pre-review analysis and are **not** used by the revised web app.

## Next data import

To enable the full 48-state explorer/map, import the accepted canonical files from:

`CANONICAL_OUTPUTS/TEA/revised_deterministic_v3/`

and the accepted Step 5A/5B scenario outputs into versioned JSON/CSV under this directory. Do not substitute the legacy root CSVs.
