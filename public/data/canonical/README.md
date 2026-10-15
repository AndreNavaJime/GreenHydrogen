# Canonical revised-v3 data staging

This directory is populated by:

```bash
python scripts/migrate_canonical_v3.py --usa-root "C:/Users/Navar/OneDrive/Documentos/USA"
```

Only accepted revised-manuscript outputs are permitted here.

Expected source families:

- `CANONICAL_OUTPUTS/TEA/revised_deterministic_v3`
- `CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1`
- `CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_observed_ng_2024_v1`
- `DATA_AUDITS/STEP7A_DOE_V3_BENCHMARK`

The migration script verifies the complete source set before copying and writes
`MIGRATION_MANIFEST.json` with SHA-256 hashes, row counts, and source-relative paths.

Do **not** populate this directory from legacy root-level TEA/ML CSVs.
