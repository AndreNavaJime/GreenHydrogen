# Local data handoff for revised-v3 web app

Run this from the local clone of `AndreNavaJime/GreenHydrogen` on branch
`revised-v3-webapp`.

The accepted USA project root is:

`C:/Users/Navar/OneDrive/Documentos/USA`

## Task

1. Fetch/checkout `revised-v3-webapp`.
2. Run:

```powershell
python scripts/migrate_canonical_v3.py --usa-root "C:/Users/Navar/OneDrive/Documentos/USA"
```

3. Confirm that:
   - `public/data/canonical/deterministic_state_year.csv` has 720 data rows;
   - `public/data/canonical/state_rank_robustness_2022.csv` has 48 data rows;
   - `MIGRATION_MANIFEST.json` exists;
   - every source file listed by the script was found;
   - no root-level legacy `USAALKALINE.csv`, `USAPEM.csv`, `USASOEC.csv`, ML, SHAP, Monte Carlo, or Cambium output was used.
4. Do not alter any canonical source file under the USA project.
5. Commit only the generated files under `public/data/canonical/` to branch
   `revised-v3-webapp`.
6. Push that branch.
7. Report:
   - commit SHA;
   - migrated file list;
   - row counts;
   - SHA-256 values from `MIGRATION_MANIFEST.json`;
   - any missing file or schema error.

Do not merge to `main`.
