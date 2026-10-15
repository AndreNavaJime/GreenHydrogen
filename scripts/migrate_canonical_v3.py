#!/usr/bin/env python3
"""Copy accepted revised-v3 scientific outputs into the web-app data directory.

Usage:
    python scripts/migrate_canonical_v3.py --usa-root "C:/Users/Navar/OneDrive/Documentos/USA"

The script copies only accepted canonical/audit CSVs and writes a SHA-256 manifest.
It does not read or reuse legacy root-level USAALKALINE/USAPEM/USASOEC files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


FILES = {
    "deterministic_state_year.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3/DETERMINISTIC_STATE_YEAR.csv",
    "national_summary.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3/NATIONAL_SUMMARY.csv",
    "technology_ordering_state_year.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3/TECHNOLOGY_ORDERING_STATE_YEAR.csv",
    "technology_ordering_summary.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3/TECHNOLOGY_ORDERING_SUMMARY.csv",
    "state_rank_robustness_2022.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1/STATE_RANK_ROBUSTNESS_2022.csv",
    "doe_capex_envelope_2022.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1/DOE_CAPEX_ENVELOPE_2022.csv",
    "pairwise_capex_break_even_2022.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1/PAIRWISE_CAPEX_BREAK_EVEN_2022.csv",
    "eia_2025_observed_price_sensitivity.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1/EIA_2025_OBSERVED_PRICE_SENSITIVITY.csv",
    "scenario_ordering_summary.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_scenario_robustness_v1/SCENARIO_ORDERING_SUMMARY.csv",
    "o_soec_2024_ng_price_sensitivity.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_observed_ng_2024_v1/O_SOEC_2024_NG_PRICE_SENSITIVITY.csv",
    "ng_2024_national_change_summary.csv": "CANONICAL_OUTPUTS/TEA/revised_deterministic_v3_observed_ng_2024_v1/NATIONAL_CHANGE_SUMMARY.csv",
    "doe_v3_benchmark_comparison.csv": "DATA_AUDITS/STEP7A_DOE_V3_BENCHMARK/DOE_V3_BENCHMARK_COMPARISON.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usa-root", required=True, help="Root of the local USA project")
    parser.add_argument(
        "--dest",
        default="public/data/canonical",
        help="Destination inside the GitHub repository",
    )
    args = parser.parse_args()

    usa_root = Path(args.usa_root).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    dest = (repo_root / args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    missing = []
    for rel in FILES.values():
        if not (usa_root / rel).is_file():
            missing.append(rel)

    if missing:
        raise SystemExit(
            "Missing accepted source files:\n- " + "\n- ".join(missing)
        )

    manifest = {
        "scientific_release": "revised_deterministic_v3",
        "source_root": "LOCAL_USA_PROJECT_ROOT",
        "destination": args.dest.replace("\\\\", "/"),
        "note": "Absolute local filesystem paths intentionally omitted from public provenance.",
        "files": {},
    }

    for out_name, rel in FILES.items():
        src = usa_root / rel
        dst = dest / out_name
        shutil.copy2(src, dst)
        manifest["files"][out_name] = {
            "source_relative_path": rel.replace("\\", "/"),
            "sha256": sha256(dst),
            "rows": row_count(dst),
            "bytes": dst.stat().st_size,
        }

    manifest_path = dest / "MIGRATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Scientific guardrails that should fail loudly.
    if manifest["files"]["deterministic_state_year.csv"]["rows"] != 720:
        raise SystemExit("deterministic_state_year.csv must contain exactly 720 data rows")
    if manifest["files"]["state_rank_robustness_2022.csv"]["rows"] != 48:
        raise SystemExit("state_rank_robustness_2022.csv must contain exactly 48 data rows")

    print(f"Migrated {len(FILES)} accepted files to {dest}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
