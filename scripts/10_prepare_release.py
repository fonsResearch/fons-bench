"""Prepare the public release bundle: parquet, calibration, figures, and Zenodo metadata.

    uv run python scripts/10_prepare_release.py

Writes release/ with everything needed for a DOI deposit. Does not upload: depositing is a
manual, credentialed step, and the metadata below is what to paste into it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REL = ROOT / "release"

FILES = ["master.parquet", "gate1_report.md", "gate2_report.md", "gate3_report.md",
         "kill_condition.md", "data_card.md", "phase4_report.md",
         "aurc_table.csv", "conditional_coverage.csv", "alpha_sweep.csv",
         "phase4_matched_abstention.csv", "phase4_leakage.csv", "rmsd_verification_8c3u.csv"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    REL.mkdir(exist_ok=True)
    (REL / "figures").mkdir(exist_ok=True)
    manifest = {}
    for name in FILES:
        src = RESULTS / name
        if not src.exists():
            print(f"missing, skipped: {name}"); continue
        dst = REL / (src.readlink().name if src.is_symlink() else name)
        shutil.copy2(src.resolve(), dst)
        manifest[dst.name] = {"bytes": dst.stat().st_size, "sha256": sha256(dst)}
    for fig in sorted((RESULTS / "figures").glob("*.png")):
        shutil.copy2(fig, REL / "figures" / fig.name)
        manifest[f"figures/{fig.name}"] = {"bytes": fig.stat().st_size, "sha256": sha256(fig)}
    art = ROOT / "src" / "fons_bench" / "artifacts" / "calibration.json"
    shutil.copy2(art, REL / "calibration.json")
    manifest["calibration.json"] = {"bytes": art.stat().st_size, "sha256": sha256(art)}

    meta = {
        "title": "fons-bench: conditional coverage of protein-ligand co-folding confidence signals under structural distribution shift",
        "upload_type": "dataset",
        "publication_date": date.today().isoformat(),
        "version": "1.0.0",
        "description": (
            "Per-prediction results table and calibration artifacts for fons-bench. One row per "
            "(ligand, model, seed, sample) for seven co-folding methods over protein-ligand systems "
            "released after each model's training cutoff, with outcomes, confidence signals, and "
            "similarity-to-training recomputed at each model's own cutoff date. Includes the "
            "conformal calibration thresholds, the pre-registered kill condition, and every report."
        ),
        "license": "cc-by-nc-4.0",
        "license_note": (
            "Non-commercial. Rows for af3 and af3_no_template derive from AlphaFold 3 output and are "
            "subject to Google's AlphaFold 3 Output Terms of Use "
            "(https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md). "
            "Redistribution must carry that notice."
        ),
        "keywords": ["conformal prediction", "selective prediction", "distribution shift",
                     "protein-ligand co-folding", "structure prediction", "calibration",
                     "uncertainty quantification", "drug discovery"],
        "related_identifiers": [
            {"relation": "isDerivedFrom", "identifier": "10.5281/zenodo.14794785",
             "resource_type": "dataset", "scheme": "doi"},
            {"relation": "isSupplementTo", "identifier": "10.1101/2025.02.03.636309",
             "resource_type": "publication-preprint", "scheme": "doi"},
        ],
        "notes": "Built by `python -m fons_bench.cli build-table`; rebuild is byte-identical.",
        "files": manifest,
    }
    (REL / "zenodo.json").write_text(json.dumps(meta, indent=1))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"release/ prepared: {len(manifest)} files, {total/1e6:.1f} MB")
    print(f"zenodo metadata: {REL / 'zenodo.json'}")
    for k in sorted(manifest):
        print(f"  {k:44s} {manifest[k]['bytes']/1e6:8.2f} MB  {manifest[k]['sha256'][:12]}")


if __name__ == "__main__":
    main()
