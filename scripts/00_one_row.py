"""First task (spec section 11): produce ONE row of master.parquet end to end.

    uv run python scripts/00_one_row.py <SYSTEM_ID> [--model smina] [--seed 0]

Not the pipeline. One system, one model, one seed, printed as a row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

from fons_bench.data.reference import build_from_rcsb, fetch_entry_metadata, load_reference
from fons_bench.predict.smina import SminaPredictor
from fons_bench.score.rmsd import symmetric_rmsd
from fons_bench.score.strain import mmff_strain
from fons_bench.score.validity import posebusters_flags
from fons_bench.shift.ligand import fps_from_smiles, max_tanimoto

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# A deliberately small, hand-picked stand-in for "the training set": common PDB ligands.
# Replaced in Phase 1 by the real pre-cutoff PDB ligand set. Kept here so the row is honest
# about what lig_tanimoto_max was computed against.
TRAIN_SAMPLE_CCDS = ["ATP", "ADP", "NAD", "FAD", "HEM", "SAM", "GDP", "STI", "ACT", "GOL",
                     "EDO", "SO4", "PO4", "NAG", "GLC", "ZN", "MG", "COA", "PLP", "FMN",
                     "IMD", "BEZ", "TRS", "DMS", "MPD", "PEG", "CIT", "MLI", "NDP", "ANP"]


def training_sample_smiles(cache: Path = RAW / "train_ligand_sample.json") -> dict[str, str]:
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for ccd in TRAIN_SAMPLE_CCDS:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd}", timeout=30)
        if not r.ok:
            continue
        smi = r.json().get("rcsb_chem_comp_descriptor", {}).get("SMILES_stereo")
        if smi:
            out[ccd] = smi
    cache.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system_id")
    ap.add_argument("--model", default="smina", choices=["smina"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set-dir", default=str(RAW / "posebusters_benchmark_set"))
    args = ap.parse_args()

    sys_dir = Path(args.set_dir) / args.system_id
    if not sys_dir.exists():
        # archive not available: build the same layout from RCSB (provenance recorded below)
        pdb, _, ccd = args.system_id.partition("_")
        sys_dir = build_from_rcsb(pdb, ccd, Path(args.set_dir), RAW / "rcsb")
    ref = load_reference(sys_dir)
    meta = fetch_entry_metadata(ref.pdb_code)

    pred_dir = INTERIM / "predictions" / args.model / ref.system_id / f"seed{args.seed}"
    predictor = SminaPredictor(exe=ROOT / ".envs" / "smina" / "bin" / "smina")
    pose = predictor.predict(ref, seed=args.seed, out_dir=pred_dir)[0]
    pose_sdf = pred_dir / "pose.sdf"

    rmsd = symmetric_rmsd(pose.ligand, ref.ligand)
    flags = posebusters_flags(pose_sdf, ref.ligand_sdf, ref.protein_pdb)
    strain = mmff_strain(pose.ligand)
    train = training_sample_smiles()
    tan = max_tanimoto(ref.ligand, fps_from_smiles(train.values()))

    rmsd_lt_2 = bool(rmsd < 2.0)
    pb_valid = flags["pb_valid"]
    row = {
        "system_id": ref.system_id, "pdb_code": ref.pdb_code, "ligand_ccd": ref.ligand_ccd,
        "uniprot": meta["uniprot"], "pfam": meta["pfam"], "deposition_date": meta["deposition_date"],
        "dataset": "posebusters", "model": args.model, "seed": args.seed, "pred_source": pose.source,
        "rmsd": rmsd, "rmsd_lt_2": rmsd_lt_2, "pb_valid": pb_valid,
        **{k: v for k, v in flags.items() if k != "pb_valid"},
        "strain_kcal": strain,
        "correct": (rmsd_lt_2 and pb_valid) if pb_valid is not None else None,
        "ptm": None, "iptm": None, "plddt_ligand": None, "pae_interface": None,
        "seed_spread": None, "consensus_rmsd": None,
        "dock_score": pose.confidence.get("dock_score"),
        "lig_tanimoto_max": tan, "pocket_identity_max": None, "family_seen": None,
        "after_cutoff": None, "shift_bin": None,
    }
    df = pd.DataFrame([row])
    out = ROOT / "results" / "one_row.parquet"
    df.to_parquet(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.T.to_string())
    print(f"\nwrote {out}  ({len(df.columns)} columns)")
    print(f"training ligand sample: {len(train)} CCDs -> {', '.join(train)}")


if __name__ == "__main__":
    main()
