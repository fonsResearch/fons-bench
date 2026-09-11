"""Assemble results/master_v{N}.parquet from the Runs N' Poses release in data/raw/.

One row per (group_key, model, seed, sample). No inference, no structure files: every
outcome and confidence column is taken from the release CSVs; shift columns are recomputed
from all_similarity_scores.parquet at each model's own training cutoff (config/models.yaml).

Row collapse rule. The release CSVs are an outer product of two ligand-copy assignments
(lDDT-PLI side x RMSD side) for systems with several copies of the same ligand, so a pose can
appear several times. We follow the authors' own convention (figures.ipynb):
    sort by (lddt_pli desc, rmsd asc), keep the first row per (group_key, seed, sample).
For methods whose CSV records the RMSD-side reference chain we also compute how often this
differs from the strict rule (RMSD-side reference chain == ligand_instance_chain) and report it.

PoseBusters. The release ran PoseBusters on ONE pose per group_key (the top-ranked one), so
pb_* and `correct` are non-null only for that pose. We join on (system_id, seed, sample,
ligand_instance_chain); we do not propagate one pose's flags to the other 24.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fons_bench.shift.binning import assign_bins, merge_small_bins
from fons_bench.shift.similarity import similarity_at_cutoff
from fons_bench.signals import DISPERSION_SIGNALS, dispersion_over_poses

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "runs_n_poses"
CONFIG = ROOT / "config"
RESULTS = ROOT / "results"

# PoseBusters checks that constitute validity. The three *_loaded flags are bookkeeping
# (could the files be read) and are kept as columns but not folded into pb_valid.
PB_LOAD_FLAGS = ["mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"]
PB_CHECKS = [
    "sanitization", "inchi_convertible", "all_atoms_connected", "molecular_formula",
    "molecular_bonds", "double_bond_stereochemistry", "tetrahedral_chirality", "bond_lengths",
    "bond_angles", "internal_steric_clash", "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy",
    "protein-ligand_maximum_distance", "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors", "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters", "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors", "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]
PB_FILE_ALIAS = {"af3_no_template": "af3_no_templ"}
IPTM_COLS = [f"{a}_chain_iptm_{s}_{m}" for a in ("prot_lig", "lig_prot") for s in ("average", "min", "max") for m in ("rmsd", "lddt_pli")]
GRAIN = ["group_key", "model", "seed", "sample"]


def _pb_col(name: str) -> str:
    return "pb_" + name.replace("-", "_")


def load_config():
    models = yaml.safe_load((CONFIG / "models.yaml").read_text())["models"]
    shift = yaml.safe_load((CONFIG / "shift.yaml").read_text())
    return models, shift


def _nullable_bool(s: pd.Series) -> pd.Series:
    """PoseBusters columns arrive as bool, float 0/1 or object; map to pandas nullable boolean."""
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    v = s.astype(object)
    out[v.isin([True, 1, 1.0, "True", "true"])] = True
    out[v.isin([False, 0, 0.0, "False", "false"])] = False
    return out


def load_method(method: str, raw: Path = RAW) -> tuple[pd.DataFrame, dict]:
    """Predictions for one method, collapsed to one row per pose, with PoseBusters joined."""
    d = pd.read_csv(raw / "predictions" / "predictions" / f"{method}.csv", low_memory=False)
    diag: dict = {"raw_rows": int(len(d))}
    for c in ("seed", "sample"):
        if c not in d:
            d[c] = 1                     # single deterministic run (rfaa); the release's PoseBusters file uses 1
    for c in ["ranking_score", "pred_pocket_f1", *IPTM_COLS, "ligand_instance_chain_rmsd", "ligand_ccd_code"]:
        if c not in d:
            d[c] = np.nan
    d["group_key"] = d["target"] + "__" + d["ligand_instance_chain"].astype(str)

    # strict-rule diagnostic where available (before collapsing)
    has_strict = d["ligand_instance_chain_rmsd"].notna().any()
    if has_strict:
        d["_strict"] = d["ligand_instance_chain_rmsd"] == d["ligand_instance_chain"]

    d = d.sort_values(["lddt_pli", "rmsd"], ascending=[False, True], kind="mergesort")
    d = d.groupby(["group_key", "seed", "sample"], sort=False).head(1).reset_index(drop=True)
    diag["poses"] = int(len(d))
    if has_strict:
        diag["authors_rule_row_not_strict_frac"] = float((~d["_strict"]).mean())
        d = d.drop(columns="_strict")

    # PoseBusters, joined only to the pose it was run on
    pb_path = raw / "posebusters_results" / "posebusters_results" / f"{PB_FILE_ALIAS.get(method, method)}.csv"
    if pb_path.exists():
        pb = pd.read_csv(pb_path, low_memory=False)
        for c in ("seed", "sample"):
            if c not in pb:
                pb[c] = 1
        pb = pb.rename(columns={"ligand_chain": "ligand_instance_chain", "system_id": "target"})
        keep = ["target", "seed", "sample", "ligand_instance_chain", *PB_LOAD_FLAGS, *PB_CHECKS]
        pb = pb[[c for c in keep if c in pb]].drop_duplicates(["target", "seed", "sample", "ligand_instance_chain"])
        for c in PB_LOAD_FLAGS + PB_CHECKS:
            if c in pb:
                pb[c] = _nullable_bool(pb[c])
        pb = pb.rename(columns={c: _pb_col(c) for c in PB_LOAD_FLAGS + PB_CHECKS})
        d = d.merge(pb, on=["target", "seed", "sample", "ligand_instance_chain"], how="left")
        diag["pb_rows"] = int(len(pb)); diag["pb_joined"] = int(d[_pb_col(PB_CHECKS[0])].notna().sum())
    else:
        diag["pb_rows"] = 0; diag["pb_joined"] = 0
    for c in PB_LOAD_FLAGS + PB_CHECKS:
        if _pb_col(c) not in d:
            d[_pb_col(c)] = pd.Series(pd.NA, index=d.index, dtype="boolean")
    checks = d[[_pb_col(c) for c in PB_CHECKS]]
    known = checks.notna().all(axis=1)               # PoseBusters was run on this pose
    d["pb_valid"] = checks.fillna(False).all(axis=1).astype("boolean").where(known)   # NA unless every check is known
    d["pb_n_failed"] = (checks == False).sum(axis=1).astype("Int64").where(known)  # noqa: E712
    d["model"] = method
    return d, diag


def build(out_dir: Path = RESULTS, raw: Path = RAW, version: int | None = None) -> tuple[Path, dict]:
    models_cfg, shift_cfg = load_config()
    ann = pd.read_csv(raw / "annotations.csv", low_memory=False)
    ann["release_date"] = pd.to_datetime(ann["release_date"], errors="coerce")
    sim = pd.read_parquet(raw / "all_similarity_scores.parquet",
                          columns=["group_key", "target_release_date", "morgan_tanimoto", "sucos_shape_pocket_qcov"])

    parts, diags = [], {}
    for method in models_cfg:
        d, diag = load_method(method, raw)
        cutoff = models_cfg[method]["pdb_cutoff"]
        if cutoff == "unknown":
            d["morgan_tanimoto_cutoff"] = np.nan; d["sucos_shape_pocket_qcov_cutoff"] = np.nan
            d["n_targets_before_cutoff"] = pd.array([pd.NA] * len(d), dtype="Int64")
            d["after_cutoff"] = pd.array([pd.NA] * len(d), dtype="boolean")
            d["training_cutoff"] = pd.NaT
        else:
            s = similarity_at_cutoff(sim, cutoff, d["group_key"])
            d = d.merge(s, on="group_key", how="left")
            d["n_targets_before_cutoff"] = d["n_targets_before_cutoff"].astype("Int64")
            d["training_cutoff"] = pd.Timestamp(cutoff)
            d["after_cutoff"] = pd.array([pd.NA] * len(d), dtype="boolean")   # filled after annotations join
        diag["cutoff"] = str(cutoff)
        diags[method] = diag
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # annotations: identity, ligand properties, the release's own similarity columns
    ann_cols = ["group_key", "system_id", "entry_pdb_id", "ligand_ccd_code", "ligand_instance_chain",
                "ligand_smiles", "ligand_is_proper", "release_date", "cluster", "target_system",
                "target_release_date", "ligand_num_heavy_atoms", "ligand_num_rot_bonds",
                "ligand_molecular_weight", "num_protein_chains", "num_ligand_chains",
                "num_proper_ligand_chains", "morgan_tanimoto", "topological_tanimoto",
                "sucos_shape_pocket_qcov", "sucos_shape", "pocket_qcov", "pocket_fident",
                "protein_fident_qcov_weighted_max", "num_training_systems_with_similar_ccds"]
    a = ann[ann_cols].rename(columns={
        "ligand_ccd_code": "ligand_ccd", "ligand_is_proper": "ligand_is_proper_ann",
        "morgan_tanimoto": "morgan_tanimoto_rnp", "sucos_shape_pocket_qcov": "sucos_shape_pocket_qcov_rnp",
        "cluster": "rnp_cluster", "target_system": "rnp_closest_train_system",
        "target_release_date": "rnp_closest_train_release_date"})
    df = df.drop(columns=["ligand_ccd_code"], errors="ignore")
    df = df.merge(a.drop(columns=["system_id", "ligand_instance_chain"]), on="group_key", how="left")
    df["in_annotations"] = df["ligand_ccd"].notna() | df["release_date"].notna()
    df["ligand_is_proper"] = df["ligand_is_proper"].astype("boolean")
    df["after_cutoff"] = (df["release_date"] > df["training_cutoff"]).astype("boolean").where(df["training_cutoff"].notna() & df["release_date"].notna())

    # outcome. `correct` is the primary label, uniform across models (Gate 1 decision):
    # rmsd < 2 A. `correct_strict` adds PoseBusters validity and is NA wherever PoseBusters was
    # not run (24 of 25 poses; all of boltz2); it is used only for sensitivity analysis.
    df["rmsd_lt_2"] = (df["rmsd"] < 2.0).astype("boolean").where(df["rmsd"].notna())
    df["correct"] = df["rmsd_lt_2"]
    both = df["rmsd_lt_2"].notna() & df["pb_valid"].notna()
    df["correct_strict"] = (df["rmsd_lt_2"].fillna(False) & df["pb_valid"].fillna(False)).astype("boolean").where(both)
    df["pb_available"] = df["pb_valid"].notna()

    # analysis grain flag: the pose a practitioner gets, the model's own top-ranked one.
    # Ties broken by (seed, sample) order for determinism; rfaa has one pose per group_key.
    order = df.sort_values(["group_key", "model", "ranking_score", "seed", "sample"], ascending=[True, True, False, True, True], kind="mergesort")
    top_idx = order.groupby(["group_key", "model"], sort=False).head(1).index
    df["is_top_ranked"] = False
    df.loc[top_idx, "is_top_ranked"] = True
    df["n_poses"] = df.groupby(["group_key", "model"])["seed"].transform("size").astype("Int64")

    # dispersion of the native signals over the poses of each (group_key, model); computed from
    # signal columns only, attached to every pose of the group
    disp = dispersion_over_poses(df)
    df = df.merge(disp, on=["group_key", "model"], how="left")

    # confidence signals in spec names; every RnP variant retained under its own name
    df["iptm"] = df["lig_prot_chain_iptm_average_rmsd"]
    for c in ("ptm", "plddt_ligand", "pae_interface", "seed_spread", "consensus_rmsd", "dock_score", "strain_kcal"):
        df[c] = np.nan
    df["device"] = None
    df["pred_source"] = "runs_n_poses"
    df["dataset"] = "runs_n_poses"
    df["pdb_code"] = df["group_key"].str[:4].str.upper()
    df["system_id"] = df["target"]
    df["uniprot"] = None; df["pfam"] = None; df["deposition_date"] = pd.NaT

    # shift bins (edges from config; merged if underpopulated, on proper post-cutoff systems)
    df["shift_bin"] = None
    proper = df["ligand_is_proper"].fillna(False) & df["after_cutoff"].fillna(False)
    bins_used = {}
    for axis, col in (("ligand", "morgan_tanimoto_cutoff"), ("pocket", "sucos_shape_pocket_qcov_cutoff")):
        edges = shift_cfg[axis]["edges"]
        sub = df.loc[proper].drop_duplicates(["group_key", "model"])
        edges2, log = merge_small_bins(sub[col], edges, shift_cfg["min_systems"], sub["system_id"])
        df[f"{axis}_bin"] = assign_bins(df[col], edges2)
        bins_used[axis] = {"edges": edges2, "merges": log}
    df["shift_bin"] = df["ligand_bin"].astype(object) + "|" + df["pocket_bin"].astype(object)
    df.loc[df["ligand_bin"].isna() | df["pocket_bin"].isna(), "shift_bin"] = None

    front = ["group_key", "system_id", "pdb_code", "ligand_ccd", "ligand_instance_chain", "uniprot", "pfam",
             "deposition_date", "release_date", "dataset", "model", "seed", "sample", "pred_source", "device",
             "ligand_is_proper", "in_annotations", "training_cutoff", "after_cutoff",
             "rmsd", "rmsd_lt_2", "lddt_pli", "lddt_lp", "bb_rmsd", "pred_pocket_f1",
             "pb_valid", "pb_available", "pb_n_failed", *[_pb_col(c) for c in PB_CHECKS], *[_pb_col(c) for c in PB_LOAD_FLAGS],
             "strain_kcal", "correct", "correct_strict", "is_top_ranked", "n_poses",
             "ranking_score", "ptm", "iptm", *IPTM_COLS, *DISPERSION_SIGNALS, "plddt_ligand", "pae_interface", "seed_spread", "consensus_rmsd", "dock_score",
             "morgan_tanimoto_cutoff", "sucos_shape_pocket_qcov_cutoff", "n_targets_before_cutoff",
             "ligand_bin", "pocket_bin", "shift_bin",
             "morgan_tanimoto_rnp", "topological_tanimoto", "sucos_shape_pocket_qcov_rnp", "sucos_shape", "pocket_qcov",
             "pocket_fident", "protein_fident_qcov_weighted_max", "num_training_systems_with_similar_ccds",
             "rnp_cluster", "rnp_closest_train_system", "rnp_closest_train_release_date",
             "ligand_smiles", "model_ligand_smiles", "model_ligand_ccd_code", "model_ligand_chain_lddt_pli", "model_ligand_chain_rmsd",
             "ligand_num_heavy_atoms", "ligand_num_rot_bonds", "ligand_molecular_weight",
             "num_protein_chains", "num_ligand_chains", "num_proper_ligand_chains"]
    df = df[front].sort_values(GRAIN).reset_index(drop=True)
    assert not df.duplicated(GRAIN).any(), "grain is not unique"

    out_dir.mkdir(parents=True, exist_ok=True)
    if version is None:
        version = 1 + max([int(p.stem.split("_v")[1]) for p in out_dir.glob("master_v*.parquet")] or [0])
    out = out_dir / f"master_v{version}.parquet"
    df.to_parquet(out, index=False)
    link = out_dir / "master.parquet"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(out.name, link)
    meta = {"version": version, "rows": int(len(df)), "built": dt.datetime.now().isoformat(timespec="seconds"),
            "methods": diags, "bins": bins_used}
    (out_dir / f"master_v{version}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out, meta
