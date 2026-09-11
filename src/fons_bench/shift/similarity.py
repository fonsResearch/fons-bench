"""Recompute similarity-to-training against a per-model training cutoff.

Follows the Runs N' Poses README recipe: from all_similarity_scores.parquet (every RnP
group_key scored against the whole PDB to 2025-01-05) keep only targets released before the
model's cutoff, take the best-scoring target per group_key for each metric, and map it back.

Two metrics are produced per model, each maximised independently (the closest ligand and the
closest pocket need not be the same training system):
    morgan_tanimoto_cutoff         max ECFP4 Tanimoto to any pre-cutoff ligand (0-100)
    sucos_shape_pocket_qcov_cutoff max SuCOS-pocket similarity to any pre-cutoff system (0-100)
A group_key with no pre-cutoff target at all gets 0 for both (nothing similar existed), and
`n_targets_before_cutoff` records how many candidate targets survived the date filter.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

METRICS = ("morgan_tanimoto", "sucos_shape_pocket_qcov")


def similarity_at_cutoff(all_sim: pd.DataFrame, cutoff: dt.date | str, group_keys: pd.Series) -> pd.DataFrame:
    """One row per group_key in ``group_keys`` with the cutoff-specific similarity columns."""
    cutoff = pd.Timestamp(cutoff)
    rel = pd.to_datetime(all_sim["target_release_date"], errors="coerce")
    before = all_sim.loc[rel < cutoff]
    out = pd.DataFrame({"group_key": pd.Series(group_keys).drop_duplicates().to_numpy()})
    for m in METRICS:
        best = before.dropna(subset=[m]).sort_values(m, ascending=False).groupby("group_key").head(1)
        out[f"{m}_cutoff"] = out["group_key"].map(dict(zip(best["group_key"], best[m])))
    counts = before.groupby("group_key").size()
    out["n_targets_before_cutoff"] = out["group_key"].map(counts).fillna(0).astype(int)
    # no pre-cutoff target at all: similarity is zero, not missing
    none = out["n_targets_before_cutoff"] == 0
    for m in METRICS:
        out.loc[none, f"{m}_cutoff"] = 0.0
    return out
