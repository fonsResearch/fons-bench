"""Verify our RMSD scorer against the release's OpenStructure BiSyRMSD, in the regime that matters.

    uv run python scripts/11_verify_rmsd.py [--n 30]

The earlier check (results/rmsd_verification_8c3u.csv) used the one system that ships predicted
structures in the repo, and every pose there is 8-30 A from correct. That leaves the decision
boundary untested: what matters for the `correct` label is whether we agree with the release
NEAR 2 A, since only there can a disagreement flip a label.

This script samples systems stratified across the RMSD range - deliberately over-sampling the
1-3 A band - extracts their predicted structures from prediction_files.tar.gz, rescores them
with fons_bench.score.bisy_rmsd, and reports agreement overall and within the critical band.

Writes results/rmsd_verification_full.csv and prints the summary.
"""
from __future__ import annotations

import argparse
import re
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fons_bench.score.bisy_rmsd import bisy_rmsd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "runs_n_poses"
RESULTS = ROOT / "results"
GT = RAW / "ground_truth.tar.gz"
PRED = RAW / "prediction_files.tar.gz"

# The archive uses one uniform layout for every method, under a long cluster-path prefix:
#   .../prediction_files/<method>/<system_id>/seed-<seed>_sample-<sample>.cif
# `af3_no_template` is stored in a directory named `af3_no_templ`.
MEMBER = re.compile(r"/prediction_files/(?P<method>[^/]+)/(?P<sys>[^/]+)/seed-(?P<seed>\d+)_sample-(?P<sample>\d+)\.cif$")
DIR_ALIAS = {"af3_no_template": "af3_no_templ"}
ANALYSABLE = ["af3", "af3_no_template", "boltz", "boltz1x", "boltz2", "chai", "protenix"]


def sample_targets(n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified sample of top-ranked poses, over-weighting the 1-3 A decision band."""
    df = pd.read_parquet(RESULTS / "master.parquet")
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.rmsd.notna())  # noqa: E712
           & (df.model.isin(ANALYSABLE))].copy()
    bands = [(0, 1, 0.15), (1, 2, 0.30), (2, 3, 0.30), (3, 8, 0.15), (8, 1e9, 0.10)]
    rng = np.random.default_rng(seed)
    picks = []
    for lo, hi, frac in bands:
        sub = d[(d.rmsd >= lo) & (d.rmsd < hi)]
        k = min(len(sub), max(1, int(round(n * frac))))
        if k:
            picks.append(sub.sample(k, random_state=rng.integers(1 << 30)))
    return pd.concat(picks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    for p in (GT, PRED):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    want = sample_targets(args.n)
    # carry the release's own ligand-copy assignment: both the reference chain it scored and the
    # model chain it scored it against. Without these we would compare the wrong copy.
    wanted = {(r.system_id, r.model, int(r.seed), int(r["sample"])):
              {"rmsd": r.rmsd, "ref_chain": r.ligand_instance_chain, "mdl_chain": r.model_ligand_chain_rmsd}
              for _, r in want.iterrows()}
    print(f"sampled {len(wanted)} poses across the RMSD range")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        systems = {s for s, _, _, _ in wanted}
        with tarfile.open(GT) as tf:
            members = [m for m in tf.getmembers()
                       if any(f"ground_truth/{s}/" in m.name for s in systems)]
            tf.extractall(td, members=members, filter="data")
        print(f"extracted ground truth for {len(systems)} systems")

        alias_to_model = {DIR_ALIAS.get(m, m): m for m in ANALYSABLE}
        found = {}
        with tarfile.open(PRED) as tf:
            for m in tf:
                mt = MEMBER.search(m.name)
                if not mt:
                    continue
                model = alias_to_model.get(mt["method"])
                if model is None:
                    continue
                key = (mt["sys"], model, int(mt["seed"]), int(mt["sample"]))
                if key in wanted and key not in found:
                    tf.extract(m, td, filter="data")
                    found[key] = td / m.name
                    if len(found) == len(wanted):
                        break
        print(f"located {len(found)}/{len(wanted)} predicted structures")

        rows = []
        for key, cif in found.items():
            sysid, model, seed, sample = key
            gtdir = next(td.glob(f"**/ground_truth/{sysid}"), None)
            if gtdir is None:
                continue
            info = wanted[key]
            lig = gtdir / "ligand_files" / f"{info['ref_chain']}.sdf"
            if not lig.exists():
                print(f"  no reference ligand {info['ref_chain']}.sdf for {sysid}"); continue
            # the release's model-chain label uses each method's own naming (protenix writes
            # "A0"/"B0"); if that chain is absent, fall back to searching every ligand copy
            mdl_chain = info["mdl_chain"] if isinstance(info["mdl_chain"], str) else None
            try:
                v = bisy_rmsd(gtdir / "receptor.cif", lig, cif, model_ligand_chain=mdl_chain)["rmsd"]
            except ValueError:
                try:
                    v = bisy_rmsd(gtdir / "receptor.cif", lig, cif)["rmsd"]
                except Exception as e:
                    print(f"  {sysid} {model}: {e}"); continue
            except Exception as e:
                print(f"  {sysid} {model}: {e}"); continue
            rows.append({"system_id": sysid, "model": model, "seed": seed, "sample": sample,
                         "ref_chain": info["ref_chain"], "mdl_chain": mdl_chain,
                         "released_rmsd": info["rmsd"], "our_rmsd": v})

    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("no poses could be rescored")
    r["diff"] = r.our_rmsd - r.released_rmsd
    r.to_csv(RESULTS / "rmsd_verification_full.csv", index=False)

    band = r[(r.released_rmsd >= 1) & (r.released_rmsd <= 3)]
    print(f"\nrescored {len(r)} poses")
    print(f"  pearson r      : {r.our_rmsd.corr(r.released_rmsd):.6f}")
    print(f"  mean |diff|    : {r['diff'].abs().mean():.4f} A")
    print(f"  max  |diff|    : {r['diff'].abs().max():.4f} A")
    print(f"  mean diff      : {r['diff'].mean():+.4f} A (systematic offset)")
    print(f"\n  in the 1-3 A decision band ({len(band)} poses):")
    if len(band):
        print(f"    mean |diff|  : {band['diff'].abs().mean():.4f} A")
        print(f"    max  |diff|  : {band['diff'].abs().max():.4f} A")
    flips = r[((r.released_rmsd < 2) != (r.our_rmsd < 2))]
    print(f"\n  label flips across the 2 A threshold: {len(flips)} of {len(r)}")
    if len(flips):
        print(flips[["system_id", "model", "released_rmsd", "our_rmsd"]].to_string(index=False))


if __name__ == "__main__":
    main()
