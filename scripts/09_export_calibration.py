"""Export the calibrated thresholds that `fons-bench evaluate` ships with.

    uv run python scripts/09_export_calibration.py

Fits each calibration on ALL post-cutoff top-ranked poses (no held-out split: this is the
shipped artifact, not an evaluation) and writes src/fons_bench/artifacts/calibration.json.
That file is small enough to ship inside the wheel, so `evaluate` needs no GPU and no
benchmark download.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fons_bench.calibrate.marginal import calibrate_marginal
from fons_bench.calibrate.mondrian import calibrate_mondrian

import importlib.util
spec = importlib.util.spec_from_file_location("c5", Path(__file__).parent / "05_calibrate.py")
c5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ART = ROOT / "src" / "fons_bench" / "artifacts"
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4]


def main() -> None:
    df = pd.read_parquet(RESULTS / "master.parquet")
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (~df.model.isin(c5.EXCLUDE)) & (df.correct.notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    sigs = c5.best_signal_per_model()
    meta = json.loads((RESULTS / (Path((RESULTS / "master.parquet").readlink()).stem + ".meta.json")).read_text())

    out = {"schema": "fons-bench-calibration-v1",
           "master_version": meta["version"], "built": meta["built"],
           "label": "correct = rmsd < 2.0 A (BiSyRMSD)",
           "group_variable": c5.BIN,
           "bin_edges": meta["bins"]["pocket"]["edges"],
           "models": {}}
    for model, dm in d.groupby("model"):
        sig = sigs[model]
        dm = dm.assign(_sig=c5.oriented(dm, sig)).dropna(subset=["_sig"])
        entry = {"signal": sig, "higher_is_better": True,
                 "note": "signal is used as oriented; if the raw column is a dispersion statistic, negate it",
                 "n_calibration": int(len(dm)), "alphas": {}}
        for a in ALPHAS:
            m = calibrate_marginal(dm, a, "_sig")
            o = calibrate_mondrian(dm, a, "_sig", group=c5.BIN)
            entry["alphas"][str(a)] = {
                "marginal_tau": None if m.tau == float("inf") else float(m.tau),
                "mondrian_tau": {str(k): (None if v == float("inf") else float(v)) for k, v in o.tau.items()},
                "mondrian_n_cal": {str(k): int(v) for k, v in o.n_cal.items()},
            }
        out["models"][model] = entry

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "calibration.json").write_text(json.dumps(out, indent=1))
    (ART / "__init__.py").write_text("")
    n_null = sum(1 for m in out["models"].values() for a in m["alphas"].values()
                 for v in a["mondrian_tau"].values() if v is None)
    print(f"wrote {ART / 'calibration.json'}")
    print(f"models: {len(out['models'])}, alphas: {ALPHAS}")
    print(f"mondrian bins with no achievable threshold (abstain entirely): {n_null}")


if __name__ == "__main__":
    main()
