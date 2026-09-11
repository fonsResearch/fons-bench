"""Robustness check: repeat the Gate 3 conditional-coverage result on the LIGAND similarity axis.

    uv run python scripts/12_robustness_ligand_axis.py

The headline analysis groups on pocket similarity (`pocket_bin`), chosen because it is the
metric the source release used to define the closest training system and because it captures
pose memorisation. This script repeats the same protocol grouping on ligand ECFP4 Tanimoto
(`ligand_bin`) to check the finding is not an artifact of that choice.

Appends a section to results/gate3_report.md and writes results/robustness_ligand_axis.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fons_bench.calibrate.evaluate import evaluate
from fons_bench.calibrate.marginal import calibrate_marginal
from fons_bench.calibrate.mondrian import calibrate_mondrian

import importlib.util
spec = importlib.util.spec_from_file_location("c5", Path(__file__).parent / "05_calibrate.py")
c5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BIN = "ligand_bin"
ALPHA = 0.1
REPEATS = 30


def main() -> None:
    df = pd.read_parquet(RESULTS / "master.parquet")
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (~df.model.isin(c5.EXCLUDE)) & (df.correct.notna()) & (df[BIN].notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    sigs = c5.best_signal_per_model()

    rows = []
    for model, dm in d.groupby("model"):
        dm = dm.assign(_sig=c5.oriented(dm, sigs[model])).dropna(subset=["_sig"])
        for rep in range(REPEATS):
            rng = np.random.default_rng(rep)
            is_cal = np.zeros(len(dm), dtype=bool)
            for _, idx in dm.groupby(BIN).indices.items():
                is_cal[rng.permutation(idx)[: len(idx) // 2]] = True
            cal, test = dm[is_cal], dm[~is_cal]
            if len(cal) < 20 or len(test) < 20:
                continue
            for name, fit in (("marginal", calibrate_marginal(cal, ALPHA, "_sig")),
                              ("mondrian", calibrate_mondrian(cal, ALPHA, "_sig", group=BIN))):
                r = evaluate(fit, test, "_sig", group=BIN)
                r.insert(0, "repeat", rep); r.insert(0, "method", name); r.insert(0, "model", model)
                rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RESULTS / "robustness_ligand_axis.csv", index=False)

    agg = (res.groupby(["method", "group"])
              .agg(realised_error=("realised_error", "mean"), abstention=("abstention", "mean"),
                   frac_violating=("violates_alpha", "mean"),
                   frac_no_acceptance=("n_accepted", lambda s: float((s == 0).mean())))
              .reset_index())
    out = ["\n\n## Robustness: the same result on the ligand-similarity axis\n",
           f"Grouping on `{BIN}` (ECFP4 Tanimoto bins) instead of pocket similarity.",
           f"alpha = {ALPHA}, {REPEATS} splits, seven models. The pocket axis is primary; this",
           "check exists to show the conditional-coverage failure is not an artifact of that choice.\n"]
    P = out.append
    for method in ["marginal", "mondrian"]:
        P(f"**{method}**\n")
        t = agg[agg.method == method].set_index("group")[
            ["realised_error", "abstention", "frac_violating", "frac_no_acceptance"]]
        P(t.round(3).to_markdown()); P("")
    m = agg[agg.method == "marginal"]
    bins = sorted([g for g in m.group if g != "all"], key=lambda b: float(b.split("-")[0]))
    lo, hi = m[m.group == bins[0]].iloc[0], m[m.group == bins[-1]].iloc[0]
    P(f"Marginal calibration: {lo.realised_error:.3f} realised error in the least-similar ligand bin "
      f"({bins[0]}) against {hi.realised_error:.3f} in the most-similar ({bins[-1]}), "
      f"target {ALPHA}. Violates in {lo.frac_violating:.0%} of splits in the former.\n")
    (RESULTS / "gate3_report.md").write_text((RESULTS / "gate3_report.md").read_text() + "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
