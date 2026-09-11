"""Phase 3, continued: how the cost of the guarantee varies with the target error rate.

    uv run python scripts/06_alpha_sweep.py

For each alpha, refit all three calibrations and record per-bin realised error, abstention and
how often a bin is certifiable at all. Writes results/alpha_sweep.csv and appends a section to
results/gate3_report.md. Fewer repeats than 05 because this runs the whole protocol per alpha.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fons_bench.calibrate.evaluate import evaluate
from fons_bench.calibrate.marginal import calibrate_marginal
from fons_bench.calibrate.mondrian import calibrate_mondrian
from fons_bench.calibrate.weighted import calibrate_weighted
from fons_bench.signals import HIGHER_IS_BETTER

import importlib.util
spec = importlib.util.spec_from_file_location("c5", Path(__file__).parent / "05_calibrate.py")
c5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4]
REPEATS = 20
BIN = c5.BIN


def main() -> None:
    df = pd.read_parquet(RESULTS / "master.parquet")
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (~df.model.isin(c5.EXCLUDE)) & (df.correct.notna()) & (df[BIN].notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    sigs = c5.best_signal_per_model()

    rows = []
    for model, dm in d.groupby("model"):
        sig = sigs[model]
        dm = dm.assign(_sig=c5.oriented(dm, sig)).dropna(subset=["_sig"])
        for alpha in ALPHAS:
            for rep in range(REPEATS):
                rng = np.random.default_rng(rep)
                is_cal = np.zeros(len(dm), dtype=bool)
                for _, idx in dm.groupby(BIN).indices.items():
                    is_cal[rng.permutation(idx)[: len(idx) // 2]] = True
                cal, test = dm[is_cal], dm[~is_cal]
                fits = {"marginal": calibrate_marginal(cal, alpha, "_sig"),
                        "mondrian": calibrate_mondrian(cal, alpha, "_sig", group=BIN),
                        "weighted": calibrate_weighted(cal, test, alpha, "_sig", seed=rep)}
                for name, fit in fits.items():
                    r = evaluate(fit, test, "_sig", group=BIN)
                    r.insert(0, "alpha", alpha); r.insert(0, "repeat", rep)
                    r.insert(0, "method", name); r.insert(0, "model", model)
                    rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RESULTS / "alpha_sweep.csv", index=False)

    agg = (res.groupby(["method", "alpha", "group"])
              .agg(realised_error=("realised_error", "mean"), abstention=("abstention", "mean"),
                   frac_violating=("violates_alpha", "mean"),
                   frac_no_acceptance=("n_accepted", lambda s: float((s == 0).mean())))
              .reset_index())
    out = ["\n\n## Alpha sweep: the cost of conditional validity\n",
           f"Averaged over {len(ALPHAS)} target levels, {REPEATS} splits, all 7 models.\n"]
    P = out.append
    for val, title in [("abstention", "Abstention rate"), ("realised_error", "Realised error among accepted"),
                       ("frac_no_acceptance", "Fraction of splits accepting nothing")]:
        P(f"### {title}, by alpha and bin\n")
        for method in ["marginal", "mondrian", "weighted"]:
            t = agg[agg.method == method].pivot_table(index="alpha", columns="group", values=val)
            P(f"**{method}**\n"); P(t.round(3).to_markdown()); P("")
    lo = sorted([g for g in agg.group.unique() if g != "all"], key=lambda b: float(b.split("-")[0]))[0]
    P(f"### Cost of the guarantee in the lowest-similarity bin ({lo})\n")
    m = agg[(agg.method == "marginal") & (agg.group == lo)].set_index("alpha")
    o = agg[(agg.method == "mondrian") & (agg.group == lo)].set_index("alpha")
    cost = pd.DataFrame({"marginal_error": m.realised_error, "marginal_abstention": m.abstention,
                         "mondrian_abstention": o.abstention, "extra_abstention": o.abstention - m.abstention,
                         "mondrian_accepts_nothing": o.frac_no_acceptance})
    P(cost.round(3).to_markdown()); P("")
    (RESULTS / "gate3_report.md").write_text((RESULTS / "gate3_report.md").read_text() + "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
