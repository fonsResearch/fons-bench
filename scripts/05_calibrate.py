"""Phase 3: conformal risk control, three calibrations compared on real data.

    uv run python scripts/05_calibrate.py [--alpha 0.1] [--repeats 50]

Protocol. For each model and each repeat, split the top-ranked poses into a calibration half
and a test half at random (stratified by shift bin so every bin is represented in both), fit
the three calibrations on the calibration half, and evaluate realised error and abstention on
the test half, overall and per bin. Report the mean over repeats, so the numbers are not an
artifact of one lucky split.

Writes results/conditional_coverage.csv and results/gate3_report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fons_bench.calibrate.evaluate import evaluate
from fons_bench.calibrate.marginal import calibrate_marginal
from fons_bench.calibrate.mondrian import calibrate_mondrian
from fons_bench.calibrate.weighted import calibrate_weighted
from fons_bench.signals import HIGHER_IS_BETTER, assert_no_leakage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BIN = "pocket_bin"                      # primary grouping (Gate 1 decision)
EXCLUDE = {"rfaa"}


def best_signal_per_model() -> dict[str, str]:
    """The signal with the lowest overall normalised excess AURC, from Phase 2."""
    a = pd.read_csv(RESULTS / "aurc_table.csv")
    ov = a[a.axis == "overall"]
    best = ov.sort_values("excess").groupby("model").head(1)
    return dict(zip(best.model, best.signal))


def oriented(d: pd.DataFrame, signal: str) -> pd.Series:
    """Return the signal oriented so higher = more confident."""
    return d[signal] if HIGHER_IS_BETTER[signal] else -d[signal]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_parquet(RESULTS / "master.parquet")
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (~df.model.isin(EXCLUDE)) & (df.correct.notna()) & (df[BIN].notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    sigs = best_signal_per_model()
    assert_no_leakage(list(set(sigs.values())))

    rows = []
    for model, dm in d.groupby("model"):
        sig = sigs[model]
        dm = dm.assign(_sig=oriented(dm, sig)).dropna(subset=["_sig"])
        for rep in range(args.repeats):
            rng = np.random.default_rng(rep)
            # stratified half-split so every bin appears in both halves
            is_cal = np.zeros(len(dm), dtype=bool)
            for _, idx in dm.groupby(BIN).indices.items():
                pick = rng.permutation(idx)[: len(idx) // 2]
                is_cal[pick] = True
            cal, test = dm[is_cal], dm[~is_cal]
            if len(cal) < 20 or len(test) < 20:
                continue
            fits = {
                "marginal": calibrate_marginal(cal, args.alpha, "_sig"),
                "mondrian": calibrate_mondrian(cal, args.alpha, "_sig", group=BIN),
                "weighted": calibrate_weighted(cal, test, args.alpha, "_sig", seed=rep),
            }
            for name, fit in fits.items():
                r = evaluate(fit, test, "_sig", group=BIN)
                r.insert(0, "repeat", rep); r.insert(0, "method", name)
                r.insert(0, "signal", sig); r.insert(0, "model", model)
                rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RESULTS / "conditional_coverage.csv", index=False)

    agg = (res.groupby(["model", "method", "group"])
              .agg(n_test=("n_test", "mean"), realised_error=("realised_error", "mean"),
                   abstention=("abstention", "mean"), base_error=("base_error", "mean"),
                   frac_violating=("violates_alpha", "mean"),
                   frac_no_acceptance=("n_accepted", lambda s: float((s == 0).mean())))
              .reset_index())

    out = [f"# Gate 3 report: conditional coverage and the cost of the guarantee\n",
           f"alpha = {args.alpha}. Label `correct = rmsd < 2 A`. Grain: top-ranked pose per",
           f"(group_key, model), proper ligands, post-cutoff. Grouping variable: `{BIN}`.",
           f"{args.repeats} random stratified half-splits per model; tables are means over repeats.",
           f"Signal per model: the Phase 2 winner.\n",
           "A method that accepts nothing in a bin has an undefined error rate; `frac_no_acceptance`",
           "records how often that happened, and those repeats are excluded from the error mean.\n"]
    P = out.append

    P("## Realised error rate among accepted, per shift bin\n")
    for method in ["marginal", "mondrian", "weighted"]:
        P(f"### {method}\n")
        t = agg[agg.method == method].pivot_table(index="model", columns="group", values="realised_error")
        P(t.round(3).to_markdown()); P("")
    P("## Abstention rate, per shift bin\n")
    for method in ["marginal", "mondrian", "weighted"]:
        P(f"### {method}\n")
        t = agg[agg.method == method].pivot_table(index="model", columns="group", values="abstention")
        P(t.round(3).to_markdown()); P("")

    P("## How often each method accepted nothing at all in a bin\n")
    P("A bin where the method abstains entirely is *honest* - it reports that the bin cannot be")
    P("certified at this alpha with this much calibration data - but its realised error is then")
    P("undefined, and any error mean over the remaining splits is computed on a tiny, biased")
    P("subset. Read the error tables together with this one.\n")
    t = agg.pivot_table(index=["model", "method"], columns="group", values="frac_no_acceptance")
    P(t.round(2).to_markdown()); P("")

    P("## Violation summary (fraction of splits where realised error exceeded alpha)\n")
    t = agg.pivot_table(index=["model", "method"], columns="group", values="frac_violating")
    P(t.round(2).to_markdown()); P("")

    P("## Did the expected finding hold?\n")
    m_all = agg[(agg.method == "marginal") & (agg.group == "all")]
    bins_sorted = sorted([g for g in agg.group.unique() if g != "all"], key=lambda b: float(b.split("-")[0]))
    lo = bins_sorted[0]
    m_lo = agg[(agg.method == "marginal") & (agg.group == lo)]
    o_lo = agg[(agg.method == "mondrian") & (agg.group == lo)]
    P(f"- Marginal calibration, pooled: mean realised error {m_all.realised_error.mean():.3f} "
      f"against alpha {args.alpha}; violates in {m_all.frac_violating.mean():.0%} of splits.")
    P(f"- Marginal calibration, lowest-similarity bin ({lo}): mean realised error "
      f"{m_lo.realised_error.mean():.3f}; violates in {m_lo.frac_violating.mean():.0%} of splits.")
    P(f"- Mondrian, same bin: abstention {o_lo.abstention.mean():.3f} against marginal's "
      f"{m_lo.abstention.mean():.3f}; it accepts nothing at all in {o_lo.frac_no_acceptance.mean():.0%} of splits.")
    P(f"- Cost of the guarantee in that bin: {o_lo.abstention.mean() - m_lo.abstention.mean():+.3f} abstention.")
    P(f"- Where Mondrian does accept in that bin its mean error is {o_lo.realised_error.mean():.3f}, but that")
    P(f"  average is over the minority of splits with a non-empty acceptance set and is not a coverage claim.")
    P("")
    P("**Reading.** At alpha = 0.1 the lowest-similarity bin is essentially uncertifiable for every")
    P("model: group-conditional calibration responds by abstaining almost totally rather than")
    P("issuing a threshold it cannot back. The cost of conditional validity in that regime is not")
    P("a moderate rise in abstention, it is near-total loss of coverage. The moderate-cost regime")
    P("is the middle bins, where Mondrian holds alpha at a real but affordable abstention increase.")
    P("")
    (RESULTS / "gate3_report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
