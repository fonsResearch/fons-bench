"""Phase 2: risk-coverage analysis and the Gate 2 verdict against results/kill_condition.md.

    uv run python scripts/04_risk_coverage.py [--boot 1000]

Writes results/aurc_table.csv, results/rc_curves.csv and results/gate2_report.md.
Reads only results/master.parquet. Every signal used is checked by the leakage guard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fons_bench.risk_coverage import bootstrap_aurc, risk_coverage
from fons_bench.signals import HIGHER_IS_BETTER, SIGNAL_COLUMNS, assert_no_leakage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXCLUDE_MODELS = {"rfaa"}          # no cutoff, no confidence signals (Gate 1 decision)
PRIMARY_BIN = "pocket_bin"
SECONDARY_BIN = "ligand_bin"
# Kill-condition thresholds, fixed in results/kill_condition.md before any curve was drawn.
KC_EXCESS_MAX = 0.25
KC_MONOTONE_TOL = 0.02
KC_DEGRADATION_MAX = 0.10


def analysis_set(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (~df.model.isin(EXCLUDE_MODELS)) & (df.correct.notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    return d


def signals_present(d: pd.DataFrame) -> list[str]:
    sig = [c for c in SIGNAL_COLUMNS if c in d.columns and d[c].notna().any()]
    assert_no_leakage(sig)
    return sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    df = pd.read_parquet(RESULTS / "master.parquet")
    d = analysis_set(df)
    sigs = signals_present(d)
    print(f"analysis set: {len(d)} top-ranked poses, {d.model.nunique()} models, {len(sigs)} signals")

    rows, curves = [], []
    strata = [("overall", "all")] + [(PRIMARY_BIN, b) for b in sorted(d[PRIMARY_BIN].dropna().unique())] \
                                  + [(SECONDARY_BIN, b) for b in sorted(d[SECONDARY_BIN].dropna().unique())]
    for model, dm in d.groupby("model"):
        for axis, b in strata:
            sub = dm if axis == "overall" else dm[dm[axis] == b]
            if len(sub) < 30 or sub.correct_num.nunique() < 2:
                continue
            for s in sigs:
                x = sub[[s, "correct_num"]].dropna()
                if len(x) < 30 or x.correct_num.nunique() < 2:
                    continue
                hib = HIGHER_IS_BETTER[s]
                rc = risk_coverage(x[s], x.correct_num, higher_is_better=hib)
                boot = bootstrap_aurc(x[s], x.correct_num, n_boot=args.boot, seed=0, higher_is_better=hib) if axis == "overall" else {}
                rows.append({"model": model, "axis": axis, "bin": b, "signal": s,
                             "n": rc.n, "n_errors": rc.n_errors, "base_error": rc.aurc_random,
                             "aurc": rc.aurc, "aurc_oracle": rc.aurc_oracle, "excess": rc.excess,
                             "monotone_violation": rc.monotonicity_violation(),
                             **{k: v for k, v in boot.items() if k not in ("aurc", "excess", "n", "n_errors")}})
                if axis in ("overall", PRIMARY_BIN):
                    step = max(1, rc.n // 200)
                    curves.append(pd.DataFrame({"model": model, "axis": axis, "bin": b, "signal": s,
                                                "coverage": rc.coverage[::step], "risk": rc.risk[::step]}))
    aurc = pd.DataFrame(rows)
    aurc.to_csv(RESULTS / "aurc_table.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(RESULTS / "rc_curves.csv", index=False)

    # ---- Gate 2 verdict --------------------------------------------------------------
    out = ["# Gate 2 report: risk-coverage and the kill check\n",
           f"Analysis set: top-ranked pose per (group_key, model), proper ligands, after each model's",
           f"cutoff, label `correct = rmsd < 2 A`. {len(d)} poses across {d.model.nunique()} models.",
           f"Signals tested: {len(sigs)} (ranking_score, 12 ipTM variants, dispersion statistics).",
           f"Bootstrap: {args.boot} resamples, percentile CI.\n"]
    P = out.append

    P("## Best signal per model, overall\n")
    ov = aurc[aurc.axis == "overall"]
    best = ov.sort_values("excess").groupby("model").head(1)
    P(best[["model", "signal", "n", "base_error", "aurc", "aurc_lo", "aurc_hi", "excess", "excess_lo", "excess_hi"]]
      .round(3).to_markdown(index=False)); P("")

    P("## Signal ranking, overall (mean normalised excess AURC across models; lower is better)\n")
    r = ov.pivot_table(index="signal", columns="model", values="excess").round(3)
    r["mean"] = r.mean(axis=1).round(3)
    P(r.sort_values("mean").to_markdown()); P("")

    P("## Dispersion signals vs native point signals\n")
    disp = ov[ov.signal.str.endswith(("_std", "_iqr", "_range", "_gap"))]
    pt = ov[~ov.signal.str.endswith(("_std", "_iqr", "_range", "_gap"))]
    P(f"- best point signal, mean excess across models: {pt.groupby('signal').excess.mean().min():.3f}")
    P(f"- best dispersion signal, mean excess across models: {disp.groupby('signal').excess.mean().min():.3f}")
    P(f"- best dispersion signal is `{disp.groupby('signal').excess.mean().idxmin()}`\n")

    P(f"## Stratified by {PRIMARY_BIN} (primary): excess AURC of each model's best overall signal\n")
    tab = []
    for model, sig in zip(best.model, best.signal):
        r_ = aurc[(aurc.model == model) & (aurc.signal == sig) & (aurc.axis == PRIMARY_BIN)]
        for _, x in r_.iterrows():
            tab.append({"model": model, "signal": sig, "bin": x["bin"], "n": x["n"],
                        "base_error": round(x.base_error, 3), "excess": round(x.excess, 3),
                        "monotone_violation": round(x.monotone_violation, 3)})
    strat = pd.DataFrame(tab)
    P(strat.pivot_table(index="model", columns="bin", values="excess").round(3).to_markdown()); P("")
    P("Base error rate per bin (fraction of top-ranked poses with rmsd >= 2 A):\n")
    P(strat.pivot_table(index="model", columns="bin", values="base_error").round(3).to_markdown()); P("")
    P("Poses per bin:\n")
    P(strat.pivot_table(index="model", columns="bin", values="n").astype("Int64").to_markdown()); P("")

    P("## Verdict against the kill condition\n")
    P(f"Criteria (fixed in advance): (1) excess <= {KC_EXCESS_MAX} in every bin; "
      f"(2) monotonicity violation <= {KC_MONOTONE_TOL}; "
      f"(3) excess(lowest bin) - excess(highest bin) <= {KC_DEGRADATION_MAX}.\n")
    verdict_rows = []
    bins_sorted = sorted(strat.bin.unique(), key=lambda b: float(b.split("-")[0]))
    lo_bin, hi_bin = bins_sorted[0], bins_sorted[-1]
    for model in sorted(strat.model.unique()):
        s_ = strat[strat.model == model].set_index("bin")
        c1 = bool((s_.excess <= KC_EXCESS_MAX).all())
        c2 = bool((s_.monotone_violation <= KC_MONOTONE_TOL).all())
        deg = (s_.loc[lo_bin, "excess"] - s_.loc[hi_bin, "excess"]) if {lo_bin, hi_bin} <= set(s_.index) else np.nan
        c3 = bool(deg <= KC_DEGRADATION_MAX) if not np.isnan(deg) else False
        verdict_rows.append({"model": model, "worst_excess": round(s_.excess.max(), 3), "c1_excess_ok": c1,
                             "worst_monotone_violation": round(s_.monotone_violation.max(), 3), "c2_monotone_ok": c2,
                             f"degradation_{lo_bin}_minus_{hi_bin}": round(deg, 3), "c3_no_degradation_ok": c3,
                             "all_pass": c1 and c2 and c3})
    v = pd.DataFrame(verdict_rows)
    P(v.to_markdown(index=False)); P("")
    killed = bool(v.all_pass.all())
    P(f"**Kill condition fires: {killed}.**")
    P("" if not killed else "\nEvery model passes every criterion in every bin. The premise is falsified; stop.")
    if not killed:
        fail = v[~v.all_pass]
        P(f"\n{len(fail)} of {len(v)} models fail at least one criterion, so the premise survives Gate 2. "
          f"Failing models: {', '.join(fail.model)}.")
        P(f"\nWorst normalised excess AURC observed in any bin: {strat.excess.max():.3f} "
          f"({strat.loc[strat.excess.idxmax(), 'model']}, bin {strat.loc[strat.excess.idxmax(), 'bin']}).")
    (RESULTS / "gate2_report.md").write_text("\n".join(out))
    (RESULTS / "gate2_verdict.json").write_text(json.dumps({"kill_condition_fires": killed, "models": verdict_rows}, indent=1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
