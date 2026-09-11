"""Phase 4: matched-abstention model comparison and leakage decomposition.

    uv run python scripts/08_phase4.py

Writes results/phase4_matched_abstention.csv, results/phase4_leakage.csv and
results/phase4_report.md. Reads only results/master.parquet.

Two analyses (spec section 7):

1. Matched-abstention comparison. Within each shift bin, force every model to the same
   abstention rate by taking the top (1-a) fraction of its own poses ranked by its own best
   signal, and compare accuracy among what each accepts. This asks whether the expensive model
   still wins on unfamiliar targets once both are allowed to decline the same share of cases.

   Deviation from the spec, stated plainly: the spec compares against smina. There is no smina
   run in this table and none is possible without the ligand structures. RoseTTAFold All-Atom is
   the only non-diffusion method in the release, but it ships one deterministic pose with NO
   confidence score, so it cannot be placed on an abstention curve at all. It is therefore
   reported as a single fixed-accuracy reference line on its shared subset, not as a matched
   competitor, and the cheap-baseline question the spec asked is NOT answerable from this data.

2. Leakage decomposition. What share of headline accuracy comes from the most ligand-similar
   decile: report accuracy overall, accuracy excluding that decile, and the gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location("c5", Path(__file__).parent / "05_calibrate.py")
c5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BIN = "pocket_bin"
ABSTENTIONS = [0.0, 0.25, 0.5, 0.75, 0.9]


def analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.after_cutoff == True)  # noqa: E712
           & (df.correct.notna())].copy()
    d["correct_num"] = d["correct"].astype(float)
    return d


def matched_abstention(d: pd.DataFrame, sigs: dict) -> pd.DataFrame:
    rows = []
    for (model, b), sub in d[d[BIN].notna()].groupby(["model", BIN]):
        if model not in sigs:
            continue
        s = c5.oriented(sub, sigs[model])
        sub = sub.assign(_sig=s).dropna(subset=["_sig"])
        if len(sub) < 30:
            continue
        for a in ABSTENTIONS:
            k = int(round(len(sub) * (1 - a)))
            if k < 10:
                continue
            top = sub.nlargest(k, "_sig")
            rows.append({"model": model, "bin": b, "abstention": a, "n_total": len(sub),
                         "n_accepted": k, "accuracy": float(top.correct_num.mean())})
    return pd.DataFrame(rows)


def rfaa_reference(df: pd.DataFrame) -> pd.DataFrame:
    """RFAA accuracy on the subset it shares with each co-folding model. No abstention: RFAA has
    no confidence signal, so it always accepts. Bins come from the *partner* model's row, since
    RFAA has no cutoff and therefore no shift coordinates of its own."""
    top = df[(df.is_top_ranked) & (df.ligand_is_proper == True) & (df.correct.notna())]  # noqa: E712
    r = top[top.model == "rfaa"][["group_key", "correct"]].rename(columns={"correct": "rfaa_correct"})
    rows = []
    for model, sub in top[top.model != "rfaa"].groupby("model"):
        m = sub[sub.after_cutoff == True][["group_key", BIN, "correct"]].merge(r, on="group_key")  # noqa: E712
        for b, g in m[m[BIN].notna()].groupby(BIN):
            if len(g) < 30:
                continue
            rows.append({"model": model, "bin": b, "n_shared": len(g),
                         "rfaa_accuracy": float(g.rfaa_correct.astype(float).mean()),
                         "model_accuracy_no_abstention": float(g.correct.astype(float).mean())})
    return pd.DataFrame(rows)


def leakage(d: pd.DataFrame) -> pd.DataFrame:
    """Two cuts, because the similarity distribution has a large mass point at exactly 100.

    Roughly 29% of post-cutoff cases have a ligand with Tanimoto 100 to something released
    before the model's training cutoff - the identical ligand, in a different structure. A
    90th-percentile cut cannot split that mass point, so "top decile" would silently mean "top
    29%". We therefore report BOTH:
      - `identical`: the exact-match group (Tanimoto = 100), named for what it is;
      - `next_decile`: a true top-10% cut taken among the NON-identical cases, which measures
        the gradient of similarity separately from exact recall.
    """
    rows = []
    for model, sub in d.groupby("model"):
        sub = sub.dropna(subset=["morgan_tanimoto_cutoff"])
        if len(sub) < 50:
            continue
        head = float(sub.correct_num.mean())
        ident = sub[sub.morgan_tanimoto_cutoff >= 100]
        non_ident = sub[sub.morgan_tanimoto_cutoff < 100]
        cut = non_ident.morgan_tanimoto_cutoff.quantile(0.9) if len(non_ident) else np.nan
        near = non_ident[non_ident.morgan_tanimoto_cutoff >= cut]
        rest = non_ident[non_ident.morgan_tanimoto_cutoff < cut]
        tot = sub.correct_num.sum()
        acc_non_ident = float(non_ident.correct_num.mean()) if len(non_ident) else np.nan
        acc_rest = float(rest.correct_num.mean()) if len(rest) else np.nan
        rows.append({"model": model, "n": len(sub),
                     "headline_accuracy": head,
                     "frac_identical_ligand": len(ident) / len(sub),
                     "identical_accuracy": float(ident.correct_num.mean()) if len(ident) else np.nan,
                     "accuracy_excl_identical": acc_non_ident,
                     "drop_excl_identical": head - acc_non_ident,
                     "share_of_correct_from_identical": float(ident.correct_num.sum() / tot) if tot else np.nan,
                     "next_decile_cut": round(float(cut), 1),
                     "next_decile_accuracy": float(near.correct_num.mean()) if len(near) else np.nan,
                     "accuracy_excl_identical_and_decile": acc_rest,
                     "drop_excl_both": head - acc_rest})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(RESULTS / "master.parquet")
    d = analysis_frame(df)
    sigs = c5.best_signal_per_model()
    ma = matched_abstention(d, sigs); ma.to_csv(RESULTS / "phase4_matched_abstention.csv", index=False)
    ref = rfaa_reference(df)
    lk = leakage(d); lk.to_csv(RESULTS / "phase4_leakage.csv", index=False)

    out = ["# Phase 4: adversarial comparison\n"]
    P = out.append
    P("## 1. Matched-abstention model comparison\n")
    P("Accuracy among accepted, when every model is forced to the same abstention rate within a")
    P("bin, ranking by its own best Phase 2 signal. Label `correct = rmsd < 2 A`.\n")
    for b in sorted(ma["bin"].unique(), key=lambda x: float(x.split("-")[0])):
        P(f"### pocket similarity {b}\n")
        t = ma[ma["bin"] == b].pivot_table(index="model", columns="abstention", values="accuracy")
        P(t.round(3).to_markdown()); P("")
    P("### RoseTTAFold All-Atom reference (no abstention possible)\n")
    P("RFAA ships one deterministic pose with no confidence score, so it cannot be given an")
    P("abstention rate. Below it is compared at zero abstention on the group_keys it shares with")
    P("each model. **The spec's cheap-baseline comparison (versus smina) is not answerable from")
    P("this release**: no docking baseline is included and reproducing one needs the ligand")
    P("structures.\n")
    piv = ref.pivot_table(index="model", columns="bin", values=["model_accuracy_no_abstention", "rfaa_accuracy"])
    piv.columns = [f"{b} {'model' if v.startswith('model') else 'RFAA'}" for v, b in piv.columns]
    P(piv[sorted(piv.columns, key=lambda c: (float(c.split('-')[0]), c))].round(3).to_markdown()); P("")

    P("## 2. Leakage decomposition\n")
    P("How much of headline accuracy rests on ligands the model has effectively already seen.")
    P("`morgan_tanimoto_cutoff` is the ECFP4 Tanimoto to the nearest ligand released before the")
    P("model's own training cutoff. It has a large mass at exactly 100 (the identical ligand in a")
    P("different structure), so the exact-match group is reported separately from a true decile")
    P("taken among the remaining cases.\n")
    P(lk.set_index("model").round(3).to_markdown()); P("")
    P(f"- Cases whose ligand is *identical* to a pre-cutoff ligand: "
      f"{lk.frac_identical_ligand.mean():.1%} of the post-cutoff benchmark.")
    P(f"- Accuracy on those: {lk.identical_accuracy.mean():.3f}, against "
      f"{lk.accuracy_excl_identical.mean():.3f} on everything else "
      f"(mean drop {lk.drop_excl_identical.mean():.3f}).")
    P(f"- They supply {lk.share_of_correct_from_identical.mean():.1%} of all correct predictions.")
    P(f"- Excluding both the identical group and the next decile, accuracy falls to "
      f"{lk.accuracy_excl_identical_and_decile.mean():.3f} "
      f"(mean drop {lk.drop_excl_both.mean():.3f} from headline).\n")
    (RESULTS / "phase4_report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
