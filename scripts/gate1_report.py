"""Gate 1 numbers, regenerated from results/master.parquet. Writes results/gate1_report.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "results" / "master.parquet")
meta = json.loads((ROOT / "results" / (Path((ROOT / "results" / "master.parquet").readlink()).stem + ".meta.json")).read_text())
out = []
P = out.append

P(f"# Gate 1 report  (master_v{meta['version']}, built {meta['built']})\n")
P(f"Rows: {len(df):,}   columns: {df.shape[1]}   grain (group_key, model, seed, sample) unique: {not df.duplicated(['group_key','model','seed','sample']).any()}\n")

P("## Row counts per model\n")
g = df.groupby("model")
t = pd.DataFrame({
    "rows": g.size(),
    "group_keys": g.group_key.nunique(),
    "systems": g.system_id.nunique(),
    "proper_rows": g.ligand_is_proper.apply(lambda s: int((s == True).sum())),
    "proper_after_cutoff_rows": g.apply(lambda x: int(((x.ligand_is_proper == True) & (x.after_cutoff == True)).sum())),
    "pb_rows": g.pb_available.sum().astype(int),
    "correct_known_rows": g.correct.apply(lambda s: int(s.notna().sum())),
    "seeds_per_gk": g.apply(lambda x: x.groupby("group_key").seed.nunique().median()),
    "samples_per_seed": g.apply(lambda x: x.groupby(["group_key", "seed"])["sample"].nunique().median()),
    "cutoff": pd.Series({m: meta["methods"][m]["cutoff"] for m in meta["methods"]}),
})
P(t.to_markdown()); P("")
P("Poses per group_key (should be 25 = 5 seeds x 5 samples):\n")
for m in sorted(df.model.unique()):
    if m == "rfaa":
        continue
    vc = df[df.model == m].groupby("group_key").size().value_counts().sort_index(ascending=False)
    P(f"- {m}: " + ", ".join(f"{k} poses: {v}" for k, v in vc.head(5).items()))
P("")

P("## Cross-method coverage (proper group_keys)\n")
sets = {m: set(df[(df.model == m) & (df.ligand_is_proper == True)].group_key) for m in sorted(df.model.unique())}
core = ["af3", "af3_no_template", "boltz", "boltz1x", "chai", "protenix"]
P("| model | proper group_keys | Jaccard vs af3 |"); P("|---|---|---|")
for m, s in sets.items():
    P(f"| {m} | {len(s)} | {len(s & sets['af3']) / len(s | sets['af3']):.3f} |")
P(f"\nCommon to the six 2021-cutoff co-folding methods: {len(set.intersection(*[sets[m] for m in core]))}; union: {len(set.union(*[sets[m] for m in core]))}.")
P(f"Common to all eight: {len(set.intersection(*sets.values()))}. Annotations list {df.drop_duplicates('group_key').ligand_is_proper.eq(True).sum()} proper group_keys overall.")
P(f"Rows whose group_key is absent from annotations.csv: {int((~df.in_annotations).sum())} ({df[~df.in_annotations].group_key.nunique()} group_keys); kept, flagged in_annotations=False.\n")

P("## Headline accuracy (proper, after each model's cutoff, top-ranked pose by ranking_score)\n")
p = df[(df.ligand_is_proper == True) & (df.after_cutoff == True)]
top = p.sort_values("ranking_score", ascending=False).groupby(["model", "group_key"]).head(1)
h = top.groupby("model").agg(n=("group_key", "size"), rmsd_lt_2=("rmsd_lt_2", "mean"), pb_known=("pb_available", "mean"),
                            pb_valid=("pb_valid", "mean"), correct=("correct", "mean")).round(3)
P(h.to_markdown()); P("")
P("`correct` is only defined where PoseBusters was run (one pose per group_key in the release); `pb_valid`/`correct` means are over those rows.\n")

P("## Shift coordinates at each model's cutoff (proper, after cutoff, distinct group_keys)\n")
for col in ["morgan_tanimoto_cutoff", "sucos_shape_pocket_qcov_cutoff"]:
    P(f"### {col}\n")
    P("| model | n | no pre-cutoff target (=0) | " + " | ".join(f"{a}-{a+10}" for a in range(0, 100, 10)) + " |")
    P("|---|---|---|" + "---|" * 10)
    for m in sorted(p.model.unique()):
        x = p[p.model == m].drop_duplicates("group_key")
        hist, _ = np.histogram(x[col].dropna(), bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0001])
        P(f"| {m} | {x[col].notna().sum()} | {int((x.n_targets_before_cutoff == 0).sum())} | " + " | ".join(map(str, hist)) + " |")
    P("")

P("## Bin populations (distinct systems; proper, after cutoff)\n")
P(f"Edges used: ligand {meta['bins']['ligand']['edges']}, pocket {meta['bins']['pocket']['edges']}; merges: {meta['bins']['ligand']['merges'] + meta['bins']['pocket']['merges'] or 'none'}\n")
for axis in ["ligand_bin", "pocket_bin"]:
    t = p.drop_duplicates(["model", "group_key"]).groupby(["model", axis]).system_id.nunique().unstack(fill_value=0)
    P(f"### {axis}\n"); P(t.to_markdown()); P("")
    small = t.stack(); small = small[small < 100]
    if len(small):
        P("**Bins under 100 systems:** " + ", ".join(f"{m}/{b}={int(v)}" for (m, b), v in small.items()) + "\n")
P("### joint shift_bin (ligand|pocket), systems per cell\n")
t = p.drop_duplicates(["model", "group_key"]).groupby(["model", "shift_bin"]).system_id.nunique().unstack(fill_value=0)
P(t.T.to_markdown()); P("")
small = (t < 100).sum(axis=1)
P("Joint cells under 100 systems per model: " + ", ".join(f"{m}={v} of {t.shape[1]}" for m, v in small.items()) + "\n")

P("## Row-collapse diagnostic\n")
for m, d in meta["methods"].items():
    if "authors_rule_row_not_strict_frac" in d:
        P(f"- {m}: {d['authors_rule_row_not_strict_frac']:.3%} of kept rows are not the strict RMSD-side assignment row")
P("")
(ROOT / "results" / "gate1_report.md").write_text("\n".join(out))
print("\n".join(out))
