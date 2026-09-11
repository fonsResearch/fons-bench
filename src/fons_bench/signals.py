"""The closed list of admissible confidence signals, and the leakage guard.

A signal is admissible only if it can be computed at inference time, i.e. without the
ground-truth structure. Anything derived from rmsd, lddt_pli, lddt_lp, bb_rmsd, pred_pocket_*
or the PoseBusters checks is an outcome, never a signal. See results/kill_condition.md.
"""
from __future__ import annotations

import pandas as pd

IPTM_COLS = [f"{a}_chain_iptm_{s}_{m}" for a in ("prot_lig", "lig_prot") for s in ("average", "min", "max") for m in ("rmsd", "lddt_pli")]
NATIVE_SIGNALS = ["ranking_score", *IPTM_COLS]
DISPERSION_STATS = ("std", "iqr", "range")
DISPERSION_SIGNALS = [f"{c}_{s}" for c in NATIVE_SIGNALS for s in DISPERSION_STATS] + ["ranking_score_top1_top2_gap"]
SIGNAL_COLUMNS = NATIVE_SIGNALS + DISPERSION_SIGNALS

# Orientation: every signal is scored with "higher = more confident". For a dispersion
# statistic the opposite is true (poses that disagree across samples are less trustworthy), so
# the analysis negates it. Keeping the raw column and flipping at scoring time means the stored
# values stay interpretable as spreads.
HIGHER_IS_BETTER = {c: (c not in DISPERSION_SIGNALS) for c in SIGNAL_COLUMNS}

OUTCOME_PREFIXES = ("rmsd", "lddt", "bb_rmsd", "pred_pocket", "pb_", "correct", "strain")


def assert_no_leakage(signals: list[str]) -> None:
    """Raise if any requested signal is not on the closed list or names an outcome column."""
    bad = [s for s in signals if s not in SIGNAL_COLUMNS or any(s.startswith(p) for p in OUTCOME_PREFIXES)]
    if bad:
        raise ValueError(f"leakage guard: not admissible as signals: {bad}")


def dispersion_over_poses(df: pd.DataFrame, keys: list[str] = ["group_key", "model"]) -> pd.DataFrame:
    """Per (group_key, model): std, IQR and range of each native signal over all poses, and the
    gap between the best and second-best ranking_score. Uses only signal columns."""
    g = df.groupby(keys, sort=False)
    out = {}
    for c in NATIVE_SIGNALS:
        out[f"{c}_std"] = g[c].std(ddof=0)
        q = g[c].quantile([0.25, 0.75]).unstack()
        out[f"{c}_iqr"] = q[0.75] - q[0.25]
        out[f"{c}_range"] = g[c].max() - g[c].min()
    top2 = g["ranking_score"].nlargest(2).groupby(level=list(range(len(keys)))).agg(["first", "last", "size"])
    out["ranking_score_top1_top2_gap"] = (top2["first"] - top2["last"]).where(top2["size"] == 2)
    res = pd.DataFrame(out).reset_index()
    return res
