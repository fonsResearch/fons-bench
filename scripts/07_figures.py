"""Generate every figure from results/*.csv. No manual plotting, no notebooks.

    uv run python scripts/07_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fons_bench.figures.style import METHOD_COLORS, MODEL_COLORS, bin_order, use_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def fig_risk_coverage() -> None:
    """Risk-coverage curves per model, stratified by pocket-similarity bin."""
    c = pd.read_csv(RESULTS / "rc_curves.csv")
    a = pd.read_csv(RESULTS / "aurc_table.csv")
    best = dict(zip(*a[a.axis == "overall"].sort_values("excess").groupby("model").head(1)[["model", "signal"]].values.T))
    c = c[c.apply(lambda r: r.signal == best.get(r.model), axis=1) & (c.axis == "pocket_bin")]
    bins = bin_order(c["bin"].unique())
    fig, axes = plt.subplots(1, len(bins), figsize=(3.0 * len(bins), 3.0), sharey=True)
    for ax, b in zip(axes, bins):
        for model, d in c[c["bin"] == b].groupby("model"):
            # below ~5% coverage the accepted set is a handful of poses and the risk estimate is
            # dominated by sampling noise; plotting it would put a meaningless spike on the axis
            d = d[d.coverage >= 0.05].sort_values("coverage")
            ax.plot(d.coverage, d.risk, color=MODEL_COLORS.get(model, "k"), lw=1.4, label=model)
        ax.set_title(f"pocket similarity {b}"); ax.set_xlabel("coverage")
        ax.set_ylim(0, 0.85); ax.set_xlim(0, 1)
    axes[0].set_ylabel("risk (error rate among accepted)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Risk-coverage by shift bin: selective prediction degrades as the pocket becomes unfamiliar", y=1.04)
    fig.savefig(FIGS / "risk_coverage_by_bin.png"); plt.close(fig)


def fig_coverage_violation() -> None:
    """The central plot: marginal calibration's realised error rises above alpha as similarity falls."""
    r = pd.read_csv(RESULTS / "conditional_coverage.csv")
    alpha = 0.1
    # average over models and splits: the claim is about the method, not one model
    agg = (r.groupby(["method", "group"])
             .agg(err=("realised_error", "mean"), abst=("abstention", "mean"),
                  none=("n_accepted", lambda s: float((s == 0).mean()))).reset_index())
    bins = bin_order(agg.group.unique())
    x = range(len(bins))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for method in ["marginal", "mondrian", "weighted"]:
        d = agg[agg.method == method].set_index("group").reindex(bins)
        ax.plot(x, d.err, "o-", color=METHOD_COLORS[method], lw=1.6, ms=4, label=method)
        ax2.plot(x, d.abst, "o-", color=METHOD_COLORS[method], lw=1.6, ms=4, label=method)
        # mark bins where the method abstains entirely: its error is undefined there
        for xi, (e_, n_) in enumerate(zip(d.err, d.none)):
            if n_ > 0.5:
                ax.plot(xi, e_, "o", mfc="white", mec=METHOD_COLORS[method], ms=8, zorder=3)
    ax.axhline(alpha, ls="--", c="k", lw=1)
    ax.text(0.02, alpha + 0.012, f"target alpha = {alpha}", fontsize=8, transform=ax.get_yaxis_transform())
    for a_, lab in ((ax, "realised error among accepted"), (ax2, "abstention rate")):
        a_.set_xticks(list(x)); a_.set_xticklabels(bins); a_.set_xlabel("pocket similarity to nearest pre-cutoff system")
        a_.set_ylabel(lab)
    ax.set_ylim(0, None)          # start at zero so the size of the violation is honest
    ax2.set_ylim(0, 1.02)
    ax.set_title("Marginal calibration violates its guarantee\nwhere similarity is low", fontsize=9.5)
    ax2.set_title("Group-conditional calibration pays for\nvalidity in abstention", fontsize=9.5)
    fig.subplots_adjust(wspace=0.28)
    ax2.legend(loc="lower left")
    ax.plot([], [], "o", mfc="white", mec="k", ms=8, label="accepts nothing in >50% of splits")
    ax.legend(loc="upper right")
    fig.savefig(FIGS / "coverage_violation.png"); plt.close(fig)


def fig_alpha_sweep() -> None:
    p = RESULTS / "alpha_sweep.csv"
    if not p.exists():
        print("skip alpha sweep figure (results/alpha_sweep.csv not present)"); return
    s = pd.read_csv(p)
    agg = (s.groupby(["method", "alpha", "group"])
             .agg(err=("realised_error", "mean"), abst=("abstention", "mean")).reset_index())
    bins = bin_order(agg.group.unique())
    fig, axes = plt.subplots(1, len(bins), figsize=(3.0 * len(bins), 3.0), sharey=True)
    for ax, b in zip(axes, bins):
        d = agg[agg.group == b]
        for method in ["marginal", "mondrian", "weighted"]:
            dd = d[d.method == method].sort_values("alpha")
            ax.plot(dd.alpha, dd.abst, "o-", color=METHOD_COLORS[method], lw=1.5, ms=4, label=method)
        ax.set_title(f"pocket similarity {b}"); ax.set_xlabel("target error rate alpha"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("abstention rate")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Cost of the guarantee: abstention required to hold alpha, by shift bin", y=1.04)
    fig.savefig(FIGS / "alpha_sweep_abstention.png"); plt.close(fig)


def main() -> None:
    use_style(); FIGS.mkdir(parents=True, exist_ok=True)
    fig_risk_coverage(); fig_coverage_violation(); fig_alpha_sweep()
    print("wrote:", *sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
