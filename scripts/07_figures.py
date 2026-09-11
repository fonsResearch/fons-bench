"""Generate every figure from results/*.csv. No manual plotting, no notebooks.

    uv run python scripts/07_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fons_bench.figures import style
from fons_bench.figures.style import METHOD_COLORS, MODEL_COLORS, bin_order, use_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def _save(fig, name: str, tight: bool = True) -> None:
    """Write the figure under the active theme.

    Light goes to ``<name>.png`` for the PDF; dark to ``<name>.dark.png`` for the
    website, whose page background is #0a0a0a. Pass ``tight=False`` when the
    figure positions its own titles or legend in figure coordinates, since the
    tight bbox recomputes the canvas and drags them onto the axes.
    """
    suffix = "" if style.THEME == "light" else ".dark"
    kw = {"bbox_inches": "tight"} if tight else {"bbox_inches": None}
    fig.savefig(FIGS / f"{name}{suffix}.png", **kw)
    plt.close(fig)


def fig_risk_coverage() -> None:
    """Risk-coverage curves per model, stratified by pocket-similarity bin."""
    c = pd.read_csv(RESULTS / "rc_curves.csv")
    a = pd.read_csv(RESULTS / "aurc_table.csv")
    best = dict(zip(*a[a.axis == "overall"].sort_values("excess").groupby("model").head(1)[["model", "signal"]].values.T))
    c = c[c.apply(lambda r: r.signal == best.get(r.model), axis=1) & (c.axis == "pocket_bin")]
    bins = bin_order(c["bin"].unique())
    # Squarer than a 1x4 strip: at the article column's width, four wide panels
    # shrink until the axis labels stop being readable.
    fig, axes = plt.subplots(1, len(bins), figsize=(2.7 * len(bins), 3.6), sharey=True)
    for i, (ax, b) in enumerate(zip(axes, bins)):
        for model, d in c[c["bin"] == b].groupby("model"):
            # below ~5% coverage the accepted set is a handful of poses and the risk estimate is
            # dominated by sampling noise; plotting it would put a meaningless spike on the axis
            d = d[d.coverage >= 0.05].sort_values("coverage")
            ax.plot(d.coverage, d.risk, color=MODEL_COLORS.get(model, style.MUTED),
                    lw=1.5, alpha=0.95, label=model)
        ax.set_title(f"pocket similarity {style.bin_label(b)}", loc="left", color=style.INK)
        ax.set_xlabel("coverage")
        ax.set_ylim(0, 0.85); ax.set_xlim(0, 1)
        # Drop the terminal tick: with panels this close, a trailing "1.00" runs
        # into the next panel's leading "0.00" and reads as one broken number.
        ax.set_xticks([0, 0.25, 0.5, 0.75])
        if i:
            ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("risk (error rate among accepted)")
    # Legend below rather than beside: a side legend costs ~13% of the width,
    # which the panels need more than the labels do. Matches the other figures.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.74, bottom=0.28, left=0.075, right=0.985, wspace=0.18)
    fig.suptitle("Risk-coverage by shift bin: selective prediction degrades as the pocket becomes unfamiliar",
                 x=0.045, y=0.96, ha="left", color=style.INK, fontsize=12, fontweight="medium")
    _save(fig, "risk_coverage_by_bin", tight=False)


def fig_coverage_violation() -> None:
    """The central plot: marginal calibration's realised error rises above alpha as similarity falls."""
    r = pd.read_csv(RESULTS / "conditional_coverage.csv")
    alpha = 0.1
    # average over models and splits: the claim is about the method, not one model
    agg = (r.groupby(["method", "group"])
             .agg(err=("realised_error", "mean"), abst=("abstention", "mean"),
                  none=("n_accepted", lambda s: float((s == 0).mean()))).reset_index())
    bins = bin_order(agg.group.unique())
    x = list(range(len(bins)))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Everything above alpha is a broken promise. Shade it once, so the violation
    # reads as an area the lines enter rather than a rule they happen to cross.
    # Bound the y-axis to the data first: shading to 1.0 would leave the lines
    # squashed into the bottom third of the panel.
    top = float(agg.err.max()) * 1.28
    ax.set_ylim(0, top)
    ax.axhspan(alpha, top, color=style.ZONE, alpha=style.ZONE_ALPHA, lw=0, zorder=0)
    ax.axhline(alpha, color=style.ZONE, lw=1.0, zorder=1)
    ax.text(-0.2, alpha + top * 0.018, f"target  α = {alpha}", fontsize=9.5,
            color=style.ZONE, va="bottom", ha="left")

    for method in ["marginal", "mondrian", "weighted"]:
        d = agg[agg.method == method].set_index("group").reindex(bins)
        em = style.emphasis(method)
        for a_, col in ((ax, d.err), (ax2, d.abst)):
            a_.plot(x, col, "o-", color=METHOD_COLORS[method], label=method,
                    mew=0, **em)
        # mark bins where the method abstains entirely: its error is undefined there
        for xi, (e_, n_) in enumerate(zip(d.err, d.none)):
            if n_ > 0.5:
                ax.plot(xi, e_, "o", mfc=style.BG, mec=METHOD_COLORS[method],
                        mew=1.4, ms=8, zorder=4)

    # Annotate the two endpoints the prose quotes, so the figure carries the claim alone.
    marg = agg[agg.method == "marginal"].set_index("group").reindex(bins)
    ax.annotate(f"{marg.err.iloc[0]:.1%}", (0, marg.err.iloc[0]), xytext=(8, 4),
                textcoords="offset points", fontsize=10.5, fontweight="medium",
                color=METHOD_COLORS["marginal"])
    ax.annotate(f"{marg.err.iloc[-1]:.1%}", (len(bins) - 1, marg.err.iloc[-1]), xytext=(10, 1),
                textcoords="offset points", fontsize=10.5, color=style.MUTED, ha="left", va="center")

    for a_, lab in ((ax, "realised error among accepted"), (ax2, "abstention rate")):
        a_.set_xticks(x); a_.set_xticklabels([style.bin_label(b) for b in bins])
        a_.set_xlabel("pocket similarity to nearest pre-cutoff system")
        a_.set_ylabel(lab)
        a_.set_xlim(-0.25, len(bins) - 0.62)
    ax2.set_ylim(0, 1.02)
    # Kept short: at this width two full sentences collide across the panel gap.
    ax.set_title("Marginal calibration violates its guarantee",
                 loc="left", color=style.INK)
    ax2.set_title("Group-conditional calibration pays in abstention",
                  loc="left", color=style.INK)

    # One legend for the whole figure: the same three series appear in both panels.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="o", mfc=style.BG,
                              mec=style.MUTED, mew=1.4, ms=8))
    labels.append("accepts nothing in >50% of splits")
    # Reserve the bands explicitly rather than letting tight-bbox reflow them:
    # the legend and the two panel titles both collide with the axes otherwise.
    fig.subplots_adjust(top=0.86, bottom=0.26, left=0.075, right=0.98, wspace=0.26)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    _save(fig, "coverage_violation", tight=False)


def fig_alpha_sweep() -> None:
    p = RESULTS / "alpha_sweep.csv"
    if not p.exists():
        print("skip alpha sweep figure (results/alpha_sweep.csv not present)"); return
    s = pd.read_csv(p)
    agg = (s.groupby(["method", "alpha", "group"])
             .agg(err=("realised_error", "mean"), abst=("abstention", "mean")).reset_index())
    bins = bin_order(agg.group.unique())
    fig, axes = plt.subplots(1, len(bins), figsize=(3.1 * len(bins), 3.2), sharey=True)
    for i, (ax, b) in enumerate(zip(axes, bins)):
        d = agg[agg.group == b]
        for method in ["marginal", "mondrian", "weighted"]:
            dd = d[d.method == method].sort_values("alpha")
            ax.plot(dd.alpha, dd.abst, "o-", color=METHOD_COLORS[method],
                    label=method, mew=0, **style.emphasis(method))
        ax.set_title(f"pocket similarity {style.bin_label(b)}", loc="left", color=style.INK)
        ax.set_xlabel("target error rate α")
        ax.set_ylim(0, 1.02); ax.set_xlim(0.02, 0.43)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        if i:
            # sharey already drops the labels; drop the spine too so the panels
            # read as one strip rather than four boxed charts
            ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("abstention rate")
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Reserve the top and bottom bands explicitly. Placing the suptitle and legend
    # in figure coords alone lets tight-bbox pull them onto the panels.
    fig.subplots_adjust(top=0.76, bottom=0.30, wspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    # Left-aligned above the first panel: a centred suptitle lands on panel 2's title.
    fig.suptitle("Cost of the guarantee: abstention required to hold α, by shift bin",
                 x=0.045, y=0.95, ha="left", color=style.INK, fontsize=12, fontweight="medium")
    _save(fig, "alpha_sweep_abstention", tight=False)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    # Light for the PDF, dark for the website. Same data, same code path.
    for theme in ("light", "dark"):
        use_style(theme)
        fig_risk_coverage(); fig_coverage_violation(); fig_alpha_sweep()
    print("wrote:", *sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
