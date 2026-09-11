"""Group-conditional (Mondrian) conformal risk control: one threshold per shift bin.

Same construction as `marginal`, applied within each group independently, so the guarantee
holds conditionally on the group. The cost is that each threshold is fitted on a fraction of
the calibration data, so the finite-sample inflation bites harder and the abstention rate rises
- that trade is the quantity the paper reports.

A group with too few calibration points to achieve alpha gets tau = +inf (abstain from
everything in that group). That is the honest behaviour: it says "this bin cannot be certified
at this level with this much data", instead of borrowing a threshold from a better-populated
bin and silently violating the guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fons_bench.calibrate.base import Calibration
from fons_bench.calibrate.marginal import fit_threshold


def calibrate_mondrian(cal: pd.DataFrame, alpha: float, signal: str, group: str,
                       label: str = "correct_num", min_group: int = 1) -> Calibration:
    taus: dict = {}
    ns: dict = {}
    notes: dict = {}
    for g, sub in cal.groupby(group):
        if len(sub) < min_group:
            taus[g] = np.inf; ns[g] = len(sub); notes[g] = {"reason": f"fewer than {min_group} calibration points"}
            continue
        tau, diag = fit_threshold(sub[signal], sub[label], alpha)
        taus[g] = tau; ns[g] = diag["n_cal"]; notes[g] = diag
    return Calibration(method="mondrian", alpha=alpha, tau=taus, n_cal=ns, notes=notes)
