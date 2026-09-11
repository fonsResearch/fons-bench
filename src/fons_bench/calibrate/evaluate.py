"""Evaluate a fitted calibration on held-out test poses: realised risk and abstention."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fons_bench.calibrate.base import Calibration


def evaluate(cal: Calibration, test: pd.DataFrame, signal: str, group: str | None = None,
             label: str = "correct_num") -> pd.DataFrame:
    """Per-group realised error rate among accepted, and abstention rate. Plus an 'all' row."""
    acc = cal.accept(test[signal].to_numpy(), test[group].to_numpy() if group else None)
    d = test.assign(_accept=acc)
    rows = []
    keys = ([("all", d)] + list(d.groupby(group))) if group else [("all", d)]
    for g, sub in keys:
        n, n_acc = len(sub), int(sub._accept.sum())
        acc_rows = sub[sub._accept]
        rows.append({"group": g, "n_test": n, "n_accepted": n_acc,
                     "abstention": 1.0 - n_acc / n if n else np.nan,
                     "realised_error": float((1.0 - acc_rows[label]).mean()) if n_acc else np.nan,
                     "base_error": float((1.0 - sub[label]).mean()) if n else np.nan,
                     "violates_alpha": bool(n_acc and (1.0 - acc_rows[label]).mean() > cal.alpha)})
    return pd.DataFrame(rows)
