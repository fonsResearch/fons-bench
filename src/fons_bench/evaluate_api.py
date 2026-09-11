"""`fons-bench evaluate`: score a user's own predictions and apply the calibrated thresholds.

Runs without a GPU and without downloading the benchmark: the thresholds ship inside the
package (`fons_bench/artifacts/calibration.json`).

Input is a CSV with one row per prediction. Required columns:

    system_id            any identifier you use
    signal               your model's confidence for that prediction (higher = more confident)

Optional but strongly recommended:

    pocket_similarity    SuCOS-pocket similarity (0-100) to the nearest structure released
                         before your model's training cutoff. Without it, only the marginal
                         threshold can be applied, and the whole point of this benchmark is
                         that the marginal threshold is not trustworthy under shift.
    correct              0/1 ground truth, if you have it; enables realised-error reporting.

The output reports, per requested alpha, which predictions the marginal and group-conditional
rules accept, and the resulting abstention rate.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


def load_calibration() -> dict:
    with resources.files("fons_bench.artifacts").joinpath("calibration.json").open() as fh:
        return json.load(fh)


def assign_bin(values: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    idx = np.clip(np.searchsorted(edges, values.to_numpy(dtype=float), side="right") - 1, 0, len(labels) - 1)
    out = pd.Series([labels[i] for i in idx], index=values.index, dtype="object")
    out[values.isna()] = None
    return out


def evaluate_predictions(csv: Path, model: str, alpha: float = 0.1) -> pd.DataFrame:
    cal = load_calibration()
    if model not in cal["models"]:
        raise SystemExit(f"no calibration for model '{model}'. Available: {', '.join(sorted(cal['models']))}")
    entry = cal["models"][model]
    key = str(alpha)
    if key not in entry["alphas"]:
        raise SystemExit(f"no calibration at alpha={alpha}. Available: {', '.join(entry['alphas'])}")
    thr = entry["alphas"][key]

    df = pd.read_csv(csv)
    if "signal" not in df.columns:
        raise SystemExit("input CSV needs a `signal` column (higher = more confident)")
    df = df.copy()

    m_tau = thr["marginal_tau"]
    df["accept_marginal"] = False if m_tau is None else df["signal"] >= m_tau

    if "pocket_similarity" in df.columns:
        df["shift_bin"] = assign_bin(df["pocket_similarity"], cal["bin_edges"])
        taus = thr["mondrian_tau"]
        df["accept_conditional"] = [
            False if (b is None or taus.get(b) is None) else (s >= taus[b])
            for b, s in zip(df["shift_bin"], df["signal"])
        ]
        df["bin_certifiable"] = [b is not None and taus.get(b) is not None for b in df["shift_bin"]]
    else:
        df["shift_bin"] = None
        df["accept_conditional"] = pd.NA
        df["bin_certifiable"] = pd.NA
    return df


def summarise(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for rule in ["accept_marginal", "accept_conditional"]:
        if df[rule].isna().all():
            continue
        acc = df[rule].fillna(False).astype(bool)
        row = {"rule": rule.replace("accept_", ""), "n": len(df), "n_accepted": int(acc.sum()),
               "abstention": 1 - acc.mean()}
        if "correct" in df.columns:
            row["realised_error"] = float(1 - df.loc[acc, "correct"].mean()) if acc.any() else np.nan
            row["exceeds_alpha"] = bool(acc.any() and (1 - df.loc[acc, "correct"].mean()) > alpha)
        rows.append(row)
    return pd.DataFrame(rows)
