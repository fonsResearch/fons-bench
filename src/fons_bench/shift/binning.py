"""Assign shift bins from a similarity column and a list of edges; merge underpopulated bins.

Bins are half-open [lo, hi) except the last, which is closed at the top so a score of exactly
100 lands in the highest bin. Labels are "lo-hi" strings so they read the same in tables and
figures. Missing similarity -> None (never a sentinel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bin_labels(edges: list[float]) -> list[str]:
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]


def assign_bins(values: pd.Series, edges: list[float]) -> pd.Series:
    edges = list(edges)
    labels = bin_labels(edges)
    idx = np.searchsorted(edges, values.to_numpy(dtype=float), side="right") - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    out = pd.Series([labels[i] for i in idx], index=values.index, dtype="object")
    out[values.isna()] = None
    return out


def merge_small_bins(values: pd.Series, edges: list[float], min_count: int,
                     unit_ids: pd.Series | None = None) -> tuple[list[float], list[str]]:
    """Drop interior edges until every bin holds >= ``min_count`` distinct units.

    ``unit_ids`` (e.g. system_id) decides what a "system" is; defaults to one row = one unit.
    Merges the smallest bin into its smaller neighbour, one step at a time, and returns the
    final edges plus a log of the merges performed so the report can state them.
    """
    edges = list(edges)
    log: list[str] = []
    units = unit_ids if unit_ids is not None else pd.Series(range(len(values)), index=values.index)
    while len(edges) > 2:
        b = assign_bins(values, edges)
        counts = pd.DataFrame({"bin": b, "u": units}).dropna().groupby("bin")["u"].nunique()
        counts = counts.reindex(bin_labels(edges), fill_value=0)
        if counts.min() >= min_count:
            break
        i = int(np.argmin(counts.to_numpy()))
        # merge with the smaller neighbour (or the only neighbour at the ends)
        if i == 0:
            drop = 1
        elif i == len(counts) - 1:
            drop = i
        else:
            drop = i if counts.iloc[i - 1] <= counts.iloc[i + 1] else i + 1
        log.append(f"merged bin {counts.index[i]} (n={counts.iloc[i]}) by removing edge {edges[drop]:g}")
        edges.pop(drop)
    return edges, log
