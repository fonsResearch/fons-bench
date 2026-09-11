import datetime as dt

import pandas as pd

from fons_bench.shift.binning import assign_bins, merge_small_bins
from fons_bench.shift.similarity import similarity_at_cutoff
from fons_bench.table import _nullable_bool


def test_assign_bins_edges_and_nulls():
    v = pd.Series([0, 29.9, 30, 69.9, 70, 100, None])
    out = assign_bins(v, [0, 30, 70, 100])
    assert out.tolist()[:6] == ["0-30", "0-30", "30-70", "30-70", "70-100", "70-100"]
    assert out.iloc[6] is None


def test_merge_small_bins_reports_merges():
    v = pd.Series([10] * 200 + [50] * 5 + [90] * 200)
    edges, log = merge_small_bins(v, [0, 30, 70, 100], min_count=50)
    assert edges in ([0, 70, 100], [0, 30, 100])
    assert len(log) == 1 and "merged bin 30-70" in log[0]


def test_similarity_at_cutoff_respects_date_and_zero_targets():
    sim = pd.DataFrame({
        "group_key": ["a", "a", "a", "b"],
        "target_release_date": pd.to_datetime(["2019-01-01", "2020-01-01", "2022-01-01", "2023-01-01"]),
        "morgan_tanimoto": [10.0, 40.0, 90.0, 50.0],
        "sucos_shape_pocket_qcov": [30.0, 20.0, 95.0, 60.0],
    })
    out = similarity_at_cutoff(sim, dt.date(2021, 9, 30), pd.Series(["a", "b", "c"])).set_index("group_key")
    assert out.loc["a", "morgan_tanimoto_cutoff"] == 40.0            # post-cutoff 90 excluded
    assert out.loc["a", "sucos_shape_pocket_qcov_cutoff"] == 30.0    # maximised independently of morgan
    assert out.loc["b", "n_targets_before_cutoff"] == 0 and out.loc["b", "morgan_tanimoto_cutoff"] == 0.0
    assert out.loc["c", "n_targets_before_cutoff"] == 0


def test_nullable_bool_handles_mixed_encodings():
    s = pd.Series([True, False, 1.0, 0.0, "True", None, float("nan")], dtype=object)
    out = _nullable_bool(s)
    assert out.tolist()[:5] == [True, False, True, False, True]
    assert pd.isna(out.iloc[5]) and pd.isna(out.iloc[6])
