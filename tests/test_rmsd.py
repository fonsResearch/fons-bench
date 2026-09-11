"""RMSD sanity tests on PoseBusters' own test fixture (1OF6_DTY).

The 0.78 A value was produced by this code and checked by eye against the fixture, not
against an independent tool yet. Replace with hand-verified values in Phase 1.
"""
from pathlib import Path

import pytest
from rdkit import Chem

from fons_bench.score.rmsd import symmetric_rmsd

FIX = Path(__file__).parent / "fixtures" / "1OF6_DTY"


def _load(name):
    return Chem.MolFromMolFile(str(FIX / name), removeHs=False)


def test_self_rmsd_is_zero():
    ref = _load("1OF6_DTY_crystal.sdf")
    assert symmetric_rmsd(ref, ref) == pytest.approx(0.0, abs=1e-6)


def test_vina_pose_rmsd():
    ref = _load("1OF6_DTY_crystal.sdf")
    pred = _load("1OF6_DTY_vina.sdf")
    assert symmetric_rmsd(pred, ref) == pytest.approx(0.778, abs=0.01)


def test_protonation_difference_ignored():
    ref = _load("1OF6_DTY_crystal.sdf")
    pred = Chem.AddHs(_load("1OF6_DTY_vina.sdf"), addCoords=True)
    assert symmetric_rmsd(pred, ref) == pytest.approx(0.778, abs=0.01)


def test_chain_mapping_search_helpers():
    """Regression guard for the homomer chain-mapping bug.

    In a homomeric complex every protein copy is 100% identical, so assigning reference chains to
    model chains by sequence identity picks arbitrarily. Superposing the binding site onto the
    wrong monomer gives an excellent backbone fit and a ligand RMSD wrong by tens of angstroms -
    a silent failure that looks like success. `_candidate_mappings` must therefore enumerate both
    assignments for a two-copy homomer, so the caller can score each and keep the best.
    """
    from fons_bench.score.bisy_rmsd import _candidate_mappings

    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    ref = {"1.A": (seq, []), "1.B": (seq, [])}
    mdl = {"A": (seq, []), "B": (seq, [])}
    maps = _candidate_mappings(ref, mdl)
    assert {"1.A": "A", "1.B": "B"} in maps
    assert {"1.A": "B", "1.B": "A"} in maps          # the assignment a greedy rule would miss
    for m in maps:
        assert len(set(m.values())) == len(m)        # one-to-one


def test_candidate_mappings_are_one_to_one_with_distinct_sequences():
    from fons_bench.score.bisy_rmsd import _candidate_mappings

    a, b = "ACDEFGHIKLMNPQRSTVWY" * 3, "WYWYWYKLKLKLNPNPNPQR" * 3
    maps = _candidate_mappings({"1.A": (a, []), "1.B": (b, [])}, {"A": (a, []), "B": (b, [])})
    assert maps and all(len(set(m.values())) == len(m) for m in maps)
