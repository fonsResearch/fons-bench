"""Symmetry-corrected heavy-atom RMSD between a predicted and a reference ligand pose.

Uses spyrmsd's graph-isomorphism RMSD. Both molecules are reduced to heavy atoms so that
differing protonation states between prediction and crystal do not matter; the comparison is
then between heavy-atom graphs. Coordinates are compared as-is (no superposition): the
poses live in the same receptor frame.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from spyrmsd import rmsd as srmsd
from spyrmsd.molecule import Molecule


def _heavy(mol: Chem.Mol) -> Chem.Mol:
    return Chem.RemoveHs(mol, sanitize=False)


def symmetric_rmsd(pred: Chem.Mol, ref: Chem.Mol) -> float:
    """Heavy-atom, symmetry-corrected RMSD in Angstrom. No alignment.

    Raises ValueError if the heavy-atom graphs are not isomorphic (different molecule).
    """
    p, r = _heavy(pred), _heavy(ref)
    if p.GetNumAtoms() != r.GetNumAtoms():
        raise ValueError(f"heavy-atom count differs: pred={p.GetNumAtoms()} ref={r.GetNumAtoms()}")
    mp, mr = Molecule.from_rdkit(p), Molecule.from_rdkit(r)
    # strip=True is a no-op here (already heavy) but keeps intent explicit
    mp.strip(); mr.strip()
    val = srmsd.symmrmsd(
        mp.coordinates, mr.coordinates,
        mp.atomicnums, mr.atomicnums,
        mp.adjacency_matrix, mr.adjacency_matrix,
        center=False, minimize=False,
    )
    return float(np.asarray(val).min())
