"""Ligand strain: MMFF94 energy of the pose minus energy after a local relaxation, kcal/mol.

The relaxation is a *local* minimization started from the pose (no global search), so the
number reports how far the pose sits above its nearest local minimum. Hydrogens are added
with coordinates before evaluation because MMFF needs them.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def mmff_strain(mol: Chem.Mol, max_iters: int = 2000) -> float | None:
    m = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    if ff is None:
        return None
    e_pose = ff.CalcEnergy()
    ff.Minimize(maxIts=max_iters)
    e_relaxed = ff.CalcEnergy()
    return float(e_pose - e_relaxed)
