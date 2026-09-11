"""Ligand novelty: max ECFP4 (radius 2, 2048 bits) Tanimoto to any training-set ligand."""
from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(mol: Chem.Mol):
    return _GEN.GetFingerprint(Chem.RemoveHs(mol))


def max_tanimoto(query: Chem.Mol, train_fps) -> float | None:
    """Return max Tanimoto of ``query`` to ``train_fps``; None if the training set is empty."""
    train_fps = list(train_fps)
    if not train_fps:
        return None
    return float(max(DataStructs.BulkTanimotoSimilarity(ecfp4(query), train_fps)))


def fps_from_smiles(smiles: Iterable[str]):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(ecfp4(m))
    return out
