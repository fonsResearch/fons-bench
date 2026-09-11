"""Predictor interface. One implementation per model; each runs in its own env."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from fons_bench.data.reference import ReferenceSystem


@dataclass
class Pose:
    model: str
    seed: int
    ligand: Chem.Mol                     # predicted ligand pose, 3D
    protein_pdb: Path                    # receptor used/produced for this pose
    confidence: dict = field(default_factory=dict)   # ptm, iptm, plddt_ligand, pae_interface, dock_score
    source: str = "generated"            # "public" | "generated"


class Predictor(ABC):
    name: str

    @abstractmethod
    def predict(self, system: ReferenceSystem, seed: int, out_dir: Path) -> list[Pose]:
        """Produce poses for ``system``. Must be cached: never re-run if ``out_dir`` is complete."""
