"""smina CPU docking baseline and rescoring.

Docking box is defined around the crystal ligand (``autobox_ligand``). That gives smina the
pocket location, which co-folding models do not receive; this is the standard 'known pocket'
docking protocol and the asymmetry is stated in the data card.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from rdkit import Chem

from fons_bench.data.reference import ReferenceSystem
from fons_bench.predict.base import Pose, Predictor


def _affinity(mol: Chem.Mol, log: Path) -> float:
    """Top-mode affinity. Some smina builds omit the SDF property block, so fall back to the log."""
    if mol.HasProp("minimizedAffinity"):
        return float(mol.GetProp("minimizedAffinity"))
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "1":
            return float(parts[1])
    raise ValueError(f"no affinity in {log}")


class SminaPredictor(Predictor):
    name = "smina"

    def __init__(self, exe: Path | str, exhaustiveness: int = 8, autobox_add: float = 4.0):
        self.exe = str(exe)
        self.exhaustiveness = exhaustiveness
        self.autobox_add = autobox_add

    def predict(self, system: ReferenceSystem, seed: int, out_dir: Path) -> list[Pose]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_sdf = out_dir / "pose.sdf"
        log = out_dir / "smina.log"
        if not out_sdf.exists():
            start = system.ligand_start_conf_sdf or system.ligand_sdf
            cmd = [
                self.exe,
                "-r", str(system.protein_pdb),
                "-l", str(start),
                "--autobox_ligand", str(system.ligand_sdf),
                "--autobox_add", str(self.autobox_add),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", "1",
                "--seed", str(seed),
                "-o", str(out_sdf),
            ]
            with open(log, "w") as fh:
                subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
        mol = Chem.MolFromMolFile(str(out_sdf), removeHs=False)
        if mol is None:
            raise ValueError(f"smina output unparsable: {out_sdf}")
        score = _affinity(mol, log)
        return [Pose(model=self.name, seed=seed, ligand=mol, protein_pdb=system.protein_pdb,
                     confidence={"dock_score": score}, source="generated")]

    def rescore(self, protein_pdb: Path, ligand_sdf: Path) -> float:
        """smina --score_only affinity for an existing pose (kcal/mol)."""
        cmd = [self.exe, "-r", str(protein_pdb), "-l", str(ligand_sdf), "--score_only"]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("Affinity:"):
                return float(line.split()[1])
        raise ValueError("no Affinity line in smina --score_only output")
