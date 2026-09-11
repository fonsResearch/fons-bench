"""PoseBusters validity checks, one boolean per check plus the AND."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from posebusters import PoseBusters


def posebusters_flags(pred_sdf: Path, ref_ligand_sdf: Path, protein_pdb: Path) -> dict[str, bool | None]:
    """Run the 'redock' PoseBusters config and return ``{"pb_<check>": bool, ..., "pb_valid": bool}``.

    Column names come straight from PoseBusters' own test names, lower-cased with spaces
    replaced by underscores. A check PoseBusters could not run is None, never False.
    """
    pb = PoseBusters(config="redock")
    df: pd.DataFrame = pb.bust(mol_pred=str(pred_sdf), mol_true=str(ref_ligand_sdf), mol_cond=str(protein_pdb))
    row = df.iloc[0]
    out: dict[str, bool | None] = {}
    for name, val in row.items():
        key = "pb_" + (str(name).strip().lower().replace(" ", "_").replace("-", "_")
                       .replace("≤", "le").replace("å", "a"))   # ascii column names only
        out[key] = None if pd.isna(val) else bool(val)
    known = [v for v in out.values() if v is not None]
    out["pb_valid"] = all(known) if known else None
    return out
