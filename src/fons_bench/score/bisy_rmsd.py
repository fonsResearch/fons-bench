"""Binding-site superposed, symmetry-corrected ligand RMSD (our re-implementation of the
'BiSyRMSD' that OpenStructure's ``compare-ligand-structures --rmsd`` reports).

Co-folded models live in an arbitrary frame, so a ligand RMSD needs a superposition first.
OpenStructure superposes the *binding site*: reference residues with any atom within a radius
of the ligand, mapped onto the model by chain mapping + sequence alignment, superposed on
backbone atoms. The ligand RMSD is then computed after applying that transform, with
graph-symmetry correction. This module does the same with gemmi + RDKit + spyrmsd so the
release's numbers can be checked without installing OpenStructure.

Known deliberate simplifications versus OST (all stated so disagreement can be diagnosed):
  * binding-site radius 4.0 A on heavy atoms (OST default radius=4.0)
  * superposition on CA atoms only (OST uses backbone; difference is small)
  * chain mapping: best sequence-identity pairing, not OST's full mapping search
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from fons_bench.score.rmsd import symmetric_rmsd

BS_RADIUS = 4.0


def _polymer_chains(st: gemmi.Structure) -> dict[str, tuple[str, list[gemmi.Residue]]]:
    """chain name -> (one-letter sequence, residues) for polymer chains."""
    out = {}
    for ch in st[0]:
        poly = ch.get_polymer()
        if len(poly) == 0:
            continue
        seq = gemmi.one_letter_code([r.name for r in poly])
        out[ch.name] = (seq, list(poly))
    return out


def _het_ligands(st: gemmi.Structure) -> list[gemmi.Residue]:
    return [r for ch in st[0] for r in ch if r.het_flag == "H" and r.name != "HOH" and len(r) > 1]


def _mol_from_residue(res: gemmi.Residue, template: Chem.Mol) -> Chem.Mol:
    """Heavy-atom RDKit mol with the template's graph and the residue's coordinates."""
    heavy = [a for a in res if a.element.name != "H"]
    rw = Chem.RWMol(); conf = Chem.Conformer(len(heavy))
    for i, a in enumerate(heavy):
        idx = rw.AddAtom(Chem.Atom(a.element.atomic_number)); conf.SetAtomPosition(idx, (a.pos.x, a.pos.y, a.pos.z))
    m = rw.GetMol(); m.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(m)
    qp = Chem.AdjustQueryParameters.NoAdjustments(); qp.makeBondsGeneric = True
    tmpl = Chem.RemoveHs(template)
    match = m.GetSubstructMatch(Chem.AdjustQueryProperties(tmpl, qp))
    if len(match) != tmpl.GetNumAtoms():
        raise ValueError(f"model ligand does not match reference graph ({len(match)}/{tmpl.GetNumAtoms()})")
    lig = Chem.Mol(tmpl); lig.RemoveAllConformers()
    c2 = Chem.Conformer(lig.GetNumAtoms())
    for t_idx, m_idx in enumerate(match):
        c2.SetAtomPosition(t_idx, m.GetConformer().GetAtomPosition(m_idx))
    lig.AddConformer(c2, assignId=True)
    return lig


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Return (R, t) minimizing ||R P + t - Q||."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def _align_map(rseq: str, mseq: str) -> dict[int, int]:
    """Map reference residue index -> model residue index via pairwise alignment."""
    aln = gemmi.align_string_sequences(list(rseq), list(mseq), [])
    rmap, ri, mi = {}, 0, 0
    for op in aln.cigar_str().replace("M", "M ").replace("I", "I ").replace("D", "D ").split():
        n, kind = int(op[:-1]), op[-1]
        for _ in range(n):
            if kind == "M":
                rmap[ri] = mi; ri += 1; mi += 1
            elif kind == "I":
                ri += 1
            else:
                mi += 1
    return rmap


def _candidate_mappings(ref_chains: dict, mdl_chains: dict, limit: int = 24) -> list[dict]:
    """All one-to-one reference->model chain assignments among sequence-compatible chains."""
    import itertools
    rnames = list(ref_chains)
    options = []
    for rc in rnames:
        rseq = ref_chains[rc][0]
        compat = [mc for mc, (mseq, _) in mdl_chains.items()
                  if gemmi.align_string_sequences(list(rseq), list(mseq), []).calculate_identity() > 80]
        options.append(compat or list(mdl_chains))
    out = []
    for combo in itertools.product(*options):
        if len(set(combo)) != len(combo):
            continue                      # assignments must be one-to-one
        out.append(dict(zip(rnames, combo)))
        if len(out) >= limit:
            break
    return out or [dict(zip(rnames, list(mdl_chains)))]


def bisy_rmsd(ref_receptor_cif: Path, ref_ligand_sdf: Path, model_cif: Path, radius: float = BS_RADIUS,
              model_ligand_chain: str | None = None) -> dict:
    """Binding-site superposed symmetry-corrected ligand RMSD.

    ``model_ligand_chain`` selects WHICH ligand copy in the model is being compared. Pass it
    whenever the caller knows the assignment (the release records it as
    ``model_ligand_chain_rmsd``). Without it this function falls back to the best-matching copy,
    which silently scores the wrong ligand in any structure containing more than one copy of the
    same molecule - every het residue is typically named ``LIG``, so the name cannot disambiguate.
    """
    ref = gemmi.read_structure(str(ref_receptor_cif)); ref.setup_entities()
    mdl = gemmi.read_structure(str(model_cif)); mdl.setup_entities()
    ref_lig = Chem.MolFromMolFile(str(ref_ligand_sdf), removeHs=False)
    lig_xyz = ref_lig.GetConformer().GetPositions()

    # binding site residues in the reference: any heavy atom within radius of any ligand heavy atom
    ref_chains = _polymer_chains(ref)
    heavy_idx = [a.GetIdx() for a in ref_lig.GetAtoms() if a.GetAtomicNum() > 1]
    lig_xyz = lig_xyz[heavy_idx]
    bs = []  # (chain, index into polymer list)
    for cname, (seq, residues) in ref_chains.items():
        for i, r in enumerate(residues):
            for a in r:
                if a.element.name == "H":
                    continue
                p = np.array([a.pos.x, a.pos.y, a.pos.z])
                if np.min(np.linalg.norm(lig_xyz - p, axis=1)) <= radius:
                    bs.append((cname, i)); break

    # Chain mapping. Homomers make sequence identity useless as a discriminator: two copies of the
    # same chain are 100% identical, so a greedy first-match assignment picks arbitrarily and can
    # superpose the binding site onto the wrong monomer - which yields a *good-looking* backbone
    # fit (the monomers superpose on each other fine) and a wildly wrong ligand RMSD. OpenStructure
    # searches over chain mappings; we do the same and keep the assignment that minimises the
    # ligand RMSD, which is the quantity the mapping exists to serve.
    mdl_chains = _polymer_chains(mdl)

    # candidate model ligand copies: the caller's assigned chain when known, else every copy
    lig_residues = [(ch.name, res) for ch in mdl[0] for res in ch
                    if res.het_flag == "H" and res.name != "HOH" and len(res) > 1
                    and (model_ligand_chain is None or ch.name == model_ligand_chain)]
    if model_ligand_chain is not None and not lig_residues:
        raise ValueError(f"model has no ligand in chain {model_ligand_chain!r}")

    best = None
    for mapping_choice in _candidate_mappings(ref_chains, mdl_chains):
        P, Q = [], []
        for cname, i in bs:
            mc = mapping_choice.get(cname)
            if mc is None or mc not in mdl_chains:
                continue
            rmap = _align_map(ref_chains[cname][0], mdl_chains[mc][0])
            if i not in rmap or rmap[i] >= len(mdl_chains[mc][1]):
                continue
            ra = ref_chains[cname][1][i].find_atom("CA", "*")
            ma = mdl_chains[mc][1][rmap[i]].find_atom("CA", "*")
            if ra is None or ma is None:
                continue
            P.append([ma.pos.x, ma.pos.y, ma.pos.z]); Q.append([ra.pos.x, ra.pos.y, ra.pos.z])
        if len(P) < 3:
            continue
        P_, Q_ = np.array(P), np.array(Q)
        R, t = _kabsch(P_, Q_)
        ca_rmsd = float(np.sqrt(np.mean(np.sum((P_ @ R.T + t - Q_) ** 2, axis=1))))
        for chname, res in lig_residues:
            try:
                mlig = _mol_from_residue(res, ref_lig)
            except ValueError:
                continue
            conf = mlig.GetConformer()
            xyz = conf.GetPositions() @ R.T + t
            for i, p in enumerate(xyz):
                conf.SetAtomPosition(i, p.tolist())
            val = symmetric_rmsd(mlig, ref_lig)
            if best is None or val < best["rmsd"]:
                best = {"rmsd": val, "model_ligand": res.name, "model_ligand_chain": chname,
                        "n_bs_residues": len(bs), "n_superposed": len(P_), "bs_ca_rmsd": ca_rmsd,
                        "chain_mapping": dict(mapping_choice)}
    if best is None:
        raise ValueError("no model ligand matched the reference ligand graph under any chain mapping")
    return best
