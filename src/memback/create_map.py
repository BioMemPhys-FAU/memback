"""
Create a MemBack .map/.bnd pair for a lipid from its Martini 3 and CHARMM36 topologies.

Run as `memback create_map ...` (see cli.py) or `python -m memback.create_map ...`.

Inputs:
  --m3itp   Martini 3 .itp (single- or multi-molecule; pick one with --m3name)
            → bead names, types, charges, CG bonds and angles
  --aaitp   CHARMM36 GROMACS .itp of the same lipid → atom names and bonds
  --refmap  map holding a lipid with the same headgroup (default: MemBack's
            shipped all_maps.map, reference chosen automatically)

Headgroup beads (non-carbon bead types) copy their atoms from the reference
lipid. Tail beads (C*, SC*, TC* types) take consecutive carbons from their
CHARMM chain: 4 per R-bead, 3 per S-bead, 2 per T-bead, plus one for the bead
Martini 3 marks as overmapped with "_5long" bonds. Hydrogens follow their
parent carbon. Massless virtual beads are skipped.

Example:
    memback create_map --m3itp martini_v3.0.0_phospholipids_PC_v2.itp \\
        --m3name POPC --aaitp POPC.itp -o my_lipids/POPC

Which CHARMM chain belongs to each Martini tail chain (A, B, …) is read from
the reference lipid's tail beads (e.g. sphingomyelin A → S, B → F). Override
with --chainorder if needed, e.g.  --chainorder S F

Always inspect the result; the script prints what it could not assign.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from memback import config
from memback.helpers import map_reader_full, read_bnd
from memback.io.read_sim_metadata import read_itp

TAIL_TYPE = re.compile(r"^[ST]?C\d")            # C1, C4h, SC1, TC4 …
CARBONS_PER_BEAD = {"S": 3, "T": 2}              # R-beads (no prefix): 4
OVERMAP_SUFFIX = "_5long"                        # b_GL_C1_glyc_5long …

def read_m3_molecules(path):
    """
    Parse every Martini 3 molecule in an .itp: beads (name, type, charge),
    bonds with their named bond type, and angles. Massless virtual beads are
    removed; their names are kept in "virtual".
    """
    molecules = {}
    mol = section = None
    for raw in open(path):
        line = raw.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        header = re.match(r"\[\s*(\w+)\s*\]", line)
        if header:
            section = header.group(1).lower()
            continue
        tok = line.split()
        if section == "moleculetype":
            mol = {"name": tok[0], "beads": [], "bonds": [], "angles": [],
                   "index": {}, "virtual": set()}
            molecules[tok[0]] = mol
        elif mol is None or not tok[0].isdigit():
            continue
        elif section == "atoms":
            bead = {"name": tok[4], "type": tok[1], "charge": float(tok[6])}
            mol["beads"].append(bead)
            mol["index"][int(tok[0])] = tok[4]
            if len(tok) >= 8 and float(tok[7]) == 0:          # explicit zero mass
                mol["virtual"].add(tok[4])
        elif section.startswith("virtual_sites"):
            mol["virtual"].add(mol["index"][int(tok[0])])
        elif section in ("bonds", "constraints"):
            # constraints are CG connectivity too; skip pairs already listed
            # (e.g. PI ring: FLEXIBLE bonds and constraints name the same pairs)
            a, b = mol["index"][int(tok[0])], mol["index"][int(tok[1])]
            if any({a, b} == {x, y} for x, y, _ in mol["bonds"]):
                continue
            bond_type = tok[2] if len(tok) > 2 and not tok[2].isdigit() else ""
            mol["bonds"].append((a, b, bond_type))
        elif section == "angles":
            mol["angles"].append(tuple(mol["index"][int(t)] for t in tok[:3]))

    for mol in molecules.values():
        virtual = mol["virtual"]
        mol["beads"] = [b for b in mol["beads"] if b["name"] not in virtual]
        mol["bonds"] = [b for b in mol["bonds"] if not set(b[:2]) & virtual]
        mol["angles"] = [a for a in mol["angles"] if not set(a) & virtual]
    return molecules


def read_m3_itp(path, molname=None):
    """One Martini 3 molecule from an .itp (see read_m3_molecules)."""
    mol = pick_molecule(read_m3_molecules(path), molname, path, "--m3name")
    if mol["virtual"]:
        print(f"Skipping massless virtual beads: {sorted(mol['virtual'])}")
    return mol


def read_aa_itp(path, molname=None):
    """CHARMM36 molecule via MemBack's itp reader: atom names and bonds."""
    molecules = read_itp(path)
    for name, mol in molecules.items():
        mol["name"] = name
    return pick_molecule(molecules, molname, path, "--aaname")


def pick_molecule(molecules, molname, path, flag):
    if molname is None and len(molecules) == 1:
        return next(iter(molecules.values()))
    if molname in molecules:
        return molecules[molname]
    sys.exit(f"error: {path} has molecules {list(molecules)}; "
             f"choose one with {flag}{f' ({molname!r} not found)' if molname else ''}.")

def choose_reference(refmap, refres, head_beads, aa_atoms):
    """
    Return (name, {bead: atoms}) of the reference lipid for the headgroup.
    Without --refres, candidates are lipids in refmap with exactly the same
    headgroup beads whose headgroup atoms all exist in the CHARMM molecule;
    the one leaving the fewest non-tail CHARMM atoms unassigned wins, and ties
    go to the headgroup layout shared by most candidates.
    """
    mapping = map_reader_full(refmap, hydrogens=True)
    aa_set = set(aa_atoms)

    def head_atoms_missing(res):
        beads = mapping[res]["atoms"]
        if not set(head_beads) <= set(beads):
            return None
        return [a for b in head_beads for a in beads[b] if a not in aa_set]

    def uncovered(res):
        """CHARMM atoms outside the tail carbons that the reference headgroup leaves out
        (tail hydrogens add the same count for every candidate)."""
        covered = {a for b in head_beads for a in mapping[res]["atoms"][b]}
        return sum(a not in covered and charmm_chain(a) == "X" for a in aa_atoms)

    if refres:
        if refres not in mapping:
            sys.exit(f"error: {refres} not found in {refmap}")
        missing = head_atoms_missing(refres)
        if missing is None:
            sys.exit(f"error: {refres} lacks headgroup beads {head_beads}")
        if missing:
            sys.exit(f"error: headgroup atoms of {refres} missing from CHARMM itp: {missing}")
        return refres, mapping[refres]["atoms"]

    candidates = [res for res in mapping
                  if {b for b, t in mapping[res]["bead_type"].items() if not TAIL_TYPE.match(t)}
                  == set(head_beads) and head_atoms_missing(res) == []]
    if candidates:
        fewest = min(map(uncovered, candidates))
        candidates = [res for res in candidates if uncovered(res) == fewest]
        # among equals, use the headgroup layout most reference lipids share
        layout = {res: tuple(tuple(a for a in mapping[res]["atoms"][b] if not a.startswith("H"))
                             for b in head_beads) for res in candidates}
        layouts = list(layout.values())
        best = max(candidates, key=lambda res: layouts.count(layout[res]))
        return best, mapping[best]["atoms"]
    sys.exit(f"error: no lipid in {refmap} matches headgroup beads {head_beads} "
             f"with atoms present in the CHARMM itp. Provide --refmap/--refres.")

def charmm_chain(atom):
    """C22, C318 → A, B (digit after C = chain); C1S, C19F → S, F; else X."""
    m = re.match(r"C([2-9])\d+$", atom)
    if m:
        return chr(ord("A") + int(m.group(1)) - 2)
    m = re.match(r"C\d+([A-Z])$", atom)
    return m.group(1) if m else "X"


def bead_chain(bead):
    """C1A, D2B → A, B; beads without a trailing letter → X."""
    return bead[-1] if bead[-1].isalpha() else "X"


def overmapped_beads(tail_names, bonds):
    """
    The bead shared by every "_5long" bond of a chain carries one extra carbon
    (e.g. DOPC: GL1-C1A and C1A-D2A are _5long → C1A maps 5 carbons).
    """
    tails = set(tail_names)
    per_chain = defaultdict(list)
    for a, b, bond_type in bonds:
        if bond_type.endswith(OVERMAP_SUFFIX):
            for chain in {bead_chain(x) for x in (a, b) if x in tails}:
                per_chain[chain].append({a, b})
    result = set()
    for chain, pairs in per_chain.items():
        shared = {b for b in set.intersection(*pairs) & tails if bead_chain(b) == chain}
        if len(shared) == 1:
            result |= shared
        else:
            print(f"WARNING: cannot place the extra carbon of chain {chain} "
                  f"from {OVERMAP_SUFFIX} bonds {pairs}")
    return result


def chain_order_from_reference(ref_atoms, named):
    """CHARMM chain of each Martini chain, read off the reference lipid's tail beads."""
    order = []
    for chain in named:
        keys = [charmm_chain(a) for bead, atoms in ref_atoms.items() if bead_chain(bead) == chain
                for a in atoms if not a.startswith("H") and charmm_chain(a) != "X"]
        if not keys:
            return None
        order.append(max(set(keys), key=keys.count))
    return order


def assign_tails(tail_beads, cg_bonds, aa, claimed, chain_order, ref_atoms):
    heavy = [a for a in aa["atoms"] if not a.startswith("H") and a not in claimed]
    hydrogens = defaultdict(list)
    for i, j in aa["bonds"]:
        if i.startswith("H") != j.startswith("H"):
            parent, h = (j, i) if i.startswith("H") else (i, j)
            hydrogens[parent].append(h)

    chains = defaultdict(list)
    for atom in heavy:
        chains[charmm_chain(atom)].append(atom)

    # Martini chain letter → CHARMM chain key (X stays X)
    named = [c for c in dict.fromkeys(bead_chain(b["name"]) for b in tail_beads) if c != "X"]
    keys = (chain_order or chain_order_from_reference(ref_atoms, named)
            or sorted(k for k in chains if k != "X"))
    if len(keys) != len(named):
        sys.exit(f"error: Martini tail chains {named} vs CHARMM chains {keys}; "
                 f"set --chainorder with one CHARMM chain per Martini chain.")
    chain_key = {"X": "X", **dict(zip(named, keys))}
    print(f"Chain mapping (Martini → CHARMM): {chain_key}")

    extra = overmapped_beads([b["name"] for b in tail_beads], cg_bonds)
    print(f"Overmapped beads (+1 carbon): {sorted(extra)}")

    carbons, cursor, last = {}, defaultdict(int), {}
    for bead in tail_beads:
        name, chain = bead["name"], bead_chain(bead["name"])
        n = CARBONS_PER_BEAD.get(bead["type"][0], 4) + (name in extra)
        pool = chains[chain_key[chain]]
        carbons[name] = pool[cursor[chain]:cursor[chain] + n]
        cursor[chain] += n
        last[chain] = name

    for chain, name in last.items():
        leftover = chains[chain_key[chain]][cursor[chain]:]
        if leftover:
            print(f"WARNING: {len(leftover)} leftover carbon(s) {leftover} added to {name}")
            carbons[name] += leftover

    return {bead: [x for c in cs for x in (c, *hydrogens[c])] for bead, cs in carbons.items()}

def write_files(name, m3, bead_atoms, prefix):
    map_lines = [f"[{name}]"]
    for b in m3["beads"]:
        charge = f" {b['charge']:g}" if b["charge"] else ""
        map_lines.append(f"{b['name']} {b['type']}{charge} {' '.join(bead_atoms[b['name']])}")
    bnd_lines = ([f"[{name}]"] + [f"{a} {b}" for a, b, _ in m3["bonds"]] + [""]
                 + [" ".join(a) for a in m3["angles"]])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext, lines in (("map", map_lines), ("bnd", bnd_lines)):
        path = prefix.with_name(f"{prefix.name}.{ext}")
        path.write_text("\n".join(lines) + "\n")
        print(f"Written: {path}")


def check_output(name, prefix, aa):
    """Read the files back with MemBack's readers; report problems, return True if none."""
    mapping = map_reader_full(prefix.with_name(f"{prefix.name}.map"), hydrogens=True)[name]
    bnd = read_bnd(prefix.with_name(f"{prefix.name}.bnd"))[name]
    ok = True
    mapped = [a for atoms in mapping["atoms"].values() for a in atoms]
    unassigned = [a for a in aa["atoms"] if a not in mapped]
    duplicated = sorted({a for a in mapped if mapped.count(a) > 1})
    for label, atoms in (("Unassigned atoms", unassigned), ("Atoms in several beads", duplicated)):
        if atoms:
            ok = False
            print(f"WARNING: {label} ({len(atoms)}): {atoms}")
    for bead, atoms in mapping["atoms"].items():
        n_heavy = sum(not a.startswith("H") for a in atoms)
        if not 0 < n_heavy <= config.max_atom_number:
            ok = False
            print(f"WARNING: bead {bead} has {n_heavy} heavy atoms "
                  f"(MemBack supports 1-{config.max_atom_number})")
    unknown = [b for b in bnd["bonds"] if not set(b) <= set(mapping["atoms"])]
    if unknown:
        ok = False
        print(f"WARNING: bonds to unknown beads: {unknown}")
    print("All checks passed." if ok else "Check the warnings above before using the map.")
    return ok

def build_parser(prog="memback create_map"):
    parser = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m3itp", required=True, type=Path, help="Martini 3 .itp")
    parser.add_argument("--m3name", help="molecule in --m3itp (needed if it holds several)")
    parser.add_argument("--aaitp", required=True, type=Path, help="CHARMM36 .itp")
    parser.add_argument("--aaname", help="molecule in --aaitp (needed if it holds several)")
    parser.add_argument("--refmap", default=Path(config.map_path), type=Path,
                        help="map with a lipid of the same headgroup (default: MemBack's all_maps.map)")
    parser.add_argument("--refres", help="lipid in --refmap to copy the headgroup from (default: automatic)")
    parser.add_argument("--name", help="residue name written to the files (default: CHARMM molecule name)")
    parser.add_argument("--chainorder", nargs="+", metavar="CHAIN",
                        help="CHARMM chain for each Martini tail chain A, B, … "
                             "(default: taken from the reference lipid, e.g. S F for sphingolipids)")
    parser.add_argument("-o", "--output", metavar="PREFIX",
                        help="output prefix, writes PREFIX.map and PREFIX.bnd (default: ./<name>)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for label, path in (("Martini 3 itp", args.m3itp), ("CHARMM36 itp", args.aaitp),
                        ("reference map", args.refmap)):
        if not path.is_file():
            raise SystemExit(f"error: {label} not found: {path}")

    m3 = read_m3_itp(args.m3itp, args.m3name)
    aa = read_aa_itp(args.aaitp, args.aaname)
    name = args.name or aa["name"]
    prefix = Path(args.output or name)
    print(f"Martini 3 : {m3['name']} ({len(m3['beads'])} beads)   CHARMM36 : {aa['name']} ({len(aa['atoms'])} atoms)")

    head = [b["name"] for b in m3["beads"] if not TAIL_TYPE.match(b["type"])]
    tails = [b for b in m3["beads"] if TAIL_TYPE.match(b["type"])]
    refres, ref_atoms = choose_reference(args.refmap, args.refres, head, aa["atoms"])
    print(f"Headgroup : {head} copied from {refres}")

    bead_atoms = {b: ref_atoms[b] for b in head}
    claimed = {a for b in head for a in ref_atoms[b]}
    bead_atoms.update(assign_tails(tails, m3["bonds"], aa, claimed, args.chainorder, ref_atoms))

    write_files(name, m3, bead_atoms, prefix)
    return 0 if check_output(name, prefix, aa) else 1

if __name__ == "__main__":
    sys.exit(main())
