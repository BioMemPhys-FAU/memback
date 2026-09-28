"""
Check the integrity of MemBack .map/.bnd files against the lipid topologies.

Run as `memback check_maps` (see cli.py) or `python -m memback.check_maps`.
Without options the shipped databases are checked; with -e the .map/.bnd files
of an extension folder are checked the way `memback <input> -e DIR` loads them.

For every lipid in the maps:

  errors (MemBack fails or produces a wrong structure)
    - no [section] in the .bnd, or two different sections with the same name
    - an atom in more than one bead
    - a bead with no heavy atom or more than max_atom_number heavy atoms
    - no CHARMM36 .itp, a mapped heavy atom missing from it, or one of its
      heavy atoms left out of the map
    - beads that differ from the Martini 3 topology (massless virtual beads
      such as PI's C4 are not mapped)
    - a bond to a bead that is not in the map

  warnings (worth a look)
    - hydrogens missing from or unknown to the CHARMM36 .itp
    - bead type, charge or bond list differing from the Martini 3 topology
    - no Martini 3 topology found, beads not connected by bonds,
      duplicated identical sections, .bnd sections without a map

Exit status is 1 when any error is found (with --strict, also on warnings).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from memback import config
from memback.create_map import read_m3_molecules
from memback.helpers import read_bnd
from memback.io.read_sim_metadata import read_itp

SECTION = re.compile(r"^\s*\[\s*(\S+)\s*\]")

# Lipids whose map is deliberately not the Martini 3 topology, so only the
# CHARMM36 and .bnd checks apply. CHOL's ROH and R3-R6 are massless virtual
# sites that still carry atoms, and its bonded network is mostly virtual; the
# CHL1 .bnd uses its own bead graph instead.
M3_CHECK_EXEMPT = {"CHL1"}

def read_sections(path):
    """[(name, [lines])] of a .map/.bnd, keeping duplicated sections."""
    sections = []
    for raw in open(path):
        line = raw.split(";")[0].strip()
        header = SECTION.match(line)
        if header:
            sections.append((header.group(1), []))
        elif line and sections:
            sections[-1][1].append(line)
    return sections


def parse_map_section(lines):
    """{bead: (type, charge, atoms)}, columns read as MemBack's map_reader_full does."""
    beads = {}
    for line in lines:
        tok = line.split()
        try:
            charge, atoms = float(tok[2]), tok[3:]
        except (ValueError, IndexError):
            charge, atoms = 0.0, tok[2:]
        beads[tok[0]] = (tok[1], charge, atoms)
    return beads


def load(map_files, bnd_files, issues):
    """Maps and bonds by residue name; the last section wins, as in MemBack."""
    loaded = []
    for files, kind in ((map_files, "map"), (bnd_files, "bnd")):
        seen = {}
        for path in files:
            for name, lines in read_sections(path):
                if name in seen:
                    same = seen[name] == lines
                    issues[name].append(("warning" if same else "error",
                                         f"{'identical' if same else 'different'} duplicate "
                                         f"[{name}] sections in .{kind} files; the last one is used"))
                seen[name] = lines
        loaded.append(seen)
    maps = {name: parse_map_section(lines) for name, lines in loaded[0].items()}
    bonds = {}
    for path in bnd_files:
        bonds.update({name: data["bonds"] for name, data in read_bnd(path).items()})
    return maps, bonds


def find_aa_itp(resname, itp_dirs):
    for directory in itp_dirs:
        path = Path(directory) / f"{resname}.itp"
        if path.is_file():
            molecules = read_itp(path)
            return molecules.get(resname) or next(iter(molecules.values()), None)
    return None

def check_lipid(resname, beads, bonds, aa, m3, issues, compare_m3=True):
    add = lambda level, msg: issues[resname].append((level, msg))
    is_h = lambda atom: atom.startswith("H")

    # Map on its own
    owner = {}
    for bead, (_, _, atoms) in beads.items():
        for atom in atoms:
            if atom in owner:
                add("error", f"atom {atom} is in beads {owner[atom]} and {bead}")
            owner.setdefault(atom, bead)
        n_heavy = sum(not is_h(a) for a in atoms)
        if not 0 < n_heavy <= config.max_atom_number:
            add("error", f"bead {bead} has {n_heavy} heavy atoms "
                         f"(MemBack supports 1-{config.max_atom_number})")

    # Map against the CHARMM36 topology
    if aa is None:
        add("error", f"no CHARMM36 {resname}.itp found")
    else:
        for heavy, level in ((False, "error"), (True, "warning")):
            kind = "hydrogens" if heavy else "heavy atoms"
            unknown = [a for a in owner if is_h(a) == heavy and a not in aa["atoms"]]
            missing = [a for a in aa["atoms"] if is_h(a) == heavy and a not in owner]
            if unknown:
                add(level, f"{kind} not in the CHARMM36 itp: {unknown}")
            if missing:
                add(level, f"CHARMM36 {kind} missing from the map: {missing}")

    # Bonds against the map
    if bonds is None:
        add("error", "no section in the .bnd files")
        bonds = []
    unknown = [b for b in bonds if not set(b) <= set(beads)]
    if unknown:
        add("error", f"bonds to beads not in the map: {unknown}")
    unbonded = [b for b in beads if not any(b in bond for bond in bonds)]
    if bonds and unbonded:
        add("warning", f"beads without bonds: {unbonded}")

    # Map and bonds against the Martini 3 topology
    if not compare_m3:
        return
    if m3 is None:
        add("warning", "no Martini 3 topology found; bead checks skipped")
        return
    m3_beads = {b["name"]: b for b in m3["beads"]}
    missing = [b for b in m3_beads if b not in beads]
    extra = [b for b in beads if b not in m3_beads]
    if missing:
        add("error", f"Martini 3 beads missing from the map: {missing}")
    if extra:
        add("error", f"map beads not in Martini 3 {m3['name']}: {extra}")
    for bead, (btype, charge, _) in beads.items():
        if bead in m3_beads:
            if btype != m3_beads[bead]["type"]:
                add("warning", f"bead {bead} type {btype}, Martini 3 has {m3_beads[bead]['type']}")
            if charge != m3_beads[bead]["charge"]:
                add("warning", f"bead {bead} charge {charge:g}, Martini 3 has {m3_beads[bead]['charge']:g}")
    ours = {frozenset(b) for b in bonds}
    theirs = {frozenset(b[:2]) for b in m3["bonds"]}
    if ours != theirs:
        add("warning", f"bonds differ from Martini 3: only in .bnd {sorted(map(sorted, ours - theirs))}, "
                       f"only in Martini 3 {sorted(map(sorted, theirs - ours))}")


def check_maps(map_files, bnd_files, itp_dirs, m3_itp_files):
    """Return {resname: [(level, message)]} for every lipid in map_files."""
    issues = defaultdict(list)
    maps, bonds = load(map_files, bnd_files, issues)

    m3 = {}
    for path in m3_itp_files:
        m3.update(read_m3_molecules(path))

    for resname in bonds:
        if resname not in maps:
            issues[resname].append(("warning", ".bnd section without a map"))
    for resname, beads in maps.items():
        m3_name = config.charmm_to_martini3.get(resname, resname)
        check_lipid(resname, beads, bonds.get(resname), find_aa_itp(resname, itp_dirs),
                    m3.get(m3_name), issues, compare_m3=resname not in M3_CHECK_EXEMPT)
    return {name: issues.get(name, []) for name in [*maps, *(b for b in bonds if b not in maps)]}

def build_parser(prog="memback check_maps"):
    parser = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--extension", type=Path, metavar="DIR",
                        help="check the .map/.bnd files of this extension folder instead of the "
                             "shipped databases; its .itp files are used before the shipped ones")
    parser.add_argument("--m3itp", nargs="+", type=Path, default=[], metavar="ITP",
                        help="extra Martini 3 .itp files to compare against "
                             "(the shipped ones are always used)")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 on warnings too")
    parser.add_argument("-v", "--verbose", action="store_true", help="also list lipids without issues")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    itp_dirs = [config.itp_db_path]
    if args.extension:
        if not args.extension.is_dir():
            raise SystemExit(f"error: extension path is not a directory: {args.extension}")
        map_files = sorted(args.extension.glob("*.map"))
        bnd_files = sorted(args.extension.glob("*.bnd"))
        itp_dirs.insert(0, args.extension)
        if not map_files:
            raise SystemExit(f"error: no .map files in {args.extension}")
    else:
        map_files, bnd_files = [Path(config.map_path)], [Path(config.bond_map_path)]
    for path in args.m3itp:
        if not path.is_file():
            raise SystemExit(f"error: Martini 3 itp not found: {path}")
    m3_itps = sorted(Path(config.itp_m3_db_path).glob("*.itp")) + args.m3itp

    print(f"Checking {', '.join(map(str, map_files + bnd_files))}")
    results = check_maps(map_files, bnd_files, itp_dirs, m3_itps)

    counts = {"error": 0, "warning": 0}
    for resname, found in results.items():
        levels = {level for level, _ in found}
        for level in levels:
            counts[level] += 1
        if found or args.verbose:
            print(f"\n[{resname}] {'OK' if not found else ''}".rstrip())
            for level, message in sorted(found, key=lambda x: x[0]):
                print(f"  {level.upper():7s} {message}")

    print(f"\n{len(results)} lipids checked: {counts['error']} with errors, "
          f"{counts['warning']} with warnings.")
    failed = counts["error"] or (args.strict and counts["warning"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
