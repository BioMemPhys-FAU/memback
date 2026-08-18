from pathlib import Path
import re

def read_hdb(filename):
    residues = {}
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(';')]

    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        resname = tokens[0]
        nentries = int(tokens[1])
        i += 1

        entries = []
        for _ in range(nentries):
            parts = lines[i].split()
            nr = int(parts[0])
            tp = int(parts[1])
            atoms = parts[2:]
            entries.append({
                "nr": nr,
                "tp": tp,
                "control_atoms": atoms
            })
            i += 1

        residues[resname] = entries

    return residues

def read_itp(filepath):
    known_blocks = {"atoms", "bonds", "angles", "dihedrals", "pairs",
                    "exclusions", "moleculetype", "system", "molecules",
                    "position_restraints", "cmap", "dihedral_restraints"}
    improper_funct = {2, 4}

    residues   = {}
    current_mol   = None
    current_block = None
    idx_to_name   = {}
    section_pattern = re.compile(r'^\s*\[\s*(\S+)\s*\]')
    comment_pattern = re.compile(r';.*$')

    with open(filepath, "r") as f:
        for raw_line in f:
            line = comment_pattern.sub('', raw_line).strip()
            if not line:
                continue

            match = section_pattern.match(line)
            if match:
                section_name = match.group(1)
                lower = section_name.lower()
                if lower in known_blocks:
                    current_block = lower
                    if lower == "moleculetype":
                        current_mol = None
                        idx_to_name = {}
                else:
                    current_block = None
                continue

            if current_block is None:
                continue

            tokens = line.split()

            if current_block == "moleculetype" and len(tokens) >= 1:
                current_mol = tokens[0]
                idx_to_name = {}
                residues[current_mol] = {
                    "atoms":     [],
                    "bonds":     [],
                    "angles":    [],
                    "dihedrals": [],
                    "impropers": [],
                    "pairs":     [],
                    "dihedral_restraints": [],
                }

            if current_mol is None:
                continue

            elif current_block == "atoms" and len(tokens) >= 5:
                try:
                    idx       = int(tokens[0])
                    atom_name = tokens[4]
                    idx_to_name[idx] = atom_name
                    residues[current_mol]["atoms"].append(atom_name)
                except ValueError:
                    continue

            elif current_block == "bonds" and len(tokens) >= 2:
                try:
                    i, j = int(tokens[0]), int(tokens[1])
                    if i in idx_to_name and j in idx_to_name:
                        residues[current_mol]["bonds"].append(
                            (idx_to_name[i], idx_to_name[j])
                        )
                except ValueError:
                    continue

            elif current_block == "pairs" and len(tokens) >= 2:
                try:
                    i, j = int(tokens[0]), int(tokens[1])
                    if i in idx_to_name and j in idx_to_name:
                        residues[current_mol]["pairs"].append(
                            (idx_to_name[i], idx_to_name[j])
                        )
                except ValueError:
                    continue

            elif current_block == "angles" and len(tokens) >= 3:
                try:
                    i, j, k = int(tokens[0]), int(tokens[1]), int(tokens[2])
                    if all(x in idx_to_name for x in (i, j, k)):
                        residues[current_mol]["angles"].append(
                            (idx_to_name[i], idx_to_name[j], idx_to_name[k])
                        )
                except ValueError:
                    continue

            elif current_block == "dihedrals" and len(tokens) >= 4:
                try:
                    i, j, k, l = (int(tokens[x]) for x in range(4))
                    funct = int(tokens[4]) if len(tokens) >= 5 else 1
                    if not all(x in idx_to_name for x in (i, j, k, l)):
                        continue
                    quad = (idx_to_name[i], idx_to_name[j],
                            idx_to_name[k], idx_to_name[l])
                    if funct in improper_funct:
                        residues[current_mol]["impropers"].append(quad)
                    else:
                        residues[current_mol]["dihedrals"].append(quad)
                except ValueError:
                    continue

            elif current_block == "dihedral_restraints" and len(tokens) >= 4:
                try:
                    i, j, k, l = (int(tokens[x]) for x in range(4))
                    # funct = int(tokens[4]) if len(tokens) >= 5 else 1
                    angle = float(tokens[5]) if len(tokens) >= 6 else None
                    if not all(x in idx_to_name for x in (i, j, k, l)):
                        continue
                    quad = (idx_to_name[i], idx_to_name[j],
                            idx_to_name[k], idx_to_name[l], angle)
                    residues[current_mol]["dihedral_restraints"].append(quad)
                except ValueError:
                    continue

    return residues

def read_itp_collection(itp_sources: dict[str, str]) -> dict:
    collection = {}
    for resname, filepath in itp_sources.items():
        parsed = read_itp(filepath)
        if not parsed:
            print(f"WARNING: No molecule found in {filepath}")
            continue
        collection.update(parsed)
    return collection

def read_itp_directory(itp_dir: str,
                        resnames: None) -> dict:
    itp_dir = Path(itp_dir)
    sources = {}
    for filepath in sorted(itp_dir.glob("*.itp")):
        stem = filepath.stem.upper()
        if resnames is None or stem in [r.upper() for r in resnames]:
            sources[stem] = str(filepath)

    return read_itp_collection(sources)
