from __future__ import annotations

import re
from pathlib import Path
import os


def _read_itp_atoms_bonds(itp_path: str):
    section = re.compile(r"^\s*\[\s*(\S+)\s*\]")
    strip_comment = re.compile(r";.*$")

    molname = None
    atoms = []
    bonds = []
    block = None

    with open(itp_path) as fh:
        for raw in fh:
            line = strip_comment.sub("", raw).strip()
            if not line:
                continue

            m = section.match(line)
            if m:
                block = m.group(1).lower()
                continue

            tok = line.split()

            if block == "moleculetype":
                molname = tok[0]

            elif block == "atoms":
                if len(tok) < 5:
                    continue
                try:
                    idx = int(tok[0])
                except ValueError:
                    continue
                atype = tok[1]
                name = tok[4]
                mass = None
                if len(tok) >= 8:
                    try:
                        mass = float(tok[7])
                    except ValueError:
                        mass = None
                atoms.append({"idx": idx, "name": name, "mass": mass, "type": atype})

            elif block == "bonds":
                if len(tok) < 2:
                    continue
                try:
                    bonds.append((int(tok[0]), int(tok[1])))
                except ValueError:
                    continue

    return molname, atoms, bonds

def _is_hydrogen(atom: dict) -> bool:
    mass = atom["mass"]
    if mass is not None:
        return mass < 3.0
    return atom["type"].upper().startswith("H") or atom["name"].upper().startswith("H")


def _name_sort_key(name: str):
    return name


def _build_graph(atoms, bonds):
    idx2atom = {a["idx"]: a for a in atoms}
    neighbours = {a["idx"]: [] for a in atoms}
    for i, j in bonds:
        if i in neighbours and j in neighbours:
            if j not in neighbours[i]:
                neighbours[i].append(j)
            if i not in neighbours[j]:
                neighbours[j].append(i)
    return idx2atom, neighbours

class HdbBuildError(Exception):
    pass


def build_hdb_entries(itp_path: str, verbose: bool = False):
    molname, atoms, bonds = _read_itp_atoms_bonds(itp_path)
    if molname is None:
        raise HdbBuildError(f"No [ moleculetype ] found in {itp_path}")

    idx2atom, neigh = _build_graph(atoms, bonds)
    name = lambda k: idx2atom[k]["name"]
    is_h = lambda k: _is_hydrogen(idx2atom[k])

    entries = []
    warnings = []

    for a in atoms:
        A = a["idx"]
        if _is_hydrogen(a):
            continue

        h_nb = [k for k in neigh[A] if is_h(k)]          # attached hydrogens
        heavy_nb = [k for k in neigh[A] if not is_h(k)]   # heavy neighbours
        nH = len(h_nb)
        nheavy = len(heavy_nb)

        if nH == 0:
            continue

        if nH == 3:
            if nheavy != 1:
                warnings.append(
                    f"{name(A)}: 3 H but {nheavy} heavy neighbours "
                    f"(expected 1 for a methyl) -- emitting type 4 anyway"
                )
            B = heavy_nb[0] if heavy_nb else None
            if B is None:
                raise HdbBuildError(f"{name(A)} has 3 H but no heavy neighbour")
            others = [k for k in neigh[B] if k != A]
            heavy_refs = sorted([k for k in others if not is_h(k)], key=lambda k: _name_sort_key(name(k)))
            h_refs = sorted([k for k in others if is_h(k)], key=lambda k: _name_sort_key(name(k)))
            refs = heavy_refs + h_refs
            if len(refs) < 3:
                warnings.append(
                    f"{name(A)}: only {len(refs)} reference atom(s) on "
                    f"{name(B)} for a methyl; reusing to fill 3."
                )
                while len(refs) < 3 and refs:
                    refs.append(refs[-1])
            for h, r in zip(h_nb, refs):
                entries.append((1, 4, [name(h)], [name(A), name(B), name(r)]))

        elif nH == 2:
            if nheavy == 2:
                j, k = heavy_nb[0], heavy_nb[1]
                entries.append((1, 6, [name(h_nb[0])], [name(A), name(j), name(k)]))
                entries.append((1, 6, [name(h_nb[1])], [name(A), name(k), name(j)]))
            elif nheavy == 1:
                B = heavy_nb[0]
                ref = _pick_reference(A, B, neigh, is_h, name)
                entries.append((2, 3, [name(h_nb[0]), name(h_nb[1])],
                                [name(A), name(B), ref]))
                warnings.append(
                    f"{name(A)}: planar 2-H group (type 3). GROMACS will name the "
                    f"two H as base+1 / base+2 -- check that rtp/itp names "
                    f"({name(h_nb[0])}, {name(h_nb[1])}) follow that pattern."
                )
            else:
                warnings.append(f"{name(A)}: 2 H with {nheavy} heavy neighbours -- skipped")

        elif nH == 1:
            h = h_nb[0]
            if nheavy == 3:
                j, k, l = heavy_nb[0], heavy_nb[1], heavy_nb[2]
                entries.append((1, 5, [name(h)], [name(A), name(j), name(k), name(l)]))
            elif nheavy == 2:
                j, k = heavy_nb[0], heavy_nb[1]
                entries.append((1, 1, [name(h)], [name(A), name(j), name(k)]))
            elif nheavy == 1:
                B = heavy_nb[0]
                ref = _pick_reference(A, B, neigh, is_h, name)
                entries.append((1, 2, [name(h)], [name(A), name(B), ref]))
            else:
                warnings.append(f"{name(A)}: 1 H with {nheavy} heavy neighbours -- skipped")

        else:
            warnings.append(f"{name(A)}: unusual {nH} H / {nheavy} heavy -- skipped")

    if verbose and warnings:
        print(f"[{molname}] notes:")
        for w in warnings:
            print("   -", w)

    return molname, entries


def _pick_reference(A, B, neigh, is_h, name):
    others = [k for k in neigh[B] if k != A]
    heavy = sorted([k for k in others if not is_h(k)], key=name)
    if heavy:
        return name(heavy[0])
    hs = sorted([k for k in others if is_h(k)], key=name)
    if hs:
        return name(hs[0])
    raise HdbBuildError(
        f"Cannot find a reference atom for H on {name(A)} (neighbour {name(B)} "
        f"has no other atoms)."
    )

def format_hdb(resname: str, entries, resname_override: str | None = None) -> str:
    rn = resname_override or resname
    lines = [f"{rn:<10s}{len(entries):d}"]
    for n_add, add_type, h_names, ctrl in entries:
        h_field = h_names[0]
        cols = [f"{n_add:d}", f"{add_type:d}", f"{h_field:<6s}"]
        cols += [f"{c:<6s}" for c in ctrl]
        lines.append("    " + "  ".join(cols).rstrip())
    return "\n".join(lines) + "\n"


def itp_to_hdb(itp_path: str,
               out_path: str | None = None,
               resname_override: str | None = None,
               append: bool = False,
               verbose: bool = True) -> str:
    molname, entries = build_hdb_entries(itp_path, verbose=verbose)
    block = format_hdb(molname, entries, resname_override=resname_override)
    if out_path:
        mode = "a" if append else "w"
        with open(out_path, mode) as fh:
            if append and Path(out_path).stat().st_size > 0:
                fh.write("\n")
            fh.write(block)
        if verbose:
            n_h = sum(e[0] for e in entries)
            verb = "Appended" if append else "Wrote"
            print(f"{verb} {len(entries)} entries ({n_h} H) for "
                  f"{resname_override or molname} -> {out_path}")
    return block


def itp_dir_to_hdb(itp_dir: str,
                   out_path: str,
                   target_itps: list[str] | None = None,
                   verbose: bool = False) -> str:
    itp_dir = Path(itp_dir)
    blocks = []
    for fp in sorted(itp_dir.glob("*.itp")):
        stem = fp.stem.upper()
        basename = os.path.basename(fp)
        if target_itps is not None and basename not in target_itps:
            continue
        blocks.append(itp_to_hdb(str(fp), resname_override=stem, verbose=verbose).rstrip("\n"))
    text = "\n".join(blocks) + "\n"
    Path(out_path).write_text(text)
    if verbose:
        print(f"\nCombined {len(blocks)} residue(s) -> {out_path}")
    return text
