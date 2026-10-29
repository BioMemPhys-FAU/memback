"""
Temporary: protein backmapping via CG2AT2 until MemBack's own protein model is ready.

MemBack backmaps the lipids, water and ions; the protein beads are handed to
CG2AT2 (https://github.com/pstansfeld/cg2at) and its all-atom protein is merged
into MemBack's output (coordinates, topol.top, toppar/ and index.ndx).

To drop this feature, delete this file and the lines marked `# prot-back` in
pipeline.py and cli.py. To replace CG2AT2 with another tool, provide a function
with the same return value as run_cg2at(): (all-atom protein Universe, list of
.itp files to copy into toppar/, [(moleculetype, count)] in coordinate order).
"""

import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import MDAnalysis as mda

PROTEIN_SELECTION = "protein"

# Non-interactive CG2AT2 run for a Martini 3 protein in CHARMM36. Arguments the
# user passes with --prot-back-args come after these and override them.
DEFAULT_CG2AT_ARGS = ["-ff", "charmm36-cg2at", "-fg", "martini_3-0_charmm36",
                      "-w", "tip3p", "-o", "none", "-silent"]


class ProteinBackmapError(RuntimeError):
    """The external protein backmapping failed."""


def protein_resnames(universe):
    """Residue names of the protein residues in the universe."""
    return set(universe.select_atoms(PROTEIN_SELECTION).residues.resnames)


def log_tail(log_path, n_lines=15):
    # CG2AT2 draws progress bars with carriage returns; keep only the text lines.
    text = Path(log_path).read_text(errors="replace").replace("\r", "\n")
    lines = [line for line in text.splitlines() if line.strip() and "[" not in line]
    return "\n".join(lines[-n_lines:])


def moleculetype_name(itp_path):
    lines = Path(itp_path).read_text().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*\[\s*moleculetype\s*\]", line):
            for entry in lines[i + 1:]:
                entry = entry.split(";")[0].strip()
                if entry:
                    return entry.split()[0]
    raise ProteinBackmapError(f"no [ moleculetype ] in {itp_path}")


def read_protein_molecules(top_path):
    """[(include name, count)] of the PROTEIN_* entries in a CG2AT2 topol_final.top."""
    molecules, in_molecules = [], False
    for line in Path(top_path).read_text().splitlines():
        line = line.split(";")[0].strip()
        if line.startswith("["):
            in_molecules = line.replace(" ", "") == "[molecules]"
        elif in_molecules and line and not line.startswith("#"):
            name, count = line.split()[:2]
            if name.startswith("PROTEIN_"):
                molecules.append((name, int(count)))
    return molecules


def run_cg2at(cg2at_bin, protein_atoms, workdir, extra_args=()):
    """
    Backmap the CG protein atoms with CG2AT2 inside workdir.

    Returns (all-atom protein Universe, .itp files for toppar/,
    [(moleculetype, count)] in the order of the coordinates).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    protein_atoms.write(workdir / "protein_cg.gro")
    log_path = workdir / "cg2at.log"
    command = [str(cg2at_bin), "-c", "protein_cg.gro", "-loc", "cg2at",
               *DEFAULT_CG2AT_ARGS, *extra_args]
    # CG2AT2 is a Python script; let it find the interpreter MemBack runs in.
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    print(f"Backmapping {protein_atoms.n_residues} protein residues with CG2AT2 "
          f"(log: {log_path})")
    with open(log_path, "w") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        # No stdin: a CG2AT2 prompt fails instead of waiting forever.
        result = subprocess.run(command, cwd=workdir, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env)

    final = workdir / "cg2at" / "FINAL"
    pdb, top = final / "final_cg2at_de_novo.pdb", final / "topol_final.top"
    if result.returncode != 0 or not pdb.is_file() or not top.is_file():
        raise ProteinBackmapError(
            f"CG2AT2 failed (exit status {result.returncode}). Last lines of {log_path}:\n\n"
            f"{log_tail(log_path)}"
        )

    molecules = []
    for name, count in read_protein_molecules(top):
        molecules.append((moleculetype_name(final / f"{name}.itp"), count))
    if not molecules:
        raise ProteinBackmapError(f"no PROTEIN_* molecules in {top}")
    itp_paths = sorted(final.glob("PROTEIN_*.itp"))

    protein_uni = mda.Universe(str(pdb), to_guess=[])
    protein_uni.atoms.translate(-cg2at_shift(workdir))
    return protein_uni, itp_paths, molecules


def cg2at_shift(workdir):
    """
    Translation CG2AT2 applied to the protein. It centres the CG input in the
    box (gmx editconf -c) and builds the atomistic protein there, so its output
    has to be moved back onto the CG beads.
    """
    cg = mda.Universe(str(workdir / "protein_cg.gro"), to_guess=[])
    centred_pdb = workdir / "cg2at" / "INPUT" / "CG_INPUT.pdb"
    if centred_pdb.is_file():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no elements in CG2AT2's PDB
            centred = mda.Universe(str(centred_pdb), to_guess=[])
        if centred.atoms.n_atoms == cg.atoms.n_atoms:
            return (centred.atoms.positions - cg.atoms.positions).mean(axis=0)
    # Same shift as gmx editconf -c: box centre minus the mean position.
    return cg.dimensions[:3] / 2 - cg.atoms.positions.mean(axis=0)


def add_protein(output_dir, system_uni, protein_uni, itp_paths, molecules,
                pred_name="backmapped_ordered.gro"):
    """
    Put the all-atom protein in front of MemBack's backmapped system and update
    the files sim_preparer wrote: coordinates, toppar/, topol.top and index.ndx.
    """
    output_dir = Path(output_dir)
    # Same topology attributes on both sides so that Merge keeps them all.
    protein = mda.Universe.empty(protein_uni.atoms.n_atoms, protein_uni.residues.n_residues,
                                 atom_resindex=protein_uni.atoms.resindices, trajectory=True)
    protein.add_TopologyAttr("names", protein_uni.atoms.names)
    protein.add_TopologyAttr("resnames", protein_uni.residues.resnames)
    protein.add_TopologyAttr("resids", protein_uni.residues.resids)
    protein.add_TopologyAttr("segids", ["MEMB"])
    protein.atoms.positions = protein_uni.atoms.positions
    system = mda.Merge(system_uni.atoms)
    for attr in ("names", "resnames", "resids"):
        if not hasattr(system.atoms, attr):
            raise ProteinBackmapError(f"backmapped system has no {attr}")
    if not hasattr(system.atoms, "segids"):
        system.add_TopologyAttr("segids")

    merged = mda.Merge(protein.atoms, system.atoms)
    merged.dimensions = system_uni.dimensions
    merged.atoms.write(output_dir / pred_name)

    toppar = output_dir / "toppar"
    for itp in itp_paths:
        shutil.copy2(itp, toppar / itp.name)
    includes = [f'#include "toppar/{itp.name}"\n' for itp in itp_paths
                if "posre" not in itp.name]

    top_path = output_dir / "topol.top"
    lines = top_path.read_text().splitlines(keepends=True)
    ff_line = next(i for i, line in enumerate(lines) if "forcefield.itp" in line)
    lines[ff_line + 1:ff_line + 1] = includes
    mol_line = next(i for i, line in enumerate(lines) if line.strip() == "[ molecules ]")
    lines[mol_line + 1:mol_line + 1] = [f"{name}  \t          {count}\n" for name, count in molecules]
    top_path.write_text("".join(lines))

    with mda.selections.gromacs.SelectionWriter(str(output_dir / "index.ndx"), mode="w") as ndx:
        ndx.write(merged.select_atoms("segid MEMB"), name="MEMB")
        ndx.write(merged.select_atoms("segid SOLV ION"), name="SOLV")
    return merged
