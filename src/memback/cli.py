"""
Command-line interface for MemBack.

Installed as the `memback` console script (see [project.scripts] in
pyproject.toml). The command name comes from that entry point, not from the
name of this file.
"""

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

from memback import __version__


EPILOG = """\
examples:
  # backmap a Martini 3 membrane, results in ./membrane_cg_backmapped/
  memback membrane_cg.gro

  # choose the output directory
  memback membrane_cg.gro -o popc_run1

  # add lipids the shipped databases do not cover
  memback membrane_cg.gro -e ./my_lipids

  # backmap residues named POP1 in the input as DPPC
  memback membrane_cg.gro --rename POP1=DPPC

  # backmap the protein with CG2AT2 (temporary, until MemBack supports proteins)
  memback membrane_cg.gro --prot-back cg2at
  memback membrane_cg.gro --prot-back /path/to/cg2at --prot-back-args "-fg martini_2-2_charmm36"

  # force CPU even on a CUDA machine
  memback membrane_cg.gro --device cpu

  # create .map/.bnd files for a lipid MemBack does not cover
  memback create_map --m3itp martini_PC.itp --m3name DAPC --aaitp DAPC.itp -o my_lipids/DAPC

  # check the .map/.bnd files of an extension folder before using them
  memback check_maps -e my_lipids

commands:
  create_map   Build a .map/.bnd pair from a lipid's Martini 3 and CHARMM36
               .itp files, for use with -e. See: memback create_map -h
  check_maps   Check .map/.bnd files against the lipid topologies (the shipped
               databases, or an extension folder with -e). See: memback check_maps -h

extension directory (-e):
  A folder holding any combination of the following, for lipids that are not
  in the built-in databases. Files are matched by suffix, not by name:

    *.map   AA -> CG mapping, one [RESNAME] section per lipid
    *.bnd   CG bead bonds and angles, one [RESNAME] section per lipid
    *.itp   GROMACS topology, one file per lipid, named <RESNAME>.itp

  Entries here override the built-in databases for the same residue name, so
  the same folder can also be used to patch a lipid that ships with MemBack.

outputs (written into the output directory):
  backmapped_ordered.gro     + water and ions, residues grouped by type
  topol.top                  GROMACS topology referencing toppar/
  min.mdp                    restrained steepest-descent minimisation
  step6.*_equilibration.mdp  six restrained equilibration stages
  index.ndx                  MEMB and SOLV groups for equilibration
  check_chirality.py         non-interactive chirality check (--fix to repair)
  run_sim.sh                 minimisation, equilibration and chirality check
  toppar/                    force field and per-lipid .itp files

The output is a raw model prediction. Always run the packaged run_sim.sh
(minimisation and equilibration) before using the structure for production MD.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memback",
        description=(
            "Backmap a coarse-grained Martini 3 membrane to an all-atom "
            "CHARMM36 structure using an SE(3)-equivariant graph network."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Coarse-grained input structure. Any single-frame format MDAnalysis "
            "can read (.gro, .pdb, ...). Must carry box dimensions."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Output directory. Created if missing. "
            "Default: <input stem>_backmapped in the current directory."
        ),
    )
    parser.add_argument(
        "-e", "--extension",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory of extra .map / .bnd / .itp files for lipids outside the "
            "built-in databases. See the notes below."
        ),
    )
    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=None,
        metavar="CKPT",
        help="Model checkpoint (.pt). Default: the version shipped in model/.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Leave out residues that have no mapping instead of stopping. "
            "By default MemBack stops and suggests memback create_map commands. "
            "Residues whose beads do not match their map are never skipped."
        ),
    )
    parser.add_argument(
        "--rename",
        type=parse_rename,
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Read residues named OLD in the input as NEW, e.g. --rename POP1=DPPC "
            "for a DPPC that was renamed in the simulation. NEW is a Martini 3 "
            "name and is also used in the output. Repeat for more lipids."
        ),
    )
    # prot-back: temporary CG2AT2 protein backmapping, see protein_backmap.py
    parser.add_argument(
        "--prot-back", "--prot_back",
        dest="prot_back",
        default=None,
        metavar="CG2AT",
        help=(
            "Temporary, until MemBack supports proteins: backmap the protein with "
            "this CG2AT2 executable (path or command name) and merge it into the "
            "output. MemBack still backmaps everything else."
        ),
    )
    parser.add_argument(
        "--prot-back-args", "--prot_back_args",
        dest="prot_back_args",
        default=None,
        metavar="ARGS",
        help=(
            "Extra CG2AT2 arguments as one quoted string, e.g. "
            "\"-fg martini_2-2_charmm36\". They override the defaults "
            "(-ff charmm36-cg2at -fg martini_3-0_charmm36 -w tip3p -o none -silent)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Torch device. 'auto' uses CUDA when available and continues on the "
            "CPU if the system does not fit in the free GPU memory. Default: auto."
        ),
    )
    parser.add_argument(
        "-V", "-v", "--version",
        action="version",
        version=f"memback {__version__}",
    )
    return parser


def parse_rename(value: str):
    """'OLD=NEW' -> (OLD, NEW), for --rename."""
    old, sep, new = value.partition("=")
    old, new = old.strip(), new.strip()
    if not sep or not old or not new or "=" in new:
        raise argparse.ArgumentTypeError(f"expected OLD=NEW, got {value!r}")
    return old, new


def resolve_device(choice: str):
    import torch

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "error: --device cuda requested but no CUDA device is visible to torch"
            )
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device) -> str:
    """Device line for the run header, with GPU name and free memory on CUDA."""
    if device.type != "cuda":
        return str(device)
    import torch

    try:
        free, total = torch.cuda.mem_get_info(device)
        name = torch.cuda.get_device_name(device)
        return f"{device} ({name}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free)"
    except RuntimeError:
        return str(device)


def missing_mapping_message(resnames, input_path, extension=None) -> str:
    """Explain how to create the missing maps with memback create_map."""
    from memback import config
    from memback.create_map import read_m3_molecules

    folder = extension or Path("my_lipids")
    m3_files = {}
    for itp in sorted(Path(config.itp_m3_db_path).glob("*.itp")):
        for name in read_m3_molecules(itp):
            m3_files.setdefault(name, itp)

    lines = [
        f"error: no mapping for residue(s) in {input_path}: {', '.join(resnames)}",
        "",
        "Create their .map/.bnd files with memback create_map, check them, and pass",
        "the folder with -e:",
        "",
    ]
    copies = []
    for resname in resnames:
        m3_name = config.charmm_to_martini3.get(resname, resname)
        m3_itp = m3_files.get(m3_name, f"<Martini 3 itp of {m3_name}>")
        aa_itp = Path(config.itp_db_path) / f"{resname}.itp"
        if extension and (extension / f"{resname}.itp").is_file():
            aa_itp = extension / f"{resname}.itp"
        elif not aa_itp.is_file():
            aa_itp = f"<CHARMM36 {resname}.itp>"
            copies.append(f"  cp {aa_itp} {folder / f'{resname}.itp'}")
        lines.append(f"  memback create_map --m3itp {m3_itp} --m3name {m3_name} \\")
        lines.append(f"      --aaitp {aa_itp} -o {folder / resname}")
    lines += [
        *copies,
        f"  memback check_maps -e {folder}",
        f"  memback {input_path} -e {folder}",
        "",
        "Or rerun with --skip-missing to leave these residues out of the output.",
    ]
    from MDAnalysis.core.selection import ProteinSelection  # prot-back
    if set(resnames) & set(ProteinSelection.prot_res):  # prot-back
        lines += ["Protein residues can be backmapped with CG2AT2 (temporary):",  # prot-back
                  f"  memback {input_path} --prot-back <path to cg2at>"]  # prot-back
    return "\n".join(lines)


def bead_mismatch_message(mismatches, input_path) -> str:
    """Explain which residues have beads that do not match their .map entry."""
    lines = [f"error: bead names in {input_path} do not match the MemBack map:"]
    for m in mismatches:
        name = m["resname"]
        if m["mapped_as"] != name:
            name += f" (read as {m['mapped_as']})"
        lines += [
            "",
            f"  [{name}] {m['n_residues']} residue(s), e.g. resid {m['example_resid']}",
            f"    input beads : {' '.join(m['input_beads'])}",
            f"    map beads   : {' '.join(m['map_beads'])}",
        ]
        if m["missing"]:
            lines.append(f"    missing     : {' '.join(m['missing'])}")
        if m["duplicated"]:
            lines.append(f"    duplicated  : {' '.join(m['duplicated'])}")
        if m["order_wrong"]:
            lines.append("    order       : same beads as the map, but in a different order")
        if m["candidates"]:
            lines.append(f"    the input beads match the map(s) of: {', '.join(m['candidates'])}")
    lines += [
        "",
        "Residue names select the map, so a renamed lipid is backmapped as another one.",
        "Read them as the right lipid with --rename OLD=NEW (e.g. --rename POP1=DPPC),",
        "put the beads in the map order, or provide a matching map for this residue",
        "name with -e (check it with memback check_maps -e DIR). These residues are",
        "not skipped by --skip-missing.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # Subcommands are dispatched by hand so that `memback <input>` keeps working
    # without a "backmap" subcommand in front of it.
    if argv and argv[0] == "create_map":
        from memback.create_map import main as create_map_main
        return create_map_main(argv[1:])
    if argv and argv[0] == "check_maps":
        from memback.check_maps import main as check_maps_main
        return check_maps_main(argv[1:])

    args = build_parser().parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"error: input structure not found: {args.input}")
    if args.extension is not None and not args.extension.is_dir():
        raise SystemExit(f"error: extension path is not a directory: {args.extension}")
    if args.model is not None and not args.model.exists():
        raise SystemExit(f"error: model checkpoint not found: {args.model}")
    prot_back = None  # prot-back
    if args.prot_back_args is not None and args.prot_back is None:  # prot-back
        raise SystemExit("error: --prot-back-args needs --prot-back")  # prot-back
    if args.prot_back is not None:  # prot-back
        prot_back = shutil.which(args.prot_back)  # prot-back
        if prot_back is None:  # prot-back
            raise SystemExit(f"error: CG2AT2 executable not found: {args.prot_back}")  # prot-back
        prot_back = os.path.abspath(prot_back)  # prot-back

    # Imported late so that --help and --version stay fast and do not require
    # torch to be importable.
    from memback.config import check_data_files, model_path as default_model_path
    from memback.pipeline import BeadMismatchError, MissingMappingError, backmapping
    from memback.protein_backmap import ProteinBackmapError  # prot-back

    missing = check_data_files()
    if missing:
        hint = (
            "MEMBACK_ROOT is set, so paths are being read from there. Unset it to "
            "use the databases bundled with the installation."
            if os.environ.get("MEMBACK_ROOT")
            else "The installation looks incomplete; try reinstalling MemBack."
        )
        raise SystemExit(
            "error: MemBack data files are missing:\n  "
            + "\n  ".join(missing)
            + f"\n\n{hint}"
        )

    output = args.output or Path.cwd() / f"{args.input.stem}_backmapped"
    model = args.model or Path(default_model_path)
    device = resolve_device(args.device)

    print(f"MemBack {__version__}")
    print(f"  input      {args.input}")
    print(f"  output     {output}")
    print(f"  model      {model}")
    print(f"  device     {describe_device(device)}")
    if args.rename:
        print(f"  rename     {', '.join(f'{old}={new}' for old, new in args.rename)}")
    if prot_back:  # prot-back
        print(f"  protein    CG2AT2 {prot_back} {args.prot_back_args or ''}".rstrip())
    if args.extension:
        print(f"  extension  {args.extension}")
    print()

    try:
        backmapping(
            input_path=str(args.input),
            model_path=str(model),
            filename=str(output),
            ext_path=str(args.extension) if args.extension else None,
            # "auto" lets the pipeline fall back to the CPU if the GPU is full.
            device="auto" if args.device == "auto" else device,
            skip_missing=args.skip_missing,
            rename=dict(args.rename),
            prot_back=prot_back,  # prot-back
            prot_back_args=shlex.split(args.prot_back_args or ""),  # prot-back
        )
    except MissingMappingError as err:
        raise SystemExit(missing_mapping_message(err.resnames, args.input, args.extension))
    except BeadMismatchError as err:
        raise SystemExit(bead_mismatch_message(err.mismatches, args.input))
    except ProteinBackmapError as err:  # prot-back
        raise SystemExit(f"error: {err}")  # prot-back

    print()
    print(f"Done. Source your preferred GROMACS version. Later, minimise and equilibrate before production MD:")
    print(f"  cd {output} && bash run_sim.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())