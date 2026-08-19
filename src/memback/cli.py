"""
Command-line interface for MemBack.

Installed as the `memback` console script (see [project.scripts] in
pyproject.toml). The command name comes from that entry point, not from the
name of this file.
"""

import argparse
import os
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

  # force CPU even on a CUDA machine
  memback membrane_cg.gro --device cpu

extension directory (-e):
  A folder holding any combination of the following, for lipids that are not
  in the built-in databases. Files are matched by suffix, not by name:

    *.map   AA -> CG mapping, one [RESNAME] section per lipid
    *.bnd   CG bead bonds and angles, one [RESNAME] section per lipid
    *.itp   GROMACS topology, one file per lipid, named <RESNAME>.itp

  Entries here override the built-in databases for the same residue name, so
  the same folder can also be used to patch a lipid that ships with MemBack.

outputs (written into the output directory):
  backmapped_ordered.gro   + water and ions, residues grouped by type
  topol.top                GROMACS topology referencing toppar/
  min.mdp                  restrained steepest-descent minimisation
  run_min.sh               grompp + mdrun for that minimisation
  toppar/                  force field and per-lipid .itp files

The output is a raw model prediction. Always run the packaged minimisation
before using the structure for production MD.
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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device. 'auto' uses CUDA when available. Default: auto.",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"memback {__version__}",
    )
    return parser


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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"error: input structure not found: {args.input}")
    if args.extension is not None and not args.extension.is_dir():
        raise SystemExit(f"error: extension path is not a directory: {args.extension}")
    if args.model is not None and not args.model.exists():
        raise SystemExit(f"error: model checkpoint not found: {args.model}")

    # Imported late so that --help and --version stay fast and do not require
    # torch to be importable.
    from memback.config import check_data_files, model_path as default_model_path
    from memback.pipeline import backmapping

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
    print(f"  device     {device}")
    if args.extension:
        print(f"  extension  {args.extension}")
    print()

    backmapping(
        input_path=str(args.input),
        model_path=str(model),
        filename=str(output),
        ext_path=str(args.extension) if args.extension else None,
        device=device,
    )

    print()
    print(f"Done. Minimise before production MD:")
    print(f"  cd {output} && bash run_min.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
