MemBack - Quick Start
=====================

Backmaps coarse-grained Martini 3 membranes to all-atom CHARMM36 structures.
This file is written to be read in a terminal (e.g. `less QUICKSTART.md`).
See README.md for full details.

1. Requirements
---------------
  - Python >= 3.9
  - PyTorch          https://pytorch.org/get-started/locally/
  - PyTorch Geometric
      https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  - MDAnalysis, NumPy, lmdb, tqdm   (installed automatically by pip)
  - GROMACS          only needed for minimisation/equilibration,
                     not for backmapping itself

  GPU is optional. The model runs fine on CPU; CUDA is used automatically
  when available.

2. Install
----------
From PyPI (recommended, ideally inside a virtual environment):

    python -m venv venv && source venv/bin/activate
    pip install memback

torch and torch-geometric are installed automatically. If you need a
specific CUDA build, install those two first following their own
instructions, then run `pip install memback`.

Check that it works:

    memback --version
    memback --help

3. Run
------

    memback membrane_cg.gro

The input must be a single frame with box dimensions (.gro, .pdb, or any
format MDAnalysis reads). Results are written to `membrane_cg_backmapped/`.

Then source your GROMACS installation and minimise + equilibrate:

    cd membrane_cg_backmapped
    bash run_sim.sh

run_sim.sh runs, in order:

  1. Steepest-descent minimisation (min.mdp)          -> min.gro
  2. Six equilibration runs (step6.1 ... step6.6)     -> step6.6_equilibration.gro
  3. A chirality check on the final frame


4. Chirality check and fix
--------------------------

Check any structure yourself:

    python check_chirality.py step6.6_equilibration.gro

If the `wrong` and `flat` columns are all zero, nothing needs to be done.
Otherwise repair with:

    python check_chirality.py step6.6_equilibration.gro --fix

This writes `step6.6_equilibration_chirfix.gro` (change with `-o FILE`).
Run a restrained energy minimisation on the fixed structure before
continuing.


5. Common options
-----------------

    -o DIR, --output DIR       output directory
                               (default: <input stem>_backmapped)
    -e DIR, --extension DIR    folder with extra .map/.bnd/.itp files
                               for lipids not built in
    -m CKPT, --model CKPT      alternative model checkpoint
    --device auto|cpu|cuda     default: auto
    -V, --version              print version and exit


6. Lipids not built in (mapping extension)
------------------------------------------

MemBack ships 55 lipid types. If the run prints
"Residue XXXX not found in mapping. Skipping...", that lipid is missing
from the output. Add it through an extension folder; files are matched by
suffix:

    my_lipids/
      mylipids.map    AA -> CG mapping, one [RESNAME] section per lipid
      mylipids.bnd    CG bonds and angles, one [RESNAME] section per lipid
      XXXX.itp        GROMACS topology, one file per lipid, named <RESNAME>.itp

Then run:

    memback membrane_cg.gro -e ./my_lipids

.map example (bead name, Martini type, charge if there are any then the atoms it maps):

    [POPC]
    NC3 Q1 1 N C12 C13 C14 C15
    PO4 Q5 -1 P O11 O12 O13 O14

.bnd example (bonds, a blank line, then angles):

    [POPC]
    NC3 PO4
    PO4 GL1

    NC3 PO4 GL1

Entries in the extension folder override built-in ones with the same
residue name, so the same folder can also patch a shipped lipid. Formats
follow PyCGTOOL conventions; see "Extending to new lipids" in README.md.


7. Quick troubleshooting
------------------------

  "Residue XXXX not found in mapping"
      That lipid is not built in; add it with -e (see section 6).

  "MemBack data files are missing"
      Incomplete install, or MEMBACK_ROOT points to the wrong place.
      Unset MEMBACK_ROOT or reinstall.

  "Could not find XXXX.itp ... for forcefield"
      No CHARMM topology for that lipid; add the .itp via -e (section 6).

  CUDA out of memory
      Use --device cpu for very large systems.


For output files, adding new lipids, the Python API and how the model works,
see README.md.
