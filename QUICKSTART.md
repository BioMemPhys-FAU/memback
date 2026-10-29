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

    python3 -m venv venv && source venv/bin/activate
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

    python3 check_chirality.py step6.6_equilibration.gro

If the `wrong` and `flat` columns are all zero, nothing needs to be done.
Otherwise repair with:

    python3 check_chirality.py step6.6_equilibration.gro --fix

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
    --rename OLD=NEW           read residues named OLD as the Martini 3
                               lipid NEW, e.g. --rename POP1=DPPC
                               (repeat for more lipids)
    --prot-back CG2AT          temporary: backmap the protein with CG2AT2
                               (installed separately); MemBack does the rest
    --prot-back-args "ARGS"    extra CG2AT2 arguments, e.g.
                               "-fg martini_2-2_charmm36"
    --device auto|cpu|cuda     default: auto
    -V, --version              print version and exit


6. Lipids not built in (mapping extension)
------------------------------------------

If the membrane contains a lipid MemBack does not ship, the run stops
with "error: no mapping for residue(s) ..." and prints the memback
create_map commands that build its mapping, for example:

    memback create_map --m3itp martini_v3.0.0_phospholipids_PC_v2.itp \
        --m3name DPPC --aaitp DPPC.itp -o my_lipids/DPPC
    memback check_maps -e my_lipids

(Add --skip-missing to the memback run to leave such lipids out instead.)
Add the lipid through an extension folder; files are matched by suffix:

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

  "error: no mapping for residue(s) ..."
      That lipid is not built in; run the printed memback create_map
      commands and add it with -e (see section 6), or use --skip-missing.

  "error: bead names in ... do not match the MemBack map"
      MemBack picks a lipid's map by its residue name, so a renamed lipid
      (e.g. a DPPC saved as POP1, which is the Martini 3 name of a PI
      lipid) is caught by the bead check. Rerun with --rename OLD=NEW,
      using one of the lipids the error lists as matching:
          memback membrane_cg.gro --rename POP1=DPPC

  "MemBack data files are missing"
      Incomplete install, or MEMBACK_ROOT points to the wrong place.
      Unset MEMBACK_ROOT or reinstall.

  "Could not find XXXX.itp ... for forcefield"
      No CHARMM topology for that lipid; add the .itp via -e (section 6).

  CUDA out of memory
      When the GPU runs out of memory: `--device auto` then continues on the CPU, 
      `--device cuda` stops. Use `--device cpu` to avoid the retries.


For output files, adding new lipids, the Python API and how the model works,
see README.md.
