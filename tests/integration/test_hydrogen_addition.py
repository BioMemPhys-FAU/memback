from pathlib import Path

import numpy as np
import pytest

from memback.config import hdb_path, itp_db_path

pytestmark = pytest.mark.integration

DATA_DIR = Path(__file__).parent / "data" / "hydrogen_addition"
CASES = sorted(p for p in DATA_DIR.glob("*") if (p / "heavy.gro").exists()) if DATA_DIR.is_dir() else []


@pytest.mark.skipif(not CASES, reason=(
    f"No fixtures found in {DATA_DIR}. Add <case>/heavy.gro (+ optional "
    "reference.gro) subdirectories to exercise place_hydrogens on real "
    "predicted structures."
))
@pytest.mark.parametrize("case_dir", CASES, ids=lambda p: p.name)
def test_place_hydrogens_adds_expected_hydrogen_count(case_dir):
    import MDAnalysis as mda

    from memback.io.read_sim_metadata import read_hdb, read_itp_directory
    from memback.structure_repair.hydrogen_adder_gmx import place_hydrogens

    heavy_uni = mda.Universe(str(case_dir / "heavy.gro"), to_guess=[])
    hdb = read_hdb(hdb_path)
    resnames = np.unique(heavy_uni.residues.resnames)
    itp = read_itp_directory(itp_db_path, resnames)

    result = place_hydrogens(heavy_uni, hdb, itp)

    assert not np.isnan(result.atoms.positions).any()

    for resname in resnames:
        expected_atoms = len(itp[resname]["atoms"])
        got_atoms = len(result.select_atoms(f"resname {resname}").residues[0].atoms)
        assert got_atoms == expected_atoms
