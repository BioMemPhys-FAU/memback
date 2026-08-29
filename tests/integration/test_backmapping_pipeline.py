from pathlib import Path

import pytest

from memback.config import check_data_files, model_path

pytestmark = pytest.mark.integration

DATA_DIR = Path(__file__).parent / "data" / "backmapping"
CG_STRUCTURES = sorted(DATA_DIR.glob("*.gro")) if DATA_DIR.is_dir() else []


def _require_model():
    missing = check_data_files()
    if missing:
        pytest.skip(f"MemBack data files missing, cannot run the pipeline: {missing}")


@pytest.mark.skipif(not CG_STRUCTURES, reason=(
    f"No CG .gro fixtures found in {DATA_DIR}. Add one or more Martini 3 "
    "structures there (see src/memback/test_sims/pred_prod_data for examples)."
))
@pytest.mark.parametrize("cg_structure", CG_STRUCTURES, ids=lambda p: p.stem)
def test_backmapping_produces_all_atom_structure(cg_structure, tmp_path):
    _require_model()
    import MDAnalysis as mda

    from memback.pipeline import backmapping

    output_dir = tmp_path / cg_structure.stem
    backmapping(str(cg_structure), model_path, str(output_dir))

    out_gro = output_dir / "backmapped_ordered.gro"
    assert out_gro.exists()

    cg_uni = mda.Universe(str(cg_structure), to_guess=[])
    aa_uni = mda.Universe(str(out_gro))

    # The all-atom structure should have strictly more atoms than the CG one
    # (each bead expands into several atoms) and the box should be preserved.
    assert len(aa_uni.atoms) > len(cg_uni.atoms)
    import numpy as np
    np.testing.assert_allclose(aa_uni.dimensions[:3], cg_uni.dimensions[:3], rtol=1e-3)

    # A GROMACS run script and topology should be ready for minimisation.
    assert (output_dir / "topol.top").exists()
    assert (output_dir / "run_min.sh").exists()
