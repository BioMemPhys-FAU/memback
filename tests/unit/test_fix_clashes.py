import numpy as np
import pytest

from memback.structure_repair.fix_clashes import detect_clashes, fix_clashes

BIG_BOX = [1000.0, 1000.0, 1000.0, 90.0, 90.0, 90.0]


def test_detect_clashes_ignores_intra_residue_contacts_by_default(make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["LIP"],
        atom_resindex=[0, 0],  # both atoms belong to the same residue
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )

    pairs, dists = detect_clashes(u, cutoff=0.5, include_intra=False)

    assert len(pairs) == 0


def test_detect_clashes_reports_intra_residue_contacts_when_requested(make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["LIP"],
        atom_resindex=[0, 0],
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )

    pairs, dists = detect_clashes(u, cutoff=0.5, include_intra=True)

    assert len(pairs) == 1
    assert dists[0] == pytest.approx(0.1, abs=1e-5)


def test_detect_clashes_reports_inter_residue_contacts(make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["LIP", "LIP"],
        atom_resindex=[0, 1],  # different residues
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )

    pairs, dists = detect_clashes(u, cutoff=0.5, include_intra=False)

    assert len(pairs) == 1


def test_detect_clashes_excludes_configured_resnames(make_universe):
    u = make_universe(
        atom_names=["OW1", "OW2"],
        resnames=["TIP3", "TIP3"],
        atom_resindex=[0, 1],
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )

    pairs, dists = detect_clashes(u, cutoff=0.5, include_intra=False)

    assert len(pairs) == 0


def test_fix_clashes_pushes_clashing_atoms_apart(make_universe):
    cutoff = 0.5
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["LIP", "LIP"],
        atom_resindex=[0, 1],
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )

    fix_clashes(u, cut_off=cutoff, max_iter=10)

    final_dist = np.linalg.norm(u.atoms.positions[0] - u.atoms.positions[1])
    assert final_dist >= cutoff
    assert final_dist == pytest.approx(cutoff + 0.01, abs=1e-4)


def test_fix_clashes_leaves_non_clashing_structure_untouched(make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["LIP", "LIP"],
        atom_resindex=[0, 1],
        positions=[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        box=BIG_BOX,
    )
    original = u.atoms.positions.copy()

    fix_clashes(u, cut_off=0.5, max_iter=10)

    np.testing.assert_allclose(u.atoms.positions, original)
