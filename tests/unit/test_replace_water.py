import math
import random

import numpy as np
import pytest

from memback.structure_repair.replace_water import (
    _random_rotation,
    _tet_offsets,
    _tet_unit,
    _tip3p_local_positions,
    build_tip3p_universe,
    h_o_h_ang,
    o_h_bond,
    replace_martini_water,
)

BOX = [1000.0, 1000.0, 1000.0, 90.0, 90.0, 90.0]


def _angle(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.acos(np.clip(cos_a, -1.0, 1.0))


def test_tip3p_local_positions_matches_charmm_geometry():
    o, h1, h2 = _tip3p_local_positions()

    np.testing.assert_allclose(o, [0.0, 0.0, 0.0])
    assert np.linalg.norm(h1 - o) == pytest.approx(o_h_bond)
    assert np.linalg.norm(h2 - o) == pytest.approx(o_h_bond)
    assert _angle(h1 - o, h2 - o) == pytest.approx(h_o_h_ang, abs=1e-6)


def test_tet_offsets_uses_tetrahedral_vertices_for_n_le_4():
    r = 1.0
    offsets = _tet_offsets(bead_radius=2.0, radius_scale=0.5, n=3)

    assert offsets.shape == (3, 3)
    np.testing.assert_allclose(offsets, _tet_unit[:3] * r)
    for offset in offsets:
        assert np.linalg.norm(offset) == pytest.approx(r)


def test_tet_offsets_uses_fibonacci_sphere_for_n_gt_4():
    r = 1.5
    offsets = _tet_offsets(bead_radius=3.0, radius_scale=0.5, n=6)

    assert offsets.shape == (6, 3)
    for offset in offsets:
        assert np.linalg.norm(offset) == pytest.approx(r)


def test_random_rotation_is_a_proper_orthogonal_matrix():
    rng = random.Random(0)
    for _ in range(5):
        R = _random_rotation(rng)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_build_tip3p_universe_preserves_tip3p_geometry():
    rng = random.Random(42)
    u = build_tip3p_universe(
        bead_positions=np.array([[5.0, 5.0, 5.0]]),
        n_per_bead=1,
        water_resname="TIP3",
        box=BOX,
        bead_radius=2.35,
        resid_start=1,
        radius_scale=0.6,
        rng=rng,
    )

    assert len(u.atoms) == 3
    assert list(u.atoms.names) == ["OH2", "H1", "H2"]
    assert list(u.residues.resnames) == ["TIP3"]

    o, h1, h2 = u.atoms.positions
    assert np.linalg.norm(h1 - o) == pytest.approx(o_h_bond, abs=1e-4)
    assert np.linalg.norm(h2 - o) == pytest.approx(o_h_bond, abs=1e-4)
    assert _angle(h1 - o, h2 - o) == pytest.approx(h_o_h_ang, abs=1e-4)


def test_replace_martini_water_no_water_beads_returns_output_unchanged(make_universe):
    input_uni = make_universe(
        atom_names=["NC3"], resnames=["POPC"], atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]], box=BOX,
    )
    output_uni = make_universe(
        atom_names=["N"], resnames=["POPC"], atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]], box=BOX,
    )

    result = replace_martini_water(input_uni, output_uni=output_uni)

    assert result is output_uni


def test_replace_martini_water_builds_tip3p_for_each_bead(make_universe):
    input_uni = make_universe(
        atom_names=["W", "W"], resnames=["W", "W"], atom_resindex=[0, 1],
        positions=[[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]], box=BOX,
    )

    result = replace_martini_water(input_uni, output_uni=None, n_per_bead=4, rng=random.Random(1))

    assert len(result.atoms) == 2 * 4 * 3  # 2 beads x 4 waters x 3 atoms
    assert set(result.residues.resnames) == {"TIP3"}


def test_replace_martini_water_merges_and_continues_resids(make_universe):
    input_uni = make_universe(
        atom_names=["W"], resnames=["W"], atom_resindex=[0],
        positions=[[1.0, 1.0, 1.0]], box=BOX,
    )
    output_uni = make_universe(
        atom_names=["N"], resnames=["POPC"], atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]], box=BOX, resids=[3],
    )

    merged = replace_martini_water(input_uni, output_uni=output_uni, n_per_bead=1, rng=random.Random(2))

    assert len(merged.atoms) == 1 + 3
    assert merged.residues.resids[-1] == 4
