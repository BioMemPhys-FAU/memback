import numpy as np
import pytest

from memback.structure_repair.fix_chirality import (
    fix_chirality,
    flip_across_substituent_plane,
    min_image,
    prepare_target_chirals,
    signed_volume,
)


def test_signed_volume_is_antisymmetric_under_vertex_swap():
    c = np.array([0.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])

    assert signed_volume(c, a, b, d) == pytest.approx(1.0)
    assert signed_volume(c, b, a, d) == pytest.approx(-1.0)


def test_signed_volume_with_box_uses_minimum_image():
    box = np.array([10.0, 10.0, 10.0])
    c = np.array([0.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    d_wrapped = d + np.array([10.0, 0.0, 0.0])

    assert signed_volume(c, a, b, d_wrapped, box_size=box) == pytest.approx(
        signed_volume(c, a, b, d, box_size=box)
    )


def test_min_image_wraps_into_half_box():
    box = np.array([10.0, 10.0, 10.0])
    dx = np.array([9.5, 0.0, 0.0])

    np.testing.assert_allclose(min_image(dx, box), np.array([-0.5, 0.0, 0.0]))


def test_flip_across_substituent_plane_no_op_when_sign_already_matches():
    X = np.array([
        [0.3, 0.3, 0.3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.sign(signed_volume(X[0], X[1], X[2], X[3])) == 1

    Y, flipped = flip_across_substituent_plane(
        X, center=0, sign_atoms=(1, 2, 3), move_atoms=[0], want_sign=1
    )

    assert flipped is False
    np.testing.assert_array_equal(Y, X)
    assert Y is not X


def test_flip_across_substituent_plane_flips_when_sign_mismatches():
    X = np.array([
        [0.3, 0.3, 0.3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.sign(signed_volume(X[0], X[1], X[2], X[3])) == 1

    Y, flipped = flip_across_substituent_plane(
        X, center=0, sign_atoms=(1, 2, 3), move_atoms=[0], want_sign=-1
    )

    assert flipped is True
    assert np.sign(signed_volume(Y[0], Y[1], Y[2], Y[3])) == -1
    assert not np.allclose(Y[0], X[0])
    np.testing.assert_array_equal(Y[1:], X[1:])


def test_flip_across_substituent_plane_raises_on_collinear_reference_atoms():
    X = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    with pytest.raises(ValueError, match="collinear"):
        flip_across_substituent_plane(X, center=0, sign_atoms=(1, 2, 3), move_atoms=[0], want_sign=1)


def _tiny_itp_with_chiral_restraint(angle=120):
    return {
        "CHI": {
            "atoms": ["A", "B", "C", "D", "H1"],
            "bonds": [("C", "A"), ("C", "B"), ("C", "D"), ("C", "H1")],
            "dihedral_restraints": [("A", "B", "C", "D", angle)],
        }
    }


def test_prepare_target_chirals_extracts_sign_and_move_atoms():
    itp = _tiny_itp_with_chiral_restraint(angle=120)

    target = prepare_target_chirals(itp)

    assert list(target.keys()) == ["CHI"]
    entry = target["CHI"][0]
    assert entry["sign_atoms"] == (0, 1, 3)
    assert entry["center"] == 2
    np.testing.assert_array_equal(np.sort(entry["move_atoms"]), [2, 4])
    assert entry["want_sign"] == 1


def test_prepare_target_chirals_negative_angle_gives_negative_want_sign():
    itp = _tiny_itp_with_chiral_restraint(angle=-120)

    target = prepare_target_chirals(itp)

    assert target["CHI"][0]["want_sign"] == -1


def test_prepare_target_chirals_ignores_non_chiral_restraints():
    itp = _tiny_itp_with_chiral_restraint(angle=60)

    target = prepare_target_chirals(itp)

    assert "CHI" not in target


def test_prepare_target_chirals_skips_residues_without_restraints():
    itp = {"PLAIN": {"atoms": [], "bonds": [], "dihedral_restraints": []}}

    target = prepare_target_chirals(itp)

    assert "PLAIN" not in target


def test_fix_chirality_flips_a_mismatched_center(make_universe):
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.3, 0.3, -0.3],
        [1.0, 0.0, 0.0],
        [0.3, 0.3, 0.7],
    ])
    u = make_universe(
        atom_names=["A", "B", "C", "D", "H1"],
        resnames=["CHI"],
        atom_resindex=[0, 0, 0, 0, 0],
        positions=positions,
        box=[100.0, 100.0, 100.0, 90.0, 90.0, 90.0],
    )
    itp = _tiny_itp_with_chiral_restraint(angle=120)

    fix_chirality(u, itp)

    residue_pos = u.residues[0].atoms.positions
    a, b, c, d = residue_pos[0], residue_pos[1], residue_pos[2], residue_pos[3]
    assert np.sign(signed_volume(c, a, b, d, box_size=u.dimensions[:3])) == 1
