import math

import numpy as np
import pytest

from memback.features.calc_h_pos import calc_h_pos

ALFA_H = math.acos(-1.0 / 3.0)
DIST_H = 1.0


def _angle(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.acos(np.clip(cos_a, -1.0, 1.0))

CENTRAL = np.array([0.0, 0.0, 0.0])
B1 = np.array([1.0, 0.0, 0.0])
B2 = np.array([0.0, 1.0, 0.0])
B3 = np.array([0.0, 0.0, 1.0])
XA = np.array([CENTRAL, B1, B2, B3])


def test_nht2_single_tetrahedral_h_bond_length():
    xh = calc_h_pos(2, XA, None)
    assert np.linalg.norm(xh[0] - CENTRAL) == pytest.approx(DIST_H)


def test_nht1_single_planar_h_bisects_the_two_neighbour_bonds():
    xh = calc_h_pos(1, XA, None)
    h = xh[0]
    assert np.linalg.norm(h - CENTRAL) == pytest.approx(DIST_H)
    expected_dir = (CENTRAL - B1) / np.linalg.norm(CENTRAL - B1) + (CENTRAL - B2) / np.linalg.norm(CENTRAL - B2)
    expected_dir /= np.linalg.norm(expected_dir)
    np.testing.assert_allclose(h - CENTRAL, DIST_H * expected_dir, atol=1e-8)


def test_nht3_two_planar_h_at_120_degrees():
    xh = calc_h_pos(3, XA, None)
    h0, h1 = xh[0], xh[1]
    assert np.linalg.norm(h0 - CENTRAL) == pytest.approx(DIST_H)
    assert np.linalg.norm(h1 - CENTRAL) == pytest.approx(DIST_H)
    alfa_hpl = 2.0 * math.pi / 3.0
    assert _angle(h0 - CENTRAL, h1 - CENTRAL) == pytest.approx(alfa_hpl, abs=1e-6)


def test_nht4_methyl_three_hydrogens_tetrahedral():
    xh = calc_h_pos(4, XA, None, n_requested=3)
    h0, h1, h2 = xh[0], xh[1], xh[2]
    for h in (h0, h1, h2):
        assert np.linalg.norm(h - CENTRAL) == pytest.approx(DIST_H)
    assert _angle(h0 - CENTRAL, h1 - CENTRAL) == pytest.approx(ALFA_H, abs=1e-6)
    assert _angle(h0 - CENTRAL, h2 - CENTRAL) == pytest.approx(ALFA_H, abs=1e-6)


def test_nht4_only_two_hydrogens_when_n_requested_is_two():
    xh = calc_h_pos(4, XA, None, n_requested=2)
    assert np.linalg.norm(xh[2]) == pytest.approx(0.0)


def test_nht5_tertiary_carbon_h_opposite_substituent_centroid():
    xh = calc_h_pos(5, XA, None)
    centroid = (B1 + B2 + B3) / 3.0
    expected_dir = CENTRAL - centroid
    expected_dir /= np.linalg.norm(expected_dir)
    h = xh[0]
    assert np.linalg.norm(h - CENTRAL) == pytest.approx(DIST_H)
    np.testing.assert_allclose((h - CENTRAL) / np.linalg.norm(h - CENTRAL), expected_dir, atol=1e-8)


def test_nht6_ch2_two_tetrahedral_hydrogens():
    xh = calc_h_pos(6, XA, None)
    h0, h1 = xh[0], xh[1]
    assert np.linalg.norm(h0 - CENTRAL) == pytest.approx(DIST_H)
    assert np.linalg.norm(h1 - CENTRAL) == pytest.approx(DIST_H)
    assert _angle(h0 - CENTRAL, h1 - CENTRAL) == pytest.approx(ALFA_H, abs=1e-6)


def test_nht7_water_hydrogens_advance_shared_l_ref_state():
    l_ref = [0]
    xh = calc_h_pos(7, XA, l_ref)

    aa, cc = 0.081649, 0.0577350
    expected_h0 = CENTRAL + np.array([aa, 0.0, cc])
    expected_h1 = CENTRAL + np.array([-aa, 0.0, cc])
    np.testing.assert_allclose(xh[0], expected_h0)
    np.testing.assert_allclose(xh[1], expected_h1)
    assert l_ref[0] == 1


def test_calc_h_pos_invalid_type_raises():
    with pytest.raises(ValueError):
        calc_h_pos(99, XA, None)
