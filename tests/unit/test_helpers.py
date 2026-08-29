import numpy as np
import pytest

from memback.helpers import calculate_dihedral, calculate_distance, map_reader_full, read_bnd


def test_map_reader_full_excludes_hydrogens_by_default(tmp_path):
    map_file = tmp_path / "tiny.map"
    map_file.write_text(
        "[POPC]\n"
        "NC3  Q1  1.0  N1  C1  H1  H2\n"
        "PO4  Q2  N2  C2\n"
    )

    result = map_reader_full(str(map_file))

    assert set(result.keys()) == {"POPC"}
    assert result["POPC"]["atoms"]["NC3"] == ["N1", "C1"]
    assert result["POPC"]["atoms"]["PO4"] == ["N2", "C2"]
    assert result["POPC"]["bead_type"] == {"NC3": "Q1", "PO4": "Q2"}


def test_map_reader_full_keeps_hydrogens_when_requested(tmp_path):
    map_file = tmp_path / "tiny.map"
    map_file.write_text("[POPC]\nNC3  Q1  1.0  N1  C1  H1  H2\n")

    result = map_reader_full(str(map_file), hydrogens=True)

    assert result["POPC"]["atoms"]["NC3"] == ["N1", "C1", "H1", "H2"]


def test_map_reader_full_ignores_comments_blank_lines_and_multiple_sections(tmp_path):
    map_file = tmp_path / "tiny.map"
    map_file.write_text(
        "; header comment\n"
        "\n"
        "[POPC]\n"
        "; a comment inside a section\n"
        "NC3  Q1  N1  C1\n"
        "\n"
        "[CHOL]\n"
        "ROH  P1  O1\n"
    )

    result = map_reader_full(str(map_file))

    assert set(result.keys()) == {"POPC", "CHOL"}
    assert result["POPC"]["atoms"]["NC3"] == ["N1", "C1"]
    assert result["CHOL"]["atoms"]["ROH"] == ["O1"]


def test_read_bnd_splits_bonds_and_angles(tmp_path):
    bnd_file = tmp_path / "tiny.bnd"
    bnd_file.write_text(
        "[POPC]\n"
        "NC3 PO4\n"
        "PO4 GL1\n"
        "\n"
        "NC3 PO4 GL1\n"
    )

    result = read_bnd(str(bnd_file))

    assert result["POPC"]["bonds"] == [("NC3", "PO4"), ("PO4", "GL1")]
    assert result["POPC"]["angles"] == [("NC3", "PO4", "GL1")]


def test_read_bnd_multiple_sections(tmp_path):
    bnd_file = tmp_path / "tiny.bnd"
    bnd_file.write_text(
        "[POPC]\n"
        "NC3 PO4\n"
        "\n"
        "[CHOL]\n"
        "ROH C1\n"
        "C1 C2\n"
        "\n"
        "ROH C1 C2\n"
    )

    result = read_bnd(str(bnd_file))

    assert result["POPC"]["bonds"] == [("NC3", "PO4")]
    assert result["POPC"]["angles"] == []
    assert result["CHOL"]["bonds"] == [("ROH", "C1"), ("C1", "C2")]
    assert result["CHOL"]["angles"] == [("ROH", "C1", "C2")]


def test_calculate_distance_no_pbc_wrap():
    box = np.array([100.0, 100.0, 100.0])
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 6.0, 3.0])

    vec, dist = calculate_distance(p1, p2, box)

    np.testing.assert_allclose(vec, p1 - p2)
    assert dist == pytest.approx(5.0)  # 3-4-5 triangle


def test_calculate_distance_applies_minimum_image_convention():
    box = np.array([10.0, 10.0, 10.0])
    p1 = np.array([0.5, 0.0, 0.0])
    p2 = np.array([9.5, 0.0, 0.0])

    _, dist = calculate_distance(p1, p2, box)

    # direct separation is 9.0 A, but across the periodic boundary it's 1.0 A
    assert dist == pytest.approx(1.0)


def test_calculate_dihedral_eclipsed_is_zero():
    pos = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    phi = calculate_dihedral(pos)

    assert phi == pytest.approx(0.0, abs=1e-6)


def test_calculate_dihedral_anti_is_pi():
    pos = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])

    phi = calculate_dihedral(pos)

    assert abs(phi) == pytest.approx(np.pi, abs=1e-6)


@pytest.mark.parametrize("theta_deg", [30, 60, 120, -45, -90])
def test_calculate_dihedral_matches_known_rotation_about_central_bond(theta_deg):
    theta = np.radians(theta_deg)
    pos0 = np.array([1.0, 1.0, 0.0])
    pos1 = np.array([0.0, 1.0, 0.0])
    pos2 = np.array([0.0, 0.0, 0.0])
    pos3 = pos2 + np.array([np.cos(theta), 0.0, -np.sin(theta)])
    pos = np.array([pos0, pos1, pos2, pos3])

    phi = calculate_dihedral(pos)

    assert phi == pytest.approx(-theta, abs=1e-6)
