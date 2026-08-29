import pytest

from memback.io.itp_to_hdb import (
    HdbBuildError,
    _is_hydrogen,
    _read_itp_atoms_bonds,
    build_hdb_entries,
    format_hdb,
    itp_dir_to_hdb,
    itp_to_hdb,
)

# A tiny propane-like molecule: C1(H1,H2,H3)-C2(H4,H5)-C3(H6,H7,H8)
# C1/C3 are methyls (3 H, 1 heavy neighbour) -> hdb type 4
# C2 is a mid-chain CH2 (2 H, 2 heavy neighbours) -> hdb type 6
PROPANE_ITP = """\
[ moleculetype ]
; name  nrexcl
PRO     3

[ atoms ]
;   nr  type  resnr  res  atom  cgnr  charge   mass
     1    CT3      1  PRO    C1     1   -0.27   12.011
     2     HA      1  PRO    H1     1    0.09    1.008
     3     HA      1  PRO    H2     1    0.09    1.008
     4     HA      1  PRO    H3     1    0.09    1.008
     5    CT2      1  PRO    C2     2   -0.18   12.011
     6     HA      1  PRO    H4     2    0.09    1.008
     7     HA      1  PRO    H5     2    0.09    1.008
     8    CT3      1  PRO    C3     3   -0.27   12.011
     9     HA      1  PRO    H6     3    0.09    1.008
    10     HA      1  PRO    H7     3    0.09    1.008
    11     HA      1  PRO    H8     3    0.09    1.008

[ bonds ]
   1   2
   1   3
   1   4
   1   5
   5   6
   5   7
   5   8
   8   9
   8  10
   8  11
"""


def test_read_itp_atoms_bonds_parses_moleculetype_atoms_and_bonds(tmp_path):
    itp_path = tmp_path / "propane.itp"
    itp_path.write_text(PROPANE_ITP)

    molname, atoms, bonds = _read_itp_atoms_bonds(str(itp_path))

    assert molname == "PRO"
    assert len(atoms) == 11
    assert atoms[0] == {"idx": 1, "name": "C1", "mass": 12.011, "type": "CT3"}
    assert (1, 2) in bonds
    assert (5, 8) in bonds
    assert len(bonds) == 10


def test_is_hydrogen_uses_mass_when_available():
    assert _is_hydrogen({"mass": 1.008, "type": "HA", "name": "H1"}) is True
    assert _is_hydrogen({"mass": 12.011, "type": "CT3", "name": "C1"}) is False


def test_is_hydrogen_falls_back_to_name_or_type_without_mass():
    assert _is_hydrogen({"mass": None, "type": "HA", "name": "H1"}) is True
    assert _is_hydrogen({"mass": None, "type": "opls_140", "name": "HB2"}) is True
    assert _is_hydrogen({"mass": None, "type": "CT3", "name": "C1"}) is False


def test_build_hdb_entries_methyl_group_is_type4(tmp_path):
    itp_path = tmp_path / "propane.itp"
    itp_path.write_text(PROPANE_ITP)

    molname, entries = build_hdb_entries(str(itp_path))

    assert molname == "PRO"
    methyl_entries_c1 = [e for e in entries if e[1] == 4 and e[3][0] == "C1"]
    assert len(methyl_entries_c1) == 3
    for n_add, add_type, h_names, ctrl in methyl_entries_c1:
        assert n_add == 1
        assert h_names[0] in {"H1", "H2", "H3"}
        assert ctrl[0] == "C1"
        assert ctrl[1] == "C2"  # heavy neighbour of C1


def test_build_hdb_entries_ch2_group_is_type6(tmp_path):
    itp_path = tmp_path / "propane.itp"
    itp_path.write_text(PROPANE_ITP)

    _, entries = build_hdb_entries(str(itp_path))

    c2_entries = [e for e in entries if e[1] == 6 and e[3][0] == "C2"]
    assert len(c2_entries) == 2
    assert sorted(e[2][0] for e in c2_entries) == ["H4", "H5"]
    for n_add, add_type, h_list, ctrl in c2_entries:
        assert n_add == 1
        assert set(ctrl[1:]) == {"C1", "C3"}


def test_build_hdb_entries_total_hydrogen_count_matches_source(tmp_path):
    itp_path = tmp_path / "propane.itp"
    itp_path.write_text(PROPANE_ITP)

    _, entries = build_hdb_entries(str(itp_path))

    assert sum(e[0] for e in entries) == 8  # H1..H8


def test_build_hdb_entries_raises_without_moleculetype(tmp_path):
    itp_path = tmp_path / "broken.itp"
    itp_path.write_text("[ atoms ]\n1 CT3 1 X C1 1 0.0 12.011\n")

    with pytest.raises(HdbBuildError):
        build_hdb_entries(str(itp_path))


def test_format_hdb_header_and_row_layout():
    entries = [(1, 4, ["H1"], ["C1", "C2", "H4"])]

    block = format_hdb("PRO", entries)

    lines = block.splitlines()
    assert lines[0].split() == ["PRO", "1"]
    assert lines[1].split() == ["1", "4", "H1", "C1", "C2", "H4"]


def test_itp_to_hdb_writes_expected_file(tmp_path):
    itp_path = tmp_path / "propane.itp"
    itp_path.write_text(PROPANE_ITP)
    out_path = tmp_path / "out.hdb"

    block = itp_to_hdb(str(itp_path), out_path=str(out_path), verbose=False)

    written = out_path.read_text()
    assert written == block
    assert written.startswith("PRO")


def test_itp_dir_to_hdb_combines_and_filters_files(tmp_path):
    # itp_dir_to_hdb names each block after the FILE's stem (uppercased), not
    # the [ moleculetype ] name inside it -- so "propane.itp" -> "PROPANE".
    (tmp_path / "propane.itp").write_text(PROPANE_ITP)
    (tmp_path / "other.itp").write_text(PROPANE_ITP.replace("PRO", "PR2"))
    out_path = tmp_path / "combined.hdb"

    text = itp_dir_to_hdb(str(tmp_path), str(out_path), target_itps=["propane.itp"])

    assert out_path.read_text() == text
    assert "PROPANE" in text  # included file, named after its filename stem
    assert "OTHER" not in text  # excluded by target_itps filter


def test_itp_dir_to_hdb_combines_multiple_files_in_sorted_order(tmp_path):
    (tmp_path / "propane.itp").write_text(PROPANE_ITP)
    (tmp_path / "other.itp").write_text(PROPANE_ITP.replace("PRO", "PR2"))
    out_path = tmp_path / "combined.hdb"

    text = itp_dir_to_hdb(str(tmp_path), str(out_path))

    assert text.startswith("OTHER")  # glob() sorts alphabetically: other.itp first
    assert "\nPROPANE" in text
    assert text.index("OTHER") < text.index("PROPANE")
