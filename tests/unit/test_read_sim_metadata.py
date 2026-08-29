from memback.config import hdb_path, itp_db_path
from memback.io.read_sim_metadata import read_hdb, read_itp, read_itp_collection, read_itp_directory

FULL_ITP = """\
[ moleculetype ]
TST     3

[ atoms ]
     1    CT3      1  TST    C1     1   -0.27   12.011
     2     HA      1  TST    H1     1    0.09    1.008
     3    CT2      1  TST    C2     2   -0.18   12.011
     4     HA      1  TST    H2     2    0.09    1.008
     5     CT      1  TST    C3     3   -0.18   12.011

[ bonds ]
   1   2
   1   3
   3   4
   3   5

[ pairs ]
   1   4

[ angles ]
   2   1   3
   1   3   4

[ dihedrals ]
   2   1   3   4
   2   1   3   4   2

[ dihedral_restraints ]
   1   2   3   5   1   120.0
"""


def test_read_hdb_parses_entries(tmp_path):
    hdb_file = tmp_path / "tiny.hdb"
    hdb_file.write_text(
        "PRO       2\n"
        "    1  4  H1      C1    C2    C3  \n"
        "    1  4  H2      C1    C2    H4  \n"
    )

    result = read_hdb(str(hdb_file))

    assert list(result.keys()) == ["PRO"]
    entries = result["PRO"]
    assert len(entries) == 2
    assert entries[0] == {"nr": 1, "tp": 4, "control_atoms": ["H1", "C1", "C2", "C3"]}
    assert entries[1]["control_atoms"] == ["H2", "C1", "C2", "H4"]


def test_read_hdb_ignores_comment_lines(tmp_path):
    hdb_file = tmp_path / "tiny.hdb"
    hdb_file.write_text(
        "; a comment\n"
        "PRO       1\n"
        "; another comment\n"
        "    1  4  H1      C1    C2    C3  \n"
    )

    result = read_hdb(str(hdb_file))

    assert len(result["PRO"]) == 1


def test_read_itp_parses_all_known_blocks(tmp_path):
    itp_path = tmp_path / "full.itp"
    itp_path.write_text(FULL_ITP)

    result = read_itp(str(itp_path))

    mol = result["TST"]
    assert mol["atoms"] == ["C1", "H1", "C2", "H2", "C3"]
    assert ("C1", "H1") in mol["bonds"]
    assert ("C2", "H2") in mol["bonds"]
    assert mol["pairs"] == [("C1", "H2")]
    assert ("H1", "C1", "C2") in mol["angles"]
    assert mol["dihedrals"] == [("H1", "C1", "C2", "H2")]  # funct defaults to 1
    assert mol["impropers"] == [("H1", "C1", "C2", "H2")]  # funct 2 -> improper
    assert mol["dihedral_restraints"] == [("C1", "H1", "C2", "C3", 120.0)]


def test_read_itp_skips_unknown_sections(tmp_path):
    itp_path = tmp_path / "with_extra.itp"
    itp_path.write_text(
        "[ moleculetype ]\nTST 3\n\n"
        "[ atoms ]\n1 CT3 1 TST C1 1 0.0 12.011\n2 CT3 1 TST C2 1 0.0 12.011\n\n"
        "[ virtual_sitesn ]\n2 1 1\n\n"  # not in known_blocks -> its rows must be ignored
        "[ bonds ]\n1 2\n"
    )

    result = read_itp(str(itp_path))

    assert result["TST"]["atoms"] == ["C1", "C2"]
    # if "2 1 1" from the skipped section had leaked in as a bond row, this
    # would contain a bogus extra entry
    assert result["TST"]["bonds"] == [("C1", "C2")]


def test_read_itp_collection_merges_sources(tmp_path):
    itp_a = tmp_path / "a.itp"
    itp_a.write_text(FULL_ITP)
    itp_b = tmp_path / "b.itp"
    itp_b.write_text(FULL_ITP.replace("TST", "TS2"))

    result = read_itp_collection({"TST": str(itp_a), "TS2": str(itp_b)})

    assert set(result.keys()) == {"TST", "TS2"}


def test_read_itp_directory_filters_by_resname(tmp_path):
    (tmp_path / "AAA.itp").write_text(FULL_ITP.replace("TST", "AAA"))
    (tmp_path / "BBB.itp").write_text(FULL_ITP.replace("TST", "BBB"))

    result = read_itp_directory(str(tmp_path), resnames=["aaa"])  # case-insensitive

    assert set(result.keys()) == {"AAA"}


def test_read_itp_directory_no_filter_reads_everything(tmp_path):
    (tmp_path / "AAA.itp").write_text(FULL_ITP.replace("TST", "AAA"))
    (tmp_path / "BBB.itp").write_text(FULL_ITP.replace("TST", "BBB"))

    result = read_itp_directory(str(tmp_path), resnames=None)

    assert set(result.keys()) == {"AAA", "BBB"}


def test_bundled_hdb_and_itp_database_load_without_error():
    """Regression guard: the shipped databases must stay parseable by read_sim_metadata."""
    hdb = read_hdb(hdb_path)
    assert len(hdb) > 0

    itp_collection = read_itp_directory(itp_db_path, resnames=["TIP3"])
    assert "TIP3" in itp_collection
    assert len(itp_collection["TIP3"]["atoms"]) > 0
