import shutil
from pathlib import Path

import pytest

from memback import config
from memback.check_maps import main
from memback.cli import main as cli_main
from memback.create_map import main as create_map_main

PC_ITP = Path(config.itp_m3_db_path) / "martini_v3.0.0_phospholipids_PC_v2.itp"
DOPC_ITP = Path(config.itp_db_path) / "DOPC.itp"


@pytest.fixture
def ext(tmp_path):
    """Extension folder holding a correct DOPC map, bnd and CHARMM itp."""
    folder = tmp_path / "ext"
    create_map_main(["--m3itp", str(PC_ITP), "--m3name", "DOPC",
                     "--aaitp", str(DOPC_ITP), "-o", str(folder / "DOPC")])
    shutil.copy(DOPC_ITP, folder)
    return folder


def _edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _check(ext, capsys):
    code = main(["-e", str(ext)])
    return code, capsys.readouterr().out


def test_clean_extension_passes(ext, capsys):
    code, out = _check(ext, capsys)
    assert code == 0
    assert "0 with errors, 0 with warnings" in out


def test_atom_in_two_beads_is_an_error(ext, capsys):
    _edit(ext / "DOPC.map", "\nPO4 Q5 -1 P ", "\nPO4 Q5 -1 N P ")
    code, out = _check(ext, capsys)
    assert code == 1
    assert "atom N is in beads NC3 and PO4" in out


def test_heavy_atom_missing_from_map_is_an_error(ext, capsys):
    _edit(ext / "DOPC.map", " O11", "")
    code, out = _check(ext, capsys)
    assert code == 1
    assert "heavy atoms missing from the map: ['O11']" in out


def test_unknown_hydrogen_is_only_a_warning(ext, capsys):
    _edit(ext / "DOPC.map", " O11", " O11 HX99")
    code, out = _check(ext, capsys)
    assert code == 0
    assert "HX99" in out
    assert main(["-e", str(ext), "--strict"]) == 1


def test_missing_bnd_section_is_an_error(ext, capsys):
    (ext / "DOPC.bnd").unlink()
    code, out = _check(ext, capsys)
    assert code == 1
    assert "no section in the .bnd files" in out


def test_bead_not_in_martini3_is_an_error(ext, capsys):
    _edit(ext / "DOPC.map", "\nGL2 ", "\nGLX ")
    code, out = _check(ext, capsys)
    assert code == 1
    assert "map beads not in Martini 3 DOPC: ['GLX']" in out


def test_conflicting_duplicate_sections_are_an_error(ext, capsys):
    text = (ext / "DOPC.map").read_text()
    (ext / "copy.map").write_text(text.replace(" O11", ""))
    code, out = _check(ext, capsys)
    assert code == 1
    assert "different duplicate [DOPC] sections" in out


def test_cli_dispatches_check_maps(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["check_maps", "-h"])
    assert exc_info.value.code == 0
    assert "memback check_maps" in capsys.readouterr().out


def test_chl1_is_exempt_from_martini3_comparison():
    from collections import defaultdict
    from memback.check_maps import check_lipid

    beads = {"ROH": ("P1", 0.0, ["O3"]), "R1": ("SC4", 0.0, ["C5"])}
    m3 = {"name": "CHOL", "beads": [{"name": "R1", "type": "SC4", "charge": 0.0}], "bonds": []}
    issues = defaultdict(list)
    check_lipid("CHL1", beads, [("ROH", "R1")], None, m3, issues, compare_m3=False)
    assert all("Martini 3" not in msg for _, msg in issues["CHL1"])
    check_lipid("CHL1", beads, [("ROH", "R1")], None, m3, issues)
    assert any("not in Martini 3 CHOL" in msg for _, msg in issues["CHL1"])
