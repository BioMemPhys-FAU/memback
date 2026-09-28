from pathlib import Path

import pytest

from memback import config
from memback.cli import main as cli_main
from memback.create_map import main, overmapped_beads
from memback.helpers import map_reader_full, read_bnd

M3_ITPS = Path(config.itp_m3_db_path)
AA_ITPS = Path(config.itp_db_path)


def _create(tmp_path, m3_file, m3name, aaname):
    prefix = tmp_path / aaname
    code = main(["--m3itp", str(M3_ITPS / m3_file), "--m3name", m3name,
                 "--aaitp", str(AA_ITPS / f"{aaname}.itp"), "-o", str(prefix)])
    return code, prefix


@pytest.mark.parametrize("m3_file, m3name, aaname", [
    ("martini_v3.0.0_phospholipids_PC_v2.itp", "DOPC", "DOPC"),    # 5-carbon first bead on both tails
    ("martini_v3.0.0_phospholipids_PE_v2.itp", "POPE", "POPE"),
    ("martini_v3.0.0_phospholipids_PI_v2.itp", "POPI", "POPI"),    # massless virtual bead C4
    ("martini_v3.0.0_phospholipids_SM_v2.itp", "NSM", "NSM"),      # letter-named chains S/F
])
def test_reproduces_shipped_map(tmp_path, m3_file, m3name, aaname):
    code, prefix = _create(tmp_path, m3_file, m3name, aaname)
    assert code == 0

    created = map_reader_full(f"{prefix}.map")[aaname]
    shipped = map_reader_full(config.map_path)[aaname]
    assert created == shipped    # heavy atoms, their order and bead types

    created_bonds = {frozenset(b) for b in read_bnd(f"{prefix}.bnd")[aaname]["bonds"]}
    shipped_bonds = {frozenset(b) for b in read_bnd(config.bond_map_path)[aaname]["bonds"]}
    assert created_bonds == shipped_bonds


def test_overmapped_bead_is_shared_by_all_5long_bonds():
    bonds = [("GL1", "C1A", "b_GL_C1_glyc_5long"), ("C1A", "D2A", "b_C1_C4_mid_5long"),
             ("D2A", "C3A", "b_C4_C1_mid"), ("GL2", "C1B", "b_GL_SC1_glyc")]
    assert overmapped_beads(["C1A", "D2A", "C3A", "C1B"], bonds) == {"C1A"}


def test_missing_input_exits_with_message(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["--m3itp", str(tmp_path / "missing.itp"), "--aaitp", str(tmp_path / "x.itp")])


def test_multi_molecule_itp_requires_m3name(tmp_path):
    with pytest.raises(SystemExit, match="--m3name"):
        main(["--m3itp", str(M3_ITPS / "martini_v3.0.0_phospholipids_PC_v2.itp"),
              "--aaitp", str(AA_ITPS / "DOPC.itp"), "-o", str(tmp_path / "DOPC")])


def test_cli_dispatches_create_map(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["create_map", "-h"])
    assert exc_info.value.code == 0
    assert "memback create_map" in capsys.readouterr().out
