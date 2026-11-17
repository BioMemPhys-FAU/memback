"""Tests for the temporary CG2AT2 protein backmapping (memback.protein_backmap)."""

import stat
import sys
from pathlib import Path

import pytest

from memback.cli import main
from memback.protein_backmap import (DEFAULT_CG2AT_ARGS, ProteinBackmapError, add_protein,
                                     protein_resnames, run_cg2at)

FAKE_PDB = """\
CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA     0      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA     0      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  N   GLY     1      12.000  10.000  10.000  1.00  0.00           N
END
"""

FAKE_TOP = """\
#include "../FINAL/charmm36-cg2at.ff/forcefield.itp"
#include "PROTEIN_0.itp"
[ system ]
x

[ molecules ]
; Compound        #mols
PROTEIN_0    1
"""

FAKE_ITP = """\
[ moleculetype ]
; Name            nrexcl
protein_0       3
#ifdef POSRES
#include "PROTEIN_0_posre.itp"
#endif
"""


def _fake_cg2at(tmp_path, fail=False):
    """Executable that records its arguments and writes CG2AT2-like FINAL files."""
    script = tmp_path / "fake_cg2at"
    script.write_text(f"""#!{sys.executable}
import sys
from pathlib import Path
Path("args.txt").write_text(" ".join(sys.argv[1:]))
if {fail}:
    print("Error: The bead SC1 is missing from the fragment library: ALA")
    sys.exit(1)
final = Path("cg2at/FINAL")
final.mkdir(parents=True)
(final / "final_cg2at_de_novo.pdb").write_text({FAKE_PDB!r})
(final / "topol_final.top").write_text({FAKE_TOP!r})
(final / "PROTEIN_0.itp").write_text({FAKE_ITP!r})
(final / "PROTEIN_0_posre.itp").write_text("[ position_restraints ]\\n")
# Like gmx editconf -c: the centred CG input, here shifted by +10 A in x.
lines = Path("protein_cg.gro").read_text().splitlines()
Path("cg2at/INPUT").mkdir(parents=True)
with open("cg2at/INPUT/CG_INPUT.pdb", "w") as pdb:
    for i, line in enumerate(lines[2:-1], 1):
        x, y, z = (float(v) * 10 for v in line[20:44].split())
        pdb.write(f"ATOM  {{i:5d}} {{line[10:15].strip():>4s}} {{line[5:10].strip():>3s}} {{i:5d}}    "
                  f"{{x + 10:8.3f}}{{y:8.3f}}{{z:8.3f}}  1.00  0.00\\n")
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def cg_system(make_universe):
    """Two protein beads (ALA, GLY) and one POPC bead."""
    u = make_universe(["BB", "BB", "NC3"], ["ALA", "GLY", "POPC"], [0, 1, 2],
                      [[1, 1, 1], [2, 1, 1], [3, 3, 3]], box=[50, 50, 50, 90, 90, 90])
    return u


def test_protein_resnames(cg_system):
    assert protein_resnames(cg_system) == {"ALA", "GLY"}


def test_run_cg2at_reads_final_outputs(tmp_path, cg_system):
    workdir = tmp_path / "work"
    protein, itps, molecules = run_cg2at(_fake_cg2at(tmp_path), cg_system.select_atoms("protein"),
                                         workdir, ["-fg", "martini_2-2_charmm36"])
    assert protein.atoms.n_atoms == 3
    assert [itp.name for itp in itps] == ["PROTEIN_0.itp", "PROTEIN_0_posre.itp"]
    # The moleculetype name of the .itp, not the include name of topol_final.top
    assert molecules == [("protein_0", 1)]
    args = (workdir / "args.txt").read_text().split()
    assert args[:4] == ["-c", "protein_cg.gro", "-loc", "cg2at"]
    assert args[4:] == DEFAULT_CG2AT_ARGS + ["-fg", "martini_2-2_charmm36"]
    assert (workdir / "protein_cg.gro").is_file()
    # CG2AT2's centring shift (+10 A in x) is undone
    assert protein.atoms.positions[0] == pytest.approx([0.0, 10.0, 10.0], abs=1e-3)


def test_run_cg2at_failure_shows_log(tmp_path, cg_system):
    with pytest.raises(ProteinBackmapError) as exc_info:
        run_cg2at(_fake_cg2at(tmp_path, fail=True), cg_system.select_atoms("protein"),
                  tmp_path / "work")
    message = str(exc_info.value)
    assert "exit status 1" in message and "cg2at.log" in message
    assert "SC1 is missing" in message


def test_add_protein_merges_files(tmp_path, cg_system, make_universe):
    protein, itps, molecules = run_cg2at(_fake_cg2at(tmp_path), cg_system.select_atoms("protein"),
                                         tmp_path / "work")
    out = tmp_path / "out"
    (out / "toppar").mkdir(parents=True)
    (out / "topol.top").write_text('#include "toppar/forcefield.itp"\n#include "toppar/POPC.itp"\n'
                                   "\n[ system ]\nx\n\n[ molecules ]\nPOPC  \t          1\n")
    system = make_universe(["N", "OW"], ["POPC", "TIP3"], [0, 1], [[5, 5, 5], [6, 6, 6]],
                           box=[50, 50, 50, 90, 90, 90])
    system.add_TopologyAttr("segids", ["MEMB"])
    water_segment = system.add_Segment(segid="SOLV")
    system.residues[1].segment = water_segment

    merged = add_protein(out, system, protein, itps, molecules, pred_name="backmapped_ordered.gro")

    assert list(merged.atoms.names) == ["N", "CA", "N", "N", "OW"]
    gro = (out / "backmapped_ordered.gro").read_text().splitlines()
    assert gro[1].strip() == "5" and "ALA" in gro[2] and "POPC" in gro[5]
    top = (out / "topol.top").read_text().splitlines()
    assert top[1] == '#include "toppar/PROTEIN_0.itp"'
    assert top[top.index("[ molecules ]") + 1].split() == ["protein_0", "1"]
    assert (out / "toppar" / "PROTEIN_0_posre.itp").is_file()
    ndx = (out / "index.ndx").read_text().split("[ SOLV ]")
    assert ndx[0].split()[3:] == ["1", "2", "3", "4"] and ndx[1].split() == ["5"]


def test_main_prot_back_args_need_prot_back(tmp_path):
    gro = tmp_path / "in.gro"
    gro.write_text("x\n0\n   5.0   5.0   5.0\n")
    with pytest.raises(SystemExit, match="--prot-back-args needs --prot-back"):
        main([str(gro), "--prot-back-args", "-fg x"])


def test_main_prot_back_executable_must_exist(tmp_path):
    gro = tmp_path / "in.gro"
    gro.write_text("x\n0\n   5.0   5.0   5.0\n")
    with pytest.raises(SystemExit, match="CG2AT2 executable not found"):
        main([str(gro), "--prot_back", str(tmp_path / "no_cg2at")])


def test_missing_mapping_message_suggests_prot_back_for_proteins():
    from memback.cli import missing_mapping_message

    assert "--prot-back" in missing_mapping_message(["ALA", "POPX"], Path("m.gro"))
    assert "--prot-back" not in missing_mapping_message(["POPX"], Path("m.gro"))
