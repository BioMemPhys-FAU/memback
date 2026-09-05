from memback.sim_preparer import min_mdp, min_mdp_prep, sim_preparer, topology_prep

BIG_BOX = [1000.0, 1000.0, 1000.0, 90.0, 90.0, 90.0]

def test_topology_prep_writes_includes_and_molecule_counts(tmp_path):
    metadata = [("POPC", 128), ("CHL1", 32)]

    topology_prep(metadata, str(tmp_path))

    content = (tmp_path / "topol.top").read_text()
    assert '#include "toppar/forcefield.itp"' in content
    assert '#include "toppar/POPC.itp"' in content
    assert '#include "toppar/CHL1.itp"' in content
    assert "[ system ]" in content
    assert "[ molecules ]" in content
    assert "POPC" in content and "128" in content
    assert "CHL1" in content and "32" in content
    # molecules block preserves input order
    assert content.index("POPC") < content.index("CHL1")


def test_min_mdp_prep_writes_expected_content(tmp_path):
    min_mdp_prep(str(tmp_path))

    written = (tmp_path / "min.mdp").read_text()
    assert written == min_mdp


def test_sim_preparer_end_to_end_with_bundled_forcefield(tmp_path, make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["TIP3"],
        atom_resindex=[0, 0],  # both atoms belong to the same residue
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )
    u.add_TopologyAttr("segid", ["SOLV"])
    pred_path = tmp_path / "input" / "backmapped.gro"
    pred_path.parent.mkdir()
    pred_path.write_text("dummy gro contents\n")
    output_path = tmp_path / "out"

    sim_preparer(u, str(output_path), pred_path=str(pred_path))

    assert (output_path / "topol.top").exists()
    assert (output_path / "min.mdp").exists()
    assert (output_path / "toppar" / "TIP3.itp").exists()
    assert (output_path / "backmapped.gro").exists()
    assert (output_path / "index.ndx").exists()
    run_sh = (output_path / "run_min.sh").read_text()
    assert "gmx grompp" in run_sh
    assert "backmapped.gro" in run_sh


def test_sim_preparer_extension_itp_overrides_bundled_one(tmp_path, make_universe):
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["TIP3"],
        atom_resindex=[0, 0],  # both atoms belong to the same residue
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )
    u.add_TopologyAttr("segid", ["SOLV"])
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "TIP3.itp").write_text("; custom override\n")
    output_path = tmp_path / "out"

    sim_preparer(u, str(output_path), ext_path=str(ext_dir))

    copied = (output_path / "toppar" / "TIP3.itp").read_text()
    assert copied == "; custom override\n"


def test_sim_preparer_missing_itp_is_reported_but_does_not_raise(tmp_path, capsys, make_universe):
    output_path = tmp_path / "out"
    u = make_universe(
        atom_names=["A", "B"],
        resnames=["NOT_A_REAL_RESIDUE"],
        atom_resindex=[0, 0],  # both atoms belong to the same residue
        positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        box=BIG_BOX,
    )
    u.add_TopologyAttr("segid", ["SOLV"])
    sim_preparer(u, str(output_path))

    captured = capsys.readouterr()
    assert "Could not find" in captured.out
    assert not (output_path / "toppar" / "NOT_A_REAL_RESIDUE.itp").exists()
