import numpy as np

from memback.structure_repair.replace_ions import replace_martini_ions

BOX = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]


def test_replace_martini_ions_maps_beads_to_charmm_atoms(make_universe):
    input_uni = make_universe(
        atom_names=["NA", "CL"],
        resnames=["ION", "ION"],
        atom_resindex=[0, 1],
        positions=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        box=BOX,
    )

    ion_u = replace_martini_ions(input_uni)

    assert len(ion_u.atoms) == 2
    assert list(ion_u.atoms.names) == ["SOD", "CLA"]
    assert list(ion_u.residues.resnames) == ["SOD", "CLA"]
    assert list(ion_u.residues.resids) == [1, 2]
    np.testing.assert_allclose(ion_u.atoms.positions, input_uni.atoms.positions)


def test_replace_martini_ions_no_ions_returns_input_unchanged(make_universe):
    input_uni = make_universe(
        atom_names=["NC3"],
        resnames=["POPC"],
        atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]],
        box=BOX,
    )

    result = replace_martini_ions(input_uni)

    assert result is input_uni


def test_replace_martini_ions_no_ions_returns_output_uni_when_given(make_universe):
    input_uni = make_universe(
        atom_names=["NC3"],
        resnames=["POPC"],
        atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]],
        box=BOX,
    )
    output_uni = make_universe(
        atom_names=["N"],
        resnames=["POPC"],
        atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]],
        box=BOX,
    )

    result = replace_martini_ions(input_uni, output_uni=output_uni)

    assert result is output_uni


def test_replace_martini_ions_merges_with_output_and_continues_resids(make_universe):
    input_uni = make_universe(
        atom_names=["CA"],
        resnames=["ION"],
        atom_resindex=[0],
        positions=[[9.0, 9.0, 9.0]],
        box=BOX,
    )
    output_uni = make_universe(
        atom_names=["N"],
        resnames=["POPC"],
        atom_resindex=[0],
        positions=[[0.0, 0.0, 0.0]],
        box=BOX,
        resids=[5],
    )

    merged = replace_martini_ions(input_uni, output_uni=output_uni)

    assert len(merged.atoms) == 2
    assert list(merged.residues.resids) == [5, 6]
    assert merged.residues.resnames[-1] == "CAL"
