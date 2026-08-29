import numpy as np
import pytest


@pytest.fixture
def big_box():
    """A box large enough that PBC wrapping never kicks in for small test coordinates."""
    return np.array([1000.0, 1000.0, 1000.0, 90.0, 90.0, 90.0])


@pytest.fixture
def make_universe():
    """Factory for minimal single-frame MDAnalysis Universes used across structure_repair tests."""
    import MDAnalysis as mda

    def _make(atom_names, resnames, atom_resindex, positions, resids=None, box=None):
        atom_names = list(atom_names)
        resnames = list(resnames)
        n_atoms = len(atom_names)
        n_residues = len(resnames)
        if resids is None:
            resids = list(range(1, n_residues + 1))

        u = mda.Universe.empty(
            n_atoms=n_atoms,
            n_residues=n_residues,
            atom_resindex=list(atom_resindex),
            trajectory=True,
        )
        u.add_TopologyAttr("name", atom_names)
        u.add_TopologyAttr("resname", resnames)
        u.add_TopologyAttr("resid", resids)
        u.atoms.positions = np.asarray(positions, dtype=np.float32)
        if box is not None:
            u.dimensions = box
        return u

    return _make
