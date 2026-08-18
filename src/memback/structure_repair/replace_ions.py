import numpy as np
import MDAnalysis as mda
from memback.config import martini3_to_charmm_ions, martini3_ion_names

def replace_martini_ions(input_uni,
                         output_uni=None,
                         ion_map=None):
    """
    Convert Martini 3 ion beads to CHARMM atomistic ions.
    """
    if ion_map is None:
        ion_map = martini3_to_charmm_ions
    # TODO Add a safeguard for ions without charmm maps
    cg_ions = input_uni.select_atoms("resname " + " or resname ".join(martini3_ion_names))

    if len(cg_ions) == 0:
        print("No Martini ion beads found for converting.")
        return output_uni if output_uni is not None else input_uni

    if output_uni is not None:
        resid_start = int(np.max(output_uni.residues.resids)) + 1
    else:
        resid_start = 1

    positions   = []
    atom_names  = []
    res_names   = []
    resids      = []

    for i, residue in enumerate(cg_ions.residues):
        cg_atomname = residue.atoms.names[0]
        charmm_resname, charmm_atomname = ion_map[cg_atomname]

        bead_pos = residue.atoms.positions[0]

        positions.append(bead_pos)
        atom_names.append(charmm_atomname)
        res_names.append(charmm_resname)
        resids.append(resid_start + i)

    n_atoms = len(positions)
    n_res   = n_atoms   # one atom per residue for monatomic ions

    print(f"Placed {n_atoms} atomistic ions ")

    ion_u = mda.Universe.empty(
        n_atoms          = n_atoms,
        n_residues       = n_res,
        n_segments       = 1,
        atom_resindex    = np.arange(n_res, dtype=int),
        residue_segindex = np.zeros(n_res, dtype=int),
        trajectory       = True,
    )
    ion_u.add_TopologyAttr("name",     atom_names)
    ion_u.add_TopologyAttr("resname",  res_names)
    ion_u.add_TopologyAttr("resid",    np.array(resids, dtype=int))
    ion_u.atoms.positions = np.array(positions, dtype=np.float32)
    ion_u.dimensions = input_uni.dimensions

    if output_uni is None:
        return ion_u
    else:
        merged = mda.Merge(output_uni.atoms, ion_u.atoms)
        merged.dimensions = input_uni.dimensions
        return merged
