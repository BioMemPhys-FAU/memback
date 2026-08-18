import numpy as np
import re
from memback import config as globals

bead_classes = globals.bead_classes
bead_sizes = globals.bead_sizes
max_atom_number = globals.max_atom_number

def map_reader_full(map_path, hydrogens=False):
    """
    Reads the lipid map file and returns a dictionary of lipid types and their corresponding atoms with excluding hydrogen atoms.
    """
    result_map = {}
    section = None

    with open(map_path) as f:
        for line in f:
            line = line.strip()
            # skip comments / empty lines
            if not line or line.startswith(("#", ";")):
                continue
            # section header
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                result_map[section] = {"atoms": {}, "bead_type": {}}
                continue
            vals = line.split()
            cg = vals[0]
            try:
                float(vals[2])
                start = 3
            except ValueError:
                start = 2
            aa = vals[start:] if hydrogens else [
                atom for atom in vals[start:]
                if not atom.startswith("H")
            ]
            result_map[section]["atoms"][cg] = aa
            result_map[section]["bead_type"][cg] = vals[1]
    return result_map

def read_bnd(path):
    """
    Reads the lipid bond file and returns a dictionary of lipid types and their corresponding bead bonds and angles.
    """
    data = {}
    section = None
    mode = "bonds"
    with open(path) as f:
        for line in f:
            line = line.strip()
            # skip comments / empty lines
            if not line or line.startswith(("#", ";")):
                if section and mode == "bonds":
                    mode = "angles"   # blank line separates bonds/angles
                continue
            # new section
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                data[section] = {"bonds": [], "angles": []}
                mode = "bonds"
                continue
            atoms = tuple(line.split())
            if len(atoms) == 2:
                data[section]["bonds"].append(atoms)
            elif len(atoms) == 3:
                data[section]["angles"].append(atoms)
    return data

def calculate_distance(position_1, position_2, box_size):
    """
    Calculates the distance between positions w.r.t PBC box dimensions.
    """
    result_pos = position_1 - position_2
    result_pos -= box_size * np.round(result_pos / box_size)
    distance = np.linalg.norm(result_pos, axis=-1)
    return result_pos, distance

def calculate_dihedral(pos, torsion_index = None, box_size = None):
    if torsion_index is None:
        torsion_index = [0, 1, 2, 3]
    i, j, k, l = torsion_index
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]
    if box_size is not None:
        b1 -= box_size * np.round(b1 / box_size)
        b2 -= box_size * np.round(b2 / box_size)
        b3 -= box_size * np.round(b3 / box_size)

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = np.linalg.norm(b2, axis=-1)

    cos_phi = np.sum(n1 * n2, axis=-1) * b2n
    sin_phi = np.sum(np.cross(n1, n2) * b2, axis=-1) #/ b2n
    phi = np.arctan2(sin_phi, cos_phi)

    return phi


# With atom information inside of node features
def prepare_residue_data_prod_v2(cg_universe, mapping, bnd_info):
    """
    Prepares a look-up table for each lipid type. Since lipid has the same features among different copies in the same type.
    """
    res_dat = {}
    for resname in np.unique(cg_universe.residues.resnames):
        if resname in mapping:
            res_dat[resname] = {}
        elif resname in globals.martini3_excluded_residues:
            continue
        else:
            print(f"Residue {resname} not found in mapping. Skipping... ")
            continue

    for resname in res_dat.keys():
        # Remove the beads that are not in mapping file. (Added for virtual bead cases such as POPI33)
        mapping_filter = "name " + " or name ".join(mapping[resname]['atoms'].keys())
        # res = cg_universe.select_atoms(f"resname {resname} and ({mapping_filter})").residues[0]
        res = cg_universe.select_atoms(f"resname {resname}").residues[0].atoms.select_atoms(mapping_filter)
        atom_indices = {name: i for i, name in enumerate(res.atoms.names)}
        res_dat[resname]['atom_indices'] = atom_indices
        # Edges:
        # Edge attribute (bond length) will be calculated later.
        connectivity = []
        neighbor_counts = {atom_name: 0 for atom_name in res.atoms.names}
        for bond in bnd_info[resname]['bonds']:
            neighbor_counts[bond[0]] += 1
            neighbor_counts[bond[1]] += 1
            first_indx = atom_indices[bond[0]]
            second_indx = atom_indices[bond[1]]
            connectivity.append([first_indx, second_indx])
            connectivity.append([second_indx, first_indx])
        # Finalizing edge index tensor
        res_dat[resname]['connectivity'] = np.array(connectivity, dtype=np.int32).T

        # Node Features:
        atom_neighbor_counts = np.array([neighbor_counts[atom_name] for atom_name in res.atoms.names], dtype=np.float32).reshape(-1, 1)
        atom_local_density_attr = np.array([len(each_map) for each_map in mapping[resname]["atoms"].values()],
                                               dtype=np.float32).reshape(-1, 1)
        # Node feature pre-processing
        bead_class = np.array(
            [next((bead_classes[char] for char in mapping[resname]['bead_type'][bead_name] if char in bead_classes), bead_classes['UNK']) for bead_name
             in res.atoms.names]).reshape(-1, 1)
        # Find the Size (defaults to UNK if not found)
        bead_size = np.array(
            [next((bead_sizes[char] for char in mapping[resname]['bead_type'][bead_name] if char in bead_sizes), bead_sizes['R']) for bead_name
             in res.atoms.names]).reshape(-1, 1)
        number_matches = [re.search(r'\d+', mapping[resname]['bead_type'][bead_name]) for bead_name
             in res.atoms.names]
        # 0 means NaN
        bead_polarity = np.array([int(number_match.group()) if number_match else 0 for number_match in number_matches]).reshape(-1, 1)

        # Finalizing node feature
        node_features = np.concatenate(
            (bead_class, bead_size, bead_polarity, atom_neighbor_counts, atom_local_density_attr),
            axis=1)
        res_dat[resname]['node_features'] = node_features
        # Mask for padding
        y_mask = np.zeros((len(res.atoms), max_atom_number), dtype=bool)
        for bead_name, bead_indx in atom_indices.items():
            n_atoms = int(atom_local_density_attr[bead_indx][0])
            y_mask[bead_indx][:n_atoms] = True
        res_dat[resname]['y_mask'] = y_mask

    return res_dat, np.array(list(res_dat.keys()))