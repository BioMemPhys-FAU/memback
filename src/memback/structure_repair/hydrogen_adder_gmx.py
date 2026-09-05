import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np

from memback.features.calc_h_pos import calc_h_pos

def lipid_name_alignment(aa_names, pred_names):
    temp_index_map = {name: i for i, name in enumerate(aa_names)}
    align_arr = np.array([temp_index_map[name] for name in pred_names], dtype=int)
    return align_arr

def place_hydrogens(pred_uni, hdb, itp):
    topo_dict = {
        "resnames": [],
        "resids": [],
        "names": [],
        "segid": ["MEMB"],
    }
    new_pos = []
    atom_resindex = []
    alignment_map = {}
    for pred_res in pred_uni.residues:
        resname = pred_res.resname

        name_pos_map = {pred_res.atoms.names[i]: pred_res.atoms.positions[i] for i in range(len(pred_res.atoms))}
        h_pos = []
        h_names = []

        pending = list(hdb[resname])
        while pending:
            still_pending = []
            placed_this_pass = 0
            for each_map in pending:
                control_atoms = each_map['control_atoms']
                hydrogen_count = each_map['nr']
                tp = each_map['tp']

                stem = control_atoms[0]
                bonded_atoms = control_atoms[1:]
                if not all(a in name_pos_map for a in bonded_atoms[:4]):
                    still_pending.append(each_map)
                    continue

                if hydrogen_count == 1:
                    h_names_this = [stem]
                else:
                    h_names_this = [f"{stem}{i + 1}" for i in range(hydrogen_count)]

                c_atom_pos = np.zeros((4, 3))
                for i, atom_name in enumerate(bonded_atoms[:4]):
                    c_atom_pos[i] = name_pos_map[atom_name]

                hydrogen_pos = calc_h_pos(tp, c_atom_pos, None,
                                          n_requested=hydrogen_count)[:hydrogen_count]
                for name, pos in zip(h_names_this, hydrogen_pos):
                    name_pos_map[name] = pos
                    h_pos.append(pos)
                    h_names.append(name)
                placed_this_pass += 1

            if placed_this_pass == 0:
                missing = {a for m in still_pending
                           for a in m['control_atoms'][1:]
                           if a not in name_pos_map}
                raise ValueError(
                    f"Cannot place {len(still_pending)} hydrogen group(s) for "
                    f"{resname} (resid {pred_res.resid}); unresolved control "
                    f"atoms: {sorted(missing)}"
                )
            pending = still_pending

        new_res_pos = np.concatenate([pred_res.atoms.positions, h_pos])
        new_res_names = np.concatenate([pred_res.atoms.names, h_names])
        if resname not in alignment_map:
            align_arr = lipid_name_alignment(new_res_names, itp[resname]['atoms'])
            alignment_map[resname] = align_arr
        new_pos.extend(new_res_pos[alignment_map[resname]])
        topo_dict["resnames"].append(resname)
        topo_dict["resids"].append(pred_res.resid)
        topo_dict["names"].extend(new_res_names[alignment_map[resname]])
        atom_resindex.extend([pred_res.resindex] * len(new_res_names))

    new_pos = np.array(new_pos)
    u = mda.Universe.empty(n_atoms=new_pos.shape[0],
                           n_residues=len(pred_uni.residues),
                           atom_resindex=np.array(atom_resindex, dtype=int),
                           n_frames=1,
                           trajectory=True)
    for attr, value in topo_dict.items():
        u.add_TopologyAttr(attr, value)
    u.trajectory = MemoryReader(new_pos, order="fac", dimensions=pred_uni.dimensions)
    return u
