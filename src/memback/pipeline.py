import os
import time
from pathlib import Path

import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch

from memback.models.equivariant_memback import EquivariantBackmap
from memback.io.itp_to_hdb import itp_dir_to_hdb
from memback.config import (itp_db_path, hdb_path, martini3_to_charmm_lipids,
                            map_path, bond_map_path, model_path as default_model_path)
from memback.helpers import map_reader_full, calculate_distance, prepare_residue_data_prod_v2, read_bnd
from memback.structure_repair.hydrogen_adder_gmx import place_hydrogens
from memback.structure_repair.replace_water import replace_martini_water
from memback.structure_repair.replace_ions import replace_martini_ions
from memback.sim_preparer import sim_preparer
from memback.structure_repair.fix_clashes import fix_clashes
from memback.structure_repair.fix_chirality import fix_chirality
from memback.io.read_sim_metadata import read_hdb, read_itp_directory

__all__ = ["backmapping", "EquivariantBackmap"]


def universe_creator(result, mapping, dimensions):
    lipid_atoms = {resname: [name for each in list(res_map['atoms'].values()) for name in each] for resname, res_map in mapping.items()}
    # Get first frame for preparing universe metadata
    residues = result['pred_data']
    pred_aa_pos = result['pred_result']
    # Universe metadata
    # Single-frame pipeline: one frame is built below and handed to MemoryReader.
    n_frames = 1
    n_atoms = 0
    topo_dict = {
        "resnames": [],
        "resids": [],
        "names": [],
        "segid": None
    }
    atom_resindex = []
    n_residues = len(residues)
    for i, residue in enumerate(residues):
        atom_count = pred_aa_pos[i].shape[0]
        topo_dict["resnames"].append(residue.resname)
        n_atoms += atom_count
        topo_dict["names"].extend(lipid_atoms[residue.resname])
        topo_dict["resids"].append(residue.resid)
        atom_resindex.extend([i] * atom_count)

    u = mda.Universe.empty(n_atoms=n_atoms,
                           n_residues=n_residues,
                           atom_resindex=atom_resindex,
                           n_frames=n_frames,
                           trajectory=True)

    # PyCGTOOL starts resids from 0 instead of 1
    if min(topo_dict["resids"]) == 0:
        topo_dict["resids"] = [resid + 1 for resid in topo_dict["resids"]]
    for attr, value in topo_dict.items():
        u.add_TopologyAttr(attr, value)
    # Trajectory
    traj = []
    traj_dim = []

    frame_traj = []
    for pred_pos in result['pred_result']:
        frame_traj.extend(pred_pos)
    traj.append(frame_traj)
    traj_dim.append(dimensions)
    traj = np.array(traj)
    u.trajectory = MemoryReader(traj, order="fac", dimensions=traj_dim)

    return u

def prediction(model, data, mapping, device, no_universe=False):
    s = time.time()
    model.eval()

    batch = Batch.from_data_list(data["pred_data"]).to(device)
    with torch.no_grad():
        pred = model(batch)
        node_preds_per_graph = unbatch(pred, batch.batch)
        mask_per_graph = unbatch(batch.mask, batch.batch)
        bead_pos = unbatch(batch.pos, batch.batch)
        prediction_per_lipid = [((each + bead_pos[i][:,np.newaxis,:])[mask_per_graph[i]].detach().cpu() + data["pred_data"][i].center_of_geometry).numpy() for i, each in enumerate(node_preds_per_graph)]
    result = {"pred_data": data["pred_data"], "pred_result": prediction_per_lipid}
    if no_universe:
        return result
    u = universe_creator(result, mapping, data["dimensions"])
    e = time.time()
    print(f"Prediction took {e - s:.2f} seconds.")
    return u

def input_handler_single_frame(cg_universe, mapping, bnd_map):
    """
    Uses only single frame (.gro/.pdb)
    """
    start_time = time.time()
    # Common feature preparation
    res_dat, unique_resnames = prepare_residue_data_prod_v2(cg_universe, mapping, bnd_map)

    box_size = cg_universe.dimensions[:3].copy()
    dimensions = cg_universe.dimensions.copy()
    graphs = []
    # Select only residues from the mapping
    for residue in cg_universe.select_atoms("resname " + " or resname ".join(unique_resnames)).residues:
        resname = residue.resname

        mapping_filter = "name " + " or name ".join(mapping[resname]['atoms'].keys())
        atoms = residue.atoms.select_atoms(mapping_filter)
        cg_pos = atoms.positions
        residue_data = res_dat[resname]
        # Bead positions relative to lipid's center of geometry
        rel_cg_pos, _ = calculate_distance(cg_pos, cg_pos.mean(axis=0), box_size=box_size)

        diff_vec = rel_cg_pos[residue_data['connectivity'][0]] - rel_cg_pos[residue_data['connectivity'][1]]
        bond_lengths_attr = np.linalg.norm(diff_vec, axis=-1)
        # Bond vector calculation
        bond_vectors_attr = diff_vec / bond_lengths_attr.reshape(-1, 1)

        if "edge_types" in residue_data:
            edge_types = residue_data["edge_types"]
            edge_attr = np.concatenate([bond_vectors_attr, bond_lengths_attr.reshape(-1,1), edge_types], axis=-1, dtype=np.float32)
        else:
            edge_attr = np.concatenate([bond_vectors_attr, bond_lengths_attr.reshape(-1,1)], axis=-1, dtype=np.float32)

        graphs.append(Data(x= torch.tensor(residue_data["node_features"]),
        edge_index= torch.tensor(residue_data["connectivity"]),  # [2, num_edges]
        edge_attr= torch.tensor(edge_attr),
        mask= torch.tensor(residue_data["y_mask"]),  # shape: [N_beads * max_atom_number]
        pos= torch.tensor(rel_cg_pos),
        resname= resname,
        resid = residue.resid,
        center_of_geometry = torch.tensor(cg_pos.mean(axis=0))
        ))
    prediction_data = graphs
    result = {"pred_data": prediction_data, "dimensions": dimensions, "resnames": unique_resnames}
    end_time = time.time()
    print(f"Input graph creation took {end_time - start_time:.2f} seconds.")
    return result

def order_universe_lipids(u):
    order = list(dict.fromkeys(u.residues.resnames))
    combined_selection = u.atoms[[]]
    for name in order:
        group = u.select_atoms(f"resname {name}")
        combined_selection = combined_selection + group
    return combined_selection

def handle_mapping_extension(ext_path, mapping, bnd_map, hdb):
    files = os.listdir(ext_path)
    map_files = [file for file in files if file.endswith(".map")]
    bnd_files = [file for file in files if file.endswith(".bnd")]
    itp_files = [file for file in files if file.endswith(".itp")]
    mapping_ext = {}
    if len(map_files) > 0:
        for map_file in map_files:
            print(f"Reading map file {map_file} from extension folder.")
            mapping_ext.update(map_reader_full(f"{ext_path}/{map_file}"))

    bnd_map_ext = {}
    if len(bnd_files) > 0:
        for bnd_file in bnd_files:
            print(f"Reading bond file {bnd_file} from extension folder.")
            bnd_map_ext.update(read_bnd(f"{ext_path}/{bnd_file}"))

    hdb_ext = {}
    if len(itp_files) > 0:
        ext_hdb_path = f"{ext_path}/ext_hdb.hdb"
        itp_dir_to_hdb(ext_path, ext_hdb_path, itp_files)
        hdb_ext = read_hdb(ext_hdb_path)
    else:
        print(f"No .itp files in extension folder {ext_path}; skipping hydrogen database extension.")

    if not (map_files or bnd_files or itp_files):
        print(f"WARNING: extension folder {ext_path} contains no .map, .bnd or .itp files.")

    mapping.update(mapping_ext)
    bnd_map.update(bnd_map_ext)
    hdb.update(hdb_ext)
    return mapping, bnd_map, hdb

def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def backmapping(input_path, model_path=None, filename=None, ext_path=None, device=None):
    """
    Backmap a single-frame coarse-grained structure to all-atom.

    input_path : CG structure readable by MDAnalysis (.gro, .pdb, ...)
    model_path : checkpoint; defaults to the version shipped in model/
    filename   : output directory; defaults to <input stem>_backmapped
    ext_path   : optional directory of extra .map / .bnd / .itp files
    device     : torch.device; defaults to CUDA when available
    """
    if model_path is None:
        model_path = default_model_path
    if filename is None:
        filename = f"{Path(input_path).stem}_backmapped"
    if device is None:
        device = get_torch_device()

    os.makedirs(filename, exist_ok=True)
    start_time = time.time()
    mapping = map_reader_full(map_path)
    bnd_map = read_bnd(bond_map_path)
    hdb = read_hdb(hdb_path)

    if ext_path is not None:
        mapping, bnd_map, hdb = handle_mapping_extension(ext_path, mapping, bnd_map, hdb)
    input_uni = mda.Universe(input_path, to_guess=[])
    # Handle naming differences
    unique_resnames = set(input_uni.residues.resnames)
    for resname in unique_resnames:
        if resname in martini3_to_charmm_lipids:
            input_uni.select_atoms(f"resname {resname}").residues.resnames = martini3_to_charmm_lipids[resname]

    input_data = input_handler_single_frame(input_uni, mapping, bnd_map)
    model = EquivariantBackmap.from_checkpoint(model_path, map_location=device)
    res_uni = prediction(model, input_data, mapping, device)
    itp = read_itp_directory(itp_db_path, np.unique(res_uni.residues.resnames))
    if ext_path is not None:
        itp_ext = read_itp_directory(ext_path, np.unique(res_uni.residues.resnames))
        itp.update(itp_ext)
    res_uni_hydro = place_hydrogens(res_uni, hdb, itp)
    fix_chirality(res_uni_hydro, itp)
    res_uni_hydro = fix_clashes(res_uni_hydro, cut_off=0.3, include_intra=True, verbose=False)
    res_uni_water = replace_martini_water(input_uni=input_uni, output_uni=res_uni_hydro)
    res_uni_water_ions = replace_martini_ions(input_uni=input_uni, output_uni=res_uni_water)
    res_uni_water_ions = order_universe_lipids(res_uni_water_ions)
    pred_path = os.path.join(filename, "backmapped_ordered.gro")
    res_uni_water_ions.atoms.write(pred_path)

    sim_preparer(res_uni_water_ions, filename, pred_path=pred_path, ext_path=ext_path)
    end_time = time.time()
    print(f"Backmapping took {end_time - start_time:.2f} seconds. {input_path}")

