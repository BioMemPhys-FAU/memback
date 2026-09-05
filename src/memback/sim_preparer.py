import MDAnalysis as mda
import os
import shutil
import numpy as np
from memback.config import itp_db_path, charmm_ff_path

min_mdp = """define                  = -DPOSRES -DPOSRES_FC_LIPID=1000.0 -DDIHRES -DDIHRES_FC=1000.0
integrator              = steep
nsteps                  = 100
emtol                   = 1000.0
nstlist                 = 10
cutoff-scheme           = Verlet
rlist                   = 1.2
vdwtype                 = Cut-off
vdw-modifier            = Force-switch
rvdw_switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
;
constraints             = none
; Save options
nstxout                 = 1      ; full-precision coords (+velocities/forces) to .trr every step
nstfout                 = 1      ; forces to .trr (optional, useful for min analysis)
nstenergy               = 1      ; energies to .edr every step
nstlog                  = 1      ; energy summary to .log every step
nstcalcenergy           = 1      ; compute energy every step (needed for nstenergy=1)
"""


def itps_prep(metadata, output_path, ext_path=None):
    os.makedirs(f"{output_path}/toppar", exist_ok=True)
    # shutil.copy2(f"{itp_db_path}/forcefield.itp", f"{output_path}/toppar")
    shutil.copytree(f"{charmm_ff_path}", f"{output_path}/toppar", dirs_exist_ok=True)
    for resname, _ in metadata:
        if ext_path is not None and os.path.exists(f"{ext_path}/{resname}.itp"):
            # Overwrites itp file from ITP database with extension itps
            shutil.copy2(f"{ext_path}/{resname}.itp", f"{output_path}/toppar")
        elif os.path.exists(f"{itp_db_path}/{resname}.itp"):
            shutil.copy2(f"{itp_db_path}/{resname}.itp", f"{output_path}/toppar")
        else:
            print(f"Could not find {resname}.itp in {itp_db_path} or {ext_path} for forcefield.")
    pass

def topology_prep(metadata, output_path, filename = "topol.top"):
    with open(f"{output_path}/{filename}", "w") as f:
        f.write('#include "toppar/forcefield.itp"\n')
        for resname, _ in metadata:
            f.write(f'#include "toppar/{resname}.itp"\n')
        f.write('\n[ system ]\n')
        f.write('Backmapped by memBmap\n')
        f.write('\n[ molecules ]\n')
        for resname, counts in metadata:
            f.write(f'{resname}  	          {counts}\n')

def min_mdp_prep(output_path):
    with open(f"{output_path}/min.mdp", "w") as f:
        f.write(min_mdp)

def sim_preparer(uni, output_path, pred_path=None, ext_path=None):
    unique_resnames, index, counts = np.unique(uni.residues.resnames, return_counts=True,
                                               return_index=True)
    unique_resnames = unique_resnames[np.argsort(index)]
    counts = counts[np.argsort(index)]
    metadata = [(resname, count) for resname, count in zip(unique_resnames, counts)]

    os.makedirs(output_path, exist_ok=True)

    itps_prep(metadata, output_path, ext_path)

    topology_prep(metadata, output_path)

    min_mdp_prep(output_path)
    if pred_path is not None:
        destination = os.path.join(output_path, os.path.basename(pred_path))
        if os.path.abspath(pred_path) != os.path.abspath(destination):
            shutil.copy2(pred_path, destination)
        run_sh = f"#!/bin/bash \ngmx grompp -f min.mdp -c {os.path.basename(pred_path)} -r {os.path.basename(pred_path)} -p topol.top -o min.tpr \ngmx mdrun -v -deffnm min"
        with open(f"{output_path}/run_min.sh", "w") as f:
            f.write(run_sh)
        # Prepare index file
        with mda.selections.gromacs.SelectionWriter(f'{output_path}/index.ndx', mode='w') as ndx:
            ndx.write(uni.select_atoms('segid MEMB'),
                      name='MEMB')
            ndx.write(uni.select_atoms('segid SOLV ION'),
                      name='SOLV')
