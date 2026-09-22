import MDAnalysis as mda
import os
import shutil
import numpy as np
from memback.config import itp_db_path, charmm_ff_path, mdp_path

run_sh = """#!/bin/bash

init="{pred_file}"
rest_prefix="{pred_file}"
mini_prefix="min"

echo "Starting Minimization..."
gmx grompp -f min.mdp -c ${{init}} -r ${{rest_prefix}} -p topol.top -o min.tpr 
gmx mdrun -v -deffnm min

echo "Starting Equilibration..."
cnt=1
cntmax=6
while [ $cnt -le $cntmax ]; do
    pcnt=$((cnt - 1))
    istep=$(printf "step6.%d_equilibration" $cnt)
    pstep=$(printf "step6.%d_equilibration" $pcnt)

    if [ $cnt -eq 1 ]; then
        pstep=${{mini_prefix}}
    fi

    echo "Running ${{istep}}"
    gmx grompp -f ${{istep}}.mdp -o ${{istep}}.tpr -c ${{pstep}}.gro -r ${{rest_prefix}} -p topol.top -n index.ndx

    gmx mdrun -v -deffnm ${{istep}}

    cnt=$((cnt + 1))
done

echo "Checking chiral centers with command python check_chirality.py step6.6_equilibration.gro ..."
python check_chirality.py step6.6_equilibration.gro
"""

def itps_prep(metadata, output_path, ext_path=None):
    os.makedirs(f"{output_path}/toppar", exist_ok=True)
    shutil.copytree(f"{charmm_ff_path}", f"{output_path}/toppar", dirs_exist_ok=True)
    for resname, _ in metadata:
        if ext_path is not None and os.path.exists(f"{ext_path}/{resname}.itp"):
            # Overwrites itp file from ITP database with extension itps
            shutil.copy2(f"{ext_path}/{resname}.itp", f"{output_path}/toppar")
        elif os.path.exists(f"{itp_db_path}/{resname}.itp"):
            shutil.copy2(f"{itp_db_path}/{resname}.itp", f"{output_path}/toppar")
        else:
            print(f"Could not find {resname}.itp in {itp_db_path} or {ext_path} for forcefield.")

def topology_prep(metadata, output_path, filename = "topol.top"):
    with open(f"{output_path}/{filename}", "w") as f:
        f.write('#include "toppar/forcefield.itp"\n')
        for resname, _ in metadata:
            f.write(f'#include "toppar/{resname}.itp"\n')
        f.write('\n[ system ]\n')
        f.write('Backmapped by MemBack\n')
        f.write('\n[ molecules ]\n')
        for resname, counts in metadata:
            f.write(f'{resname}  	          {counts}\n')

def mdp_prep(output_path):
    shutil.copytree(f"{mdp_path}", f"{output_path}", dirs_exist_ok=True)

def sim_preparer(uni, output_path, pred_path=None, ext_path=None):
    unique_resnames, index, counts = np.unique(uni.residues.resnames, return_counts=True,
                                               return_index=True)
    unique_resnames = unique_resnames[np.argsort(index)]
    counts = counts[np.argsort(index)]
    metadata = [(resname, count) for resname, count in zip(unique_resnames, counts)]

    os.makedirs(output_path, exist_ok=True)

    itps_prep(metadata, output_path, ext_path)

    topology_prep(metadata, output_path)

    mdp_prep(output_path)
    if pred_path is not None:
        destination = os.path.join(output_path, os.path.basename(pred_path))
        if os.path.abspath(pred_path) != os.path.abspath(destination):
            shutil.copy2(pred_path, destination)
        write_sh = run_sh.format(pred_file=os.path.basename(pred_path))
        with open(f"{output_path}/run_sim.sh", "w") as f:
            f.write(write_sh)
        # Prepare index file
        with mda.selections.gromacs.SelectionWriter(f'{output_path}/index.ndx', mode='w') as ndx:
            ndx.write(uni.select_atoms('segid MEMB'),
                      name='MEMB')
            ndx.write(uni.select_atoms('segid SOLV ION'),
                      name='SOLV')
