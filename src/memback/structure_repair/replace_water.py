"""
martini_to_charmm_water.py
==========================
Replaces Martini3 coarse-grained water beads (W / PW / BMW) with
atomistic TIP3P water molecules suitable for CHARMM force fields.

Placement strategy:
  - Each Martini3 W bead has sigma=0.47 nm -> radius ~ 2.35 Angstrom.
  - 4 TIP3P molecules are placed at tetrahedral positions INSIDE the
    bead sphere, scaled to a fraction of the bead radius so all oxygens
    stay within the bead volume.
  - Each molecule is randomly rotated about its oxygen.
  - All positions are PBC-wrapped into the primary box before writing.
  - Energy minimisation after backmapping is mandatory.
"""

import argparse
import math
import random
from memback import config as globals
import numpy as np
import MDAnalysis as mda


martini3_water_radius = globals.bead_radii['W']
martini3_water_names = globals.martini3_water_names

# ---------------------------------------------------------------------------
# TIP3P geometry — CHARMM values
# ---------------------------------------------------------------------------
o_h_bond = 0.9572 # Angstrom
h_o_h_ang = math.radians(104.52)


def _tip3p_local_positions():
    """O at origin, molecule in xz-plane. Returns (3,3) array in Angstrom."""
    half = h_o_h_ang / 2.0
    return np.array([
        [0.0, 0.0, 0.0],
        [o_h_bond * math.sin(half), 0.0, o_h_bond * math.cos(half)],
        [-o_h_bond * math.sin(half), 0.0, o_h_bond * math.cos(half)],
    ])


# ---------------------------------------------------------------------------
# Tetrahedral unit vectors (4 directions, symmetric)
# ---------------------------------------------------------------------------
_tet_unit = np.array([
    [ 1, 1, 1],
    [-1,-1, 1],
    [-1, 1, -1],
    [ 1,-1, -1],
], dtype=float)
_tet_unit /= np.linalg.norm(_tet_unit[0])   # normalise to unit vectors


def _tet_offsets(bead_radius, radius_scale, n):
    """
    Return (n, 3) offset vectors placed inside the bead sphere.
    Offsets are scaled to bead_radius * radius_scale so all oxygens
    remain within the bead volume.

    For n <= 4: use tetrahedral vertices.
    For n > 4:  use Fibonacci sphere on the same radius.
    """
    r = bead_radius * radius_scale

    if n <= 4:
        return _tet_unit[:n] * r

    # Fibonacci sphere for n > 4
    offsets = []
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi   = math.pi * (1 + 5**0.5) * i
        offsets.append([math.sin(theta)*math.cos(phi),
                        math.sin(theta)*math.sin(phi),
                        math.cos(theta)])
    return np.array(offsets) * r


# ---------------------------------------------------------------------------
# Uniform random rotation (Shoemake quaternion method)
# ---------------------------------------------------------------------------

def _random_rotation(rng):
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    w = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    x = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    y = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    z = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def build_tip3p_universe(bead_positions, n_per_bead, water_resname,
                         box, bead_radius, resid_start, radius_scale, rng, verbose=False):
    """
    For each bead centre, place n_per_bead TIP3P molecules at tetrahedral
    positions inside the bead sphere. All positions are PBC-wrapped.
    """
    offsets = _tet_offsets(bead_radius, radius_scale, n_per_bead)
    tip3p_loc = _tip3p_local_positions()
    box_size = box[:3]

    all_positions = []
    all_atomnames = []
    all_resindex  = []
    water_idx     = 0

    for bead_pos in bead_positions:
        # Wrap bead center into primary box
        centre = bead_pos % box_size

        for k in range(n_per_bead):
            # Oxygen position: bead center + tetrahedral offset, wrapped
            o_pos = (centre + offsets[k]) % box_size

            # Random rotation of the TIP3P molecule about the oxygen
            rotate = _random_rotation(rng)
            rotated = tip3p_loc @ rotate.T    # (3,3): row i = position of atom i
            # H positions relative to oxygen, wrapped individually
            h1 = (o_pos + rotated[1]) % box_size
            h2 = (o_pos + rotated[2]) % box_size

            all_positions += [o_pos, h1, h2]
            all_atomnames += ["OH2", "H1", "H2"]
            all_resindex += [water_idx, water_idx, water_idx]
            water_idx += 1

    n_waters = water_idx
    n_atoms = len(all_positions)
    print(f"Placed {n_waters} TIP3P molecules ({n_atoms} atoms)")
    if verbose:
        print(f"Bead radius       : {bead_radius:.2f} A")
        print(f"O placement radius: {bead_radius * radius_scale:.2f} A "
              f"({radius_scale*100:.0f}% of bead radius)")

    u = mda.Universe.empty(
        n_atoms = n_atoms,
        n_residues = n_waters,
        n_segments = 1,
        atom_resindex = np.array(all_resindex, dtype=int),
        residue_segindex = np.zeros(n_waters, dtype=int),
        trajectory = True,
    )
    u.add_TopologyAttr("name",    all_atomnames)
    u.add_TopologyAttr("resname", [water_resname] * n_waters)
    u.add_TopologyAttr("resid",   np.arange(resid_start, resid_start + n_waters ))
    u.add_TopologyAttr("segid",   ["SOLV"])
    u.atoms.positions = np.array(all_positions, dtype=np.float32)
    return u

def replace_martini_water(input_uni, output_uni=None, water_resname_out="TIP3", n_per_bead=4,
                          radius_scale=0.6, rng=random.Random(42), verbose=False):

    box = input_uni.dimensions

    resname_sel = " or ".join(f"resname {r}" for r in martini3_water_names)
    cg_water  = input_uni.select_atoms(resname_sel)

    if len(cg_water) == 0:
        print("No CG water beads found for converting.")
        return output_uni if output_uni is not None else input_uni

    if output_uni is not None:
        resid_start = int(np.max(output_uni.residues.resids)) + 1
    else:
        resid_start = 1

    water_positions = cg_water.atoms.positions
    tip3p_u = build_tip3p_universe(
        water_positions, n_per_bead, water_resname_out,
        box, martini3_water_radius, resid_start, radius_scale, rng, verbose
    )
    if output_uni is None:
        return tip3p_u
    else:
        merged = mda.Merge(output_uni.atoms, tip3p_u.atoms)
        merged.dimensions = box
        return merged
