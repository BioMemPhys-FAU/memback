import numpy as np
import MDAnalysis as mda
from collections import defaultdict

def signed_volume(c, a, b, d):
    return float(np.dot(a - c, np.cross(b - c, d - c)))

def min_image(dx, box):
    return dx - box * np.round(dx / box)

def signed_volume(c, a, b, d, box_size = None):
    if box_size is None:
        return float(np.dot(a - c, np.cross(b - c, d - c)))
    box_size = np.asarray(box_size, float)
    A = min_image(np.asarray(a, float) - c, box_size)
    B = min_image(np.asarray(b, float) - c, box_size)
    D = min_image(np.asarray(d, float) - c, box_size)
    return float(np.dot(A, np.cross(B, D)))

def flip_across_substituent_plane(X, center, sign_atoms, move_atoms, want_sign, box_size = None):
    X = np.asarray(X, float)
    a, b, d = sign_atoms
    if np.sign(signed_volume(X[center], X[a], X[b], X[d], box_size=box_size)) == want_sign:
        return X.copy(), False

    # plane through a, b, d
    p0 = X[a]
    n = np.cross(X[b] - X[a], X[d] - X[a])
    nrm = np.linalg.norm(n)
    if nrm < 1e-8:
        raise ValueError("substituent atoms are collinear; plane undefined")
    n /= nrm

    Y = X.copy()
    sel = np.asarray(move_atoms, int)
    v = X[sel] - p0
    Y[sel] = X[sel] - 2.0 * np.outer(v @ n, n)   # reflect across the plane
    return Y, True

def prepare_target_chirals(itp):
    target_chiral = defaultdict(list)
    for resname, itp_dat in itp.items():
        dihres = itp_dat['dihedral_restraints']
        if len(dihres) == 0:
            continue
        for dih in dihres:
            # R/S chirality only
            if abs(dih[-1]) == 120:
                itp[resname]['atoms'] = np.array(itp[resname]['atoms'])
                itp[resname]['bonds'] = np.array(itp[resname]['bonds'])
                sign_atoms = (np.where(itp[resname]["atoms"] == dih[0])[0][0],
                              np.where(itp[resname]["atoms"] == dih[1])[0][0],
                              np.where(itp[resname]["atoms"] == dih[3])[0][0])
                center = np.where(itp[resname]["atoms"] == dih[2])[0][0]
                idx = np.where(dih[2] == itp[resname]['bonds'])
                center_neighbors = itp[resname]["bonds"][idx[0], ~idx[1]]
                center_H = center_neighbors[np.where(np.char.startswith(center_neighbors, "H"))]
                center_H_indx = np.where(itp[resname]["atoms"] == center_H[0])[0]
                move_atoms = np.append(center_H_indx, center)
                want_sign = -1 if dih[-1] < 0 else 1
                target_chiral[resname].append({"sign_atoms": sign_atoms,
                                          "move_atoms": move_atoms,
                                          "center": center,
                                          "want_sign": want_sign})
    return target_chiral

def fix_chirality(uni, itp):
    target_chiral = prepare_target_chirals(itp)
    fix_count = 0
    for residue in uni.residues:
        resname = residue.resname
        if resname in target_chiral:
            for chiral in target_chiral[resname]:
                Y, flipped = flip_across_substituent_plane(
                    X = residue.atoms.positions,
                    box_size=uni.dimensions[:3],
                    # **target_chiral[resname]
                    **chiral
                )
                if flipped:
                    fix_count += 1
                    residue.atoms.positions = Y
    print(f"In total, {fix_count} chiral centers are fixed.")
    return uni
