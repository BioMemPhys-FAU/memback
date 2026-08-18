import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance, minimize_vectors
import warnings
warnings.filterwarnings("ignore")

def detect_clashes(universe, cutoff=0.50, include_intra=False, exclude_resnames=["TIP3", "SOD", "CLA"]):
    box = universe.dimensions
    query_str = "not resname " + " and not resname ".join(exclude_resnames)

    pos = universe.select_atoms(query_str).atoms.positions.astype(np.float32)
    pairs, dists = self_capped_distance(pos, max_cutoff=cutoff, box=box,
                                        return_distances=True)
    resix = universe.atoms.resindices

    if len(pairs) == 0:
        return pairs, dists

    i, j = pairs[:, 0], pairs[:, 1]
    is_clash = np.ones(len(pairs), dtype=bool)
    if not include_intra:
        is_clash = resix[i] != resix[j] # ignore same-lipid (bonded) contacts
    return pairs[is_clash], dists[is_clash]

def fix_clashes(universe, cut_off=0.50,
                max_iter=60, verbose=False, include_intra=False,):
    box = universe.dimensions
    pos = universe.atoms.positions.astype(np.float32)
    resix = universe.atoms.resindices
    counts = 0
    for it in range(1, max_iter + 1):
        universe.atoms.positions = pos
        pairs, dists = detect_clashes(universe, cut_off, include_intra)
        n = len(pairs)
        if verbose:
            worst = dists.min() if n else float("nan")
            print(f"  iter {it:2d}: {n:7d} clashes, closest = {worst:.3f} A")
        if n == 0:
            break

        order = np.argsort(dists)
        i_arr, j_arr = pairs[:, 0], pairs[:, 1]

        moved_res = set()
        for k in order:
            i, j = int(i_arr[k]), int(j_arr[k])
            ri, rj = int(resix[i]), int(resix[j])
            if ri in moved_res or rj in moved_res:
                continue
            vec = minimize_vectors((pos[i] - pos[j])[None, :], box)[0]
            d = float(np.linalg.norm(vec))
            if d >= cut_off:
                continue
            direction = vec / d
            half = 0.5 * (cut_off + 0.01 - d)
            pos[i] += direction * half
            pos[j] -= direction * half
            moved_res.add(ri)
            moved_res.add(rj)
            counts += 1
    else:
        if verbose:
            print(f"  reached max_iter={max_iter} with clashes remaining")
    print(f"{counts} clashes fixed in {it} iterations. (Cutoff {cut_off:.2f} A)")
    universe.atoms.positions = pos
    return universe

