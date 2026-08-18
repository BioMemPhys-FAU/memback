import numpy as np
import math

DIM = 3
NOTSET = np.nan

def norm(v):
    return np.linalg.norm(v)

def copy_rvec(src):
    return np.array(src, dtype=float)

def rvec_sub(a, b):
    return np.array(a) - np.array(b)

def cprod(a, b):
    return np.cross(a, b)

def square(x):
    return x * x

def gen_waterhydrogen(nh, xa, xh, l_ref):
    AA = 0.081649
    BB = 0.0
    CC = 0.0577350

    matrix1 = np.array([
        [ AA,  BB,  CC],
        [ AA,  BB,  CC],
        [ AA,  BB,  CC],
        [-AA,  BB,  CC],
        [-AA,  BB,  CC],
        [ BB,  AA, -CC],
    ])

    matrix2 = np.array([
        [-AA,  BB,  CC],
        [ BB,  AA, -CC],
        [ BB, -AA, -CC],
        [ BB,  AA, -CC],
        [ BB, -AA, -CC],
        [ BB, -AA, -CC],
    ])

    idx = l_ref[0]
    xAI = xa[0]

    xh[0][:] = xAI + matrix1[idx]
    xh[1][:] = xAI + matrix2[idx]

    if nh > 2:
        xh[2][:] = copy_rvec(xAI)
    if nh > 3:
        xh[3][:] = copy_rvec(xAI)

    l_ref[0] = (idx + 1) % 6

def calc_h_pos(nht, xa, l_ref, n_requested=None):
    xh = np.zeros(shape=(4,3), dtype=float)
    alfaH   = math.acos(-1.0 / 3.0)     # 109.47°
    alfaHpl = 2.0 * math.pi / 3.0      # 120°
    distH   = 1

    alfaCOM = math.radians(117)
    alfaCO  = math.radians(121)
    alfaCOA = math.radians(115)

    distO  = 0.123
    distOA = 0.125
    distOM = 0.136

    central_atom, first_bonded_atom, second_bonded_atom, third_bonded_atom = xa

    sa = np.zeros(3)
    sb = np.zeros(3)
    sij = np.zeros(3)

    s6 = 0.5 * math.sqrt(3.0)

    if nht in (2, 3, 4, 8, 9):
        sij = central_atom - first_bonded_atom
        sb  = first_bonded_atom - second_bonded_atom

        rij = norm(sij)
        sij /= rij

        sa = np.cross(sij, sb)
        sa /= norm(sa)

        sb = np.cross(sa, sij)

    if nht == 1:
        sij = central_atom - first_bonded_atom
        sb  = central_atom - second_bonded_atom

        rij = norm(sij)
        rb  = norm(sb)

        sa = sij/rij + sb/rb
        sa /= norm(sa)

        xh[0][:] = central_atom + distH * sa

    elif nht == 2:
        xh[0][:] = central_atom + distH * math.sin(alfaH) * sb \
                   - distH * math.cos(alfaH) * sij

    elif nht == 3:
        xh[0][:] = central_atom - distH * math.sin(alfaHpl) * sb \
                   - distH * math.cos(alfaHpl) * sij

        xh[1][:] = central_atom + distH * math.sin(alfaHpl) * sb \
                   - distH * math.cos(alfaHpl) * sij

    elif nht == 4:
        xh[0][:] = central_atom + distH * math.sin(alfaH) * sb \
                   - distH * math.cos(alfaH) * sij

        xh[1][:] = central_atom - distH * math.sin(alfaH)*0.5 * sb \
                   + distH * math.sin(alfaH)*s6 * sa \
                   - distH * math.cos(alfaH) * sij

        # if not np.isnan(xh[2]).any():
        # Third H is only placed when 3 hydrogens are requested (e.g. NH3, CH3)
        if n_requested is None or n_requested >= 3:
            xh[2][:] = central_atom - distH * math.sin(alfaH)*0.5 * sb \
                       - distH * math.sin(alfaH)*s6 * sa \
                       - distH * math.cos(alfaH) * sij

    elif nht == 5:
        center = (first_bonded_atom + second_bonded_atom + third_bonded_atom) / 3.0
        dxc = central_atom - center
        xh[0][:] = central_atom + dxc * distH / norm(dxc)

    elif nht == 6:
        rBB = central_atom - 0.5*(first_bonded_atom + second_bonded_atom)
        bb  = norm(rBB)

        rCC1 = central_atom - first_bonded_atom
        rCC2 = central_atom - second_bonded_atom
        rNN  = np.cross(rCC1, rCC2)
        nn   = norm(rNN)

        xh[0][:] = central_atom + distH * (
            math.cos(alfaH/2.0) * rBB/bb +
            math.sin(alfaH/2.0) * rNN/nn
        )

        xh[1][:] = central_atom + distH * (
            math.cos(alfaH/2.0) * rBB/bb -
            math.sin(alfaH/2.0) * rNN/nn
        )

    elif nht == 7:
        gen_waterhydrogen(2, xa, xh, l_ref)

    elif nht == 10:
        gen_waterhydrogen(3, xa, xh, l_ref)

    elif nht == 11:
        gen_waterhydrogen(4, xa, xh, l_ref)

    elif nht == 8:
        xh[0][:] = central_atom - distOM * math.sin(alfaCOM) * sb \
                   - distOM * math.cos(alfaCOM) * sij

        xh[1][:] = central_atom + distOM * math.sin(alfaCOM) * sb \
                   - distOM * math.cos(alfaCOM) * sij

    elif nht == 9:
        xh[0][:] = central_atom - distO * math.sin(alfaCO) * sb \
                   - distO * math.cos(alfaCO) * sij

        xh[1][:] = central_atom + distOA * math.sin(alfaCOA) * sb \
                   - distOA * math.cos(alfaCOA) * sij

        xa2 = [
            copy_rvec(xh[1]),
            copy_rvec(central_atom),
            copy_rvec(first_bonded_atom),
            copy_rvec(second_bonded_atom),
        ]

        calc_h_pos(2, xa2, xh[2:], l_ref)

    else:
        raise ValueError(f"Invalid nht value: {nht}")

    return xh


# | nht | Geometry constructed             | Example functional group |
# | --- | -------------------------------- | ------------------------ |
# | 1   | One planar H                     | peptide, aromatic        |
# | 2   | One tetrahedral/dihedral H       | hydroxyl                 |
# | 3   | Two planar H                     | -NH2                     |
# | 4   | 2–3 tetrahedral H                | -CH3                     |
# | 5   | One tetrahedral (3 substituents) | tertiary carbon          |
# | 6   | Two tetrahedral                  | -CH2-                    |
# | 7   | 2 water H                        | H2O                      |
# | 8   | 2 carboxyl oxygens               | -COO-                    |
# | 9   | -COOH (2 O + 1 H)                | carboxylic acid          |
# | 10  | 3 water sites                    | water models             |
# | 11  | 4 water sites                    | water + lone pairs       |