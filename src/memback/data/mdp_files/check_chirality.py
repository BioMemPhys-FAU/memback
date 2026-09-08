#!/usr/bin/env python3
"""check_chirality.py -- detect (and optionally fix) R/S chirality defects."""
import sys, os, glob, math

USAGE = """\
check_chirality.py -- detect (and optionally fix) R/S chirality defects in a
.gro file, using the +/-120 deg dihedral restraints found in toppar/*.itp.

For every stereocenter it computes the signed volume of the three heavy
substituents and reports one of:
  ok         correct sign, well formed tetrahedron
  wrong      healthy center, but inverted handedness
  flat       center (nearly) coplanar with its substituents, sign meaningless
  collin     substituent tips collinear, plane undefined -> skipped, never fixed

With --fix, 'wrong' and 'flat' centers are repaired by lifting the center atom
onto the correct face of the substituent plane and reseating its hydrogen.
Substituents and their tails are never moved. Each fix is re-verified.

usage:
  python check_chirality.py <conf.gro> [--fix] [-o out.gro] [--toppar DIR]

options:
  -h, --help     show this message and exit
  --fix          repair defects and write a new .gro (default: detect only)
  -o FILE        output file for --fix (default: <conf>_chirfix.gro)
  --toppar DIR   directory holding the .itp files (default: toppar)

notes:
  velocities, header and box line are preserved; only the coordinates of the
  repaired center atoms and their hydrogens are rewritten. A short restrained
  minimisation is recommended after --fix.\
"""

FLAT_TOL = 0.15     # A, min |height| of center above substituent plane
VMIN     = 0.5      # A^3, min |signed volume|
SIN_TOL  = 0.15     # collinearity of the three substituent tips
TARGET_H = 0.5      # A, height to lift a flat center to
BOND_H   = 1.09     # A, C-H bond length


def sub(a, b):   return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
def add(a, b):   return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def dot(a, b):   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def scale(a, s): return [a[0]*s, a[1]*s, a[2]*s]
def norm(a):     return math.sqrt(dot(a, a))
def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def unit(a):
    n = norm(a)
    return [a[0]/n, a[1]/n, a[2]/n]

def min_image(d, box):
    out = []
    for i in range(3):
        if box[i] > 0:
            out.append(d[i] - box[i]*round(d[i]/box[i]))
        else:
            out.append(d[i])
    return out

def signed_volume(A, B, D):
    return dot(A, cross(B, D))


def read_itp(path):
    itps = {}
    mol, block, names = None, None, None
    for raw in open(path):
        line = raw.split(';')[0].strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            block = line.strip('[] ').strip().lower()
            if block == 'moleculetype':
                mol = None
            continue
        if block is None:
            continue
        tok = line.split()
        if block == 'moleculetype':
            mol = tok[0]
            itps[mol] = {'atoms': {}, 'bonds': [], 'dihres': []}
            names = itps[mol]['atoms']
        elif mol is None:
            continue
        elif block == 'atoms' and len(tok) >= 5:
            names[int(tok[0])] = tok[4]
        elif block == 'bonds' and len(tok) >= 2:
            itps[mol]['bonds'].append((int(tok[0]), int(tok[1])))
        elif block == 'dihedral_restraints' and len(tok) >= 6:
            itps[mol]['dihres'].append((int(tok[0]), int(tok[1]),
                                        int(tok[2]), int(tok[3]), float(tok[5])))
    return itps


def read_toppar(topdir):
    itps = {}
    files = sorted(glob.glob(os.path.join(topdir, '*.itp')))
    if not files:
        sys.exit("no .itp files found in %s" % topdir)
    for f in files:
        itps.update(read_itp(f))
    return itps


def prepare_target_chirals(itps):
    target = {}
    for mol, dat in itps.items():
        names = dat['atoms']
        out = []
        for (i, j, k, l, ang) in dat['dihres']:
            if abs(ang) != 120.0:
                continue
            if not all(x in names for x in (i, j, k, l)):
                continue
            h = None
            for (a, b) in dat['bonds']:
                if a == k:
                    other = b
                elif b == k:
                    other = a
                else:
                    continue
                if names[other].startswith('H'):
                    h = names[other]
                    break
            out.append({'center': names[k],
                        'sign_atoms': (names[i], names[j], names[l]),
                        'h': h,
                        'want_sign': -1 if ang < 0 else 1})
        if out:
            target[mol] = out
    return target


def read_gro(path):
    lines = open(path).read().splitlines()
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2+natoms]
    box = [float(x)*10.0 for x in lines[2+natoms].split()[:3]]
    coords, keys, names = [], [], []
    for ln in atom_lines:
        keys.append(ln[0:10])
        names.append(ln[10:15].strip())
        coords.append([float(ln[20:28])*10.0,
                       float(ln[28:36])*10.0,
                       float(ln[36:44])*10.0])
    return lines, natoms, atom_lines, box, coords, keys, names


def residues(keys, natoms):
    res, start = [], 0
    for i in range(1, natoms):
        if keys[i] != keys[i-1]:
            res.append((start, i))
            start = i
    res.append((start, natoms))
    return res


def analyse(coords, ci, ai, bi, di, box):
    """Return (status, V, h, n_hat, A, B, D) in the center-local frame."""
    c = coords[ci]
    A = min_image(sub(coords[ai], c), box)
    B = min_image(sub(coords[bi], c), box)
    D = min_image(sub(coords[di], c), box)
    V = signed_volume(A, B, D)

    e1, e2 = sub(B, A), sub(D, A)
    n = cross(e1, e2)
    nn = norm(n)
    if nn < 1e-8 or norm(e1) < 1e-8 or norm(e2) < 1e-8:
        return 'COLLINEAR', V, 0.0, None, A, B, D
    sin_t = norm(cross(unit(e1), unit(e2)))
    if sin_t < SIN_TOL:
        return 'COLLINEAR', V, 0.0, None, A, B, D

    n_hat = scale(n, 1.0/nn)
    h = -dot(A, n_hat)                      # signed height of center
    return None, V, h, n_hat, A, B, D


def classify(status, V, h, want_sign):
    if status == 'COLLINEAR':
        return 'COLLINEAR'
    if abs(h) < FLAT_TOL or abs(V) < VMIN:
        return 'FLAT'
    if (1 if V > 0 else -1) != want_sign:
        return 'WRONG'
    return 'OK'


def fix_center(coords, ci, hi, A, B, D, h, n_hat, want_sign):
    """Lift center onto the correct face of the substituent plane, reseat H."""
    foot = scale(n_hat, -h)                                  # local
    probe = add(foot, n_hat)
    Vp = signed_volume(sub(A, probe), sub(B, probe), sub(D, probe))
    n_ok = n_hat if (1 if Vp > 0 else -1) == want_sign else scale(n_hat, -1.0)

    new_local = add(foot, scale(n_ok, max(abs(h), TARGET_H)))
    us = [unit(sub(t, new_local)) for t in (A, B, D)]
    dirv = scale(add(add(us[0], us[1]), us[2]), -1.0)
    dirv = unit(dirv)

    c0 = coords[ci]
    new_c = add(c0, new_local)
    new_h = add(c0, add(new_local, scale(dirv, BOND_H)))
    return new_c, new_h


def write_gro(path, lines, natoms, coords, changed):
    out = lines[:2]
    for i in range(natoms):
        ln = lines[2+i]
        if i in changed:
            x, y, z = coords[i]
            ln = ln[:20] + "%8.3f%8.3f%8.3f" % (x/10.0, y/10.0, z/10.0) + ln[44:]
        out.append(ln)
    out += lines[2+natoms:]
    open(path, 'w').write("\n".join(out) + "\n")


def main():
    args = sys.argv[1:]
    if not args or '-h' in args or '--help' in args:
        print(USAGE)
        return
    gro = args[0]
    do_fix = '--fix' in args
    topdir = 'toppar'
    if '--toppar' in args:
        topdir = args[args.index('--toppar')+1]
    out = os.path.splitext(gro)[0] + '_chirfix.gro'
    if '-o' in args:
        out = args[args.index('-o')+1]

    itps = read_toppar(topdir)
    target = prepare_target_chirals(itps)
    if not target:
        sys.exit("no +/-120 dihedral restraints found in %s" % topdir)

    lines, natoms, atom_lines, box, coords, keys, names = read_gro(gro)

    stats, missing, changed = {}, {}, set()
    for (s, e) in residues(keys, natoms):
        resname = keys[s][5:10].strip()
        if resname not in target:
            continue
        idx = {}
        for i in range(s, e):
            idx[names[i]] = i
        for ch in target[resname]:
            need = (ch['center'],) + ch['sign_atoms'] + (ch['h'],)
            if any(nm is None or nm not in idx for nm in need):
                missing[resname] = missing.get(resname, 0) + 1
                continue
            ci = idx[ch['center']]
            ai, bi, di = (idx[nm] for nm in ch['sign_atoms'])
            hi = idx[ch['h']]

            st, V, h, n_hat, A, B, D = analyse(coords, ci, ai, bi, di, box)
            tag = classify(st, V, h, ch['want_sign'])
            key = (resname, ch['center'])
            stats.setdefault(key, {'OK': 0, 'WRONG': 0, 'FLAT': 0, 'COLLINEAR': 0,
                                   'FIXED': 0, 'FAILED': 0})
            stats[key][tag] += 1

            if do_fix and tag in ('WRONG', 'FLAT'):
                nc, nh = fix_center(coords, ci, hi, A, B, D, h, n_hat, ch['want_sign'])
                coords[ci], coords[hi] = nc, nh
                changed.add(ci)
                changed.add(hi)
                st2, V2, h2, _, _, _, _ = analyse(coords, ci, ai, bi, di, box)
                if classify(st2, V2, h2, ch['want_sign']) == 'OK':
                    stats[key]['FIXED'] += 1
                else:
                    stats[key]['FAILED'] += 1

    print("%-8s %-6s %8s %8s %8s %10s %8s %8s" %
          ("resname", "center", "total", "ok", "wrong", "flat", "collin", "fixed"))
    tot = {'OK': 0, 'WRONG': 0, 'FLAT': 0, 'COLLINEAR': 0, 'FIXED': 0, 'FAILED': 0}
    for (resname, center) in sorted(stats):
        c = stats[(resname, center)]
        n = c['OK'] + c['WRONG'] + c['FLAT'] + c['COLLINEAR']
        for k in tot:
            tot[k] += c[k]
        print("%-8s %-6s %8d %8d %8d %10d %8d %8d" %
              (resname, center, n, c['OK'], c['WRONG'], c['FLAT'],
               c['COLLINEAR'], c['FIXED']))
    n = tot['OK'] + tot['WRONG'] + tot['FLAT'] + tot['COLLINEAR']
    print("-" * 70)
    print("total centers   : %d" % n)
    if n:
        print("correct         : %d (%.2f%%)" % (tot['OK'], 100.0*tot['OK']/n))
        print("wrong sign      : %d" % tot['WRONG'])
        print("flat/degenerate : %d" % tot['FLAT'])
        print("collinear tips  : %d (skipped)" % tot['COLLINEAR'])
        if not do_fix and tot['WRONG'] or tot['FLAT']:
            print("Re-run check_chirality.py with --fix option to fix wrong sign and flat defects.")
    for r, k in missing.items():
        print("warning: %s missing restraint atoms in %d residues" % (r, k))

    if do_fix:
        print("fixed           : %d" % tot['FIXED'])
        print("failed          : %d" % tot['FAILED'])
        if changed:
            write_gro(out, lines, natoms, coords, changed)
            print("wrote %s (%d atoms modified)" % (out, len(changed)))
            print("Energy minimisation is recommended before running the next step.")
        else:
            print("nothing to fix, no file written")


if __name__ == '__main__':
    main()