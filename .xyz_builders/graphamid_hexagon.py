#!/usr/bin/env python3
"""Generate graphamid flakes with hexagonal boundary.
Places monomers on a hexagonal lattice within a hexagonal boundary and
adds appropriate terminating groups (amide bonds and aldehydes) at edges.

Usage
-----
python graphamid_hexagon.py G_TARGET

* G_TARGET: target number of generations (hexagon radius)
"""

import math
import argparse
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree as KDTree

# --------------------------------------------------
# Constants (all in Angstrom)
# --------------------------------------------------
C_C_RING   = 1.39  # aromatic C-C
C_N_AMIDE  = 1.35  # C-N amide bond
N_C_SINGLE = 1.45  # N-C single bond
C_C_SINGLE = 1.48  # C-C single
BOND_CUTOFF = 1.70  # generic heavy-atom cutoff for KD-tree bonding

# --------------------------------------------------
# Helper: basic XYZ I/O
# --------------------------------------------------

def read_xyz(path):
    """Return list(elems), ndarray(N,3) for an XYZ file."""
    with open(path) as fh:
        _ = fh.readline()            # atom count
        _ = fh.readline()            # comment
        lines = fh.readlines()
    elems  = [ln.split()[0] for ln in lines]
    coords = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines])
    return elems, coords


def write_xyz(path, elems, coords, comment=""):
    with open(path, "w") as fh:
        fh.write(f"{len(elems)}\n{comment}\n")
        for e, (x, y, z) in zip(elems, coords):
            fh.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n")


def load_monomer(path):
    elems, coords = read_xyz(path)
    centre = coords.mean(axis=0)
    return [(e, *(pt - centre)) for e, pt in zip(elems, coords)]


# --------------------------------------------------
# Lattice generation (hexagonal boundary)
# --------------------------------------------------

def generate_hex_centers(G, a):
    """Generate hexagonal lattice centers within G generations (hexagonal boundary)."""
    centers = []
    for i in range(-G, G + 1):
        for j in range(max(-G, -i - G), min(G, -i + G) + 1):
            x = a * (i + 0.5 * j)
            y = a * (math.sqrt(3) / 2 * j)
            centers.append((x, y, 0.0))
    return centers


# --------------------------------------------------
# Build full atom list & cap edges
# --------------------------------------------------

def build_structure(template, centres):
    elems, coords = [], []
    for cx, cy, cz in centres:
        for e, dx, dy, dz in template:
            elems.append(e)
            coords.append([cx + dx, cy + dy, cz + dz])
    return elems, np.array(coords)


def cap_edges(elems, coords):
    import random
    kd = KDTree(coords)
    pairs = kd.query_pairs(r=BOND_CUTOFF)
    neighbours = defaultdict(set)
    for i, j in pairs:
        neighbours[i].add(j)
        neighbours[j].add(i)

    # Identify amide bonds (C-N with specific distance range and carbon coordination)
    amides = [(i, j) if elems[i] == "C" else (j, i) for i, j in pairs
              if {elems[i], elems[j]} == {"C", "N"}
              and 1.3 < np.linalg.norm(coords[i] - coords[j]) < 1.5
              and len(neighbours[i if elems[i] == "C" else j]) == 2]

    termC = [i for i, e in enumerate(elems)
             if e == "C" and i not in {c for c, n in amides} and len(neighbours[i]) == 1]
    termN = [i for i, e in enumerate(elems)
             if e == "N" and i not in {n for c, n in amides} and len(neighbours[i]) == 1]

    new_elems  = list(elems)
    new_coords = coords.tolist()

    # Add C=O and N-H to amide groups
    for c, n in amides:
        # Simple approach: place C=O primarily in z direction to avoid ring conflicts
        z_sign = 1.0 if random.random() < 0.5 else -1.0
        # Mostly z-direction with small xy component
        v_cn = coords[n] - coords[c]
        v_cn /= np.linalg.norm(v_cn)
        xy_perp = np.array([-v_cn[1], v_cn[0], 0.0])
        xy_perp /= np.linalg.norm(xy_perp) if np.linalg.norm(xy_perp) > 0.01 else 1
        o_dir = 0.05 * xy_perp + 0.95 * np.array([0.0, 0.0, z_sign])
        o_dir /= np.linalg.norm(o_dir)
        
        # N-H direction: perpendicular to C-N in xy plane
        h_perp = np.array([-v_cn[1], v_cn[0], 0.0])
        h_perp /= np.linalg.norm(h_perp) if np.linalg.norm(h_perp) > 0.01 else 1
        
        new_elems.extend(["O", "H"])
        new_coords.extend([(coords[c] + o_dir * 1.21).tolist(),
                          (coords[n] + h_perp * 1.01).tolist()])

    # Aldehyde on terminal carbons
    for i in termC:
        j = next(iter(neighbours[i]))
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        # For aldehyde: C=O along v direction, C-H perpendicular to both v and z
        p_xy = np.array([v[1], -v[0], 0.0])
        p_xy /= np.linalg.norm(p_xy) if np.linalg.norm(p_xy) > 0.01 else 1
        new_elems.extend(["O", "H"])
        new_coords.extend([(coords[i] + v * 1.21).tolist(),
                          (coords[i] + p_xy * 1.11).tolist()])

    # NH2 on terminal nitrogens
    for i in termN:
        j = next(iter(neighbours[i]))
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        p = np.array([v[1], -v[0], 0.0])
        p /= np.linalg.norm(p) if np.linalg.norm(p) > 0.01 else 1
        for sign in [+1, -1]:
            new_elems.append("H")
            new_coords.append((coords[i] + p * 1.01 * sign).tolist())

    return new_elems, np.array(new_coords), len(amides), len(termC)


# --------------------------------------------------
# TSV logging
# --------------------------------------------------

def log_counts(filename, g_target, amide, aldehyde, method):
    import os, csv
    header = ["Filename", "Method", "G_Target", "Amide_Count", "Aldehyde_Count", "Total"]
    needs_head = not os.path.exists("amide_group_counts.txt")
    with open("amide_group_counts.txt", "a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if needs_head:
            w.writerow(header)
        w.writerow([filename, method, g_target, amide, aldehyde, amide + aldehyde])


# --------------------------------------------------
# Main CLI driver
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Graphamid hexagon generator")
    p.add_argument("G_target", type=int, help="target number of generations")
    args = p.parse_args()

    # Calculate lattice parameters
    a = C_C_RING + C_C_SINGLE + C_N_AMIDE + N_C_SINGLE + C_C_RING

    template = load_monomer("monomer_graphamid.xyz")
    centres = generate_hex_centers(args.G_target, a)
    
    elems, coords = build_structure(template, centres)
    elems, coords, n_amide, n_ald = cap_edges(elems, coords)

    fname = f"G={args.G_target}_graphamid_hexagon.xyz"
    write_xyz(fname, elems, coords,
              f"G_target={args.G_target} hexagon")
    log_counts(fname, args.G_target, n_amide, n_ald, "hexagon")
    print(f"G_target={args.G_target}, atoms={len(elems)}, centers={len(centres)}")
    print(f"Functional groups: {n_amide} amide, {n_ald} aldehyde, {n_amide + n_ald} total")


if __name__ == "__main__":
    main()