#!/usr/bin/env python3
"""
Add edge hydrogens/oxygens to a graphimine XYZ.

Input  (required) :
    graphimine_cluster.xyz        # heavy atoms only (C, N) – any size
    atom_monomer_map.csv          # from classify_monomers.py
    monomer_atoms.json            #          »            »

Output (new file):
    graphimine_cluster_patched.xyz   # now includes O + H atoms

Heavy deps: numpy, scipy, networkx           (pip install numpy scipy networkx)
No RDKit / OpenBabel required.
"""

import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree as KDTree

# ──────────────────────────────────────────────────────────────────────────────
# Tunables
C_N_IMINE = 1.30   # Å  C=N
C_N_TOL   = 0.07
C_C_RING  = 1.39   # Å  aromatic C–C
C_C_TOL   = 0.07
C_SINGLE  = 1.60   # Å  generic heavy-atom single bond
BOND_CACHE_RADIUS = 1.7

ALD_C_O   = 1.23   # Å  C=O
ALD_C_H   = 1.09   # Å  C–H  (aldehyde)
NH_LEN    = 1.01   # Å  N–H

# ──────────────────────────────────────────────────────────────────────────────
def load_xyz(xyz_path):
    with open(xyz_path) as f:
        n = int(f.readline())
        _ = f.readline()
        elems, coord = [], []
        for line in f:
            s, x, y, z = line.split()
            elems.append(s)
            coord.append([float(x), float(y), float(z)])
    return elems, np.asarray(coord, float)

def load_tables(csv_path, json_path):
    atom_data = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            idx = int(row['atom_index'])
            atom_data[idx] = {
                'element': row['element'],
                'coords': np.array([float(row['x']), float(row['y']), float(row['z'])]),
                'monomer_id': int(row['monomer_id'])
            }
    with open(json_path) as fp:
        atoms_of_mono = {int(k): v for k, v in json.load(fp).items()}
    return atom_data, atoms_of_mono

# ──────────────────────────────────────────────────────────────────────────────
def build_neighbor_list(coords):
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=BOND_CACHE_RADIUS)
    return pairs

def is_imine(pair_dist, e1, e2):
    return {e1, e2} == {"C", "N"} and abs(pair_dist - C_N_IMINE) <= C_N_TOL

def classify_linkers(monomer_atoms, elems, coords):
    """
    Return per-monomer dict:
        {"ald_c": [atom_ids], "amine_n": [atom_ids]}
    Uses: ring carbon has two ~1.39-Å C neighbours inside monomer.
          Any peripheral C with exactly one ring C neighbour = aldehyde C
          Any N   ''            one ring C neighbour = amine N
    """
    linker_map = {}
    coords_arr = coords
    # pre-compute fast KD search inside monomer
    for mid, atoms in monomer_atoms.items():
        ring = [a for a in atoms if elems[a] == "C"
                and sum(1 for b in atoms
                        if b != a and elems[b] == "C"
                        and abs(np.linalg.norm(coords_arr[a]-coords_arr[b])
                                - C_C_RING) < C_C_TOL) == 2]
        ring_set = set(ring)
        ald_c, am_n = [], []
        for a in atoms:
            if a in ring_set:
                continue
            # neighbour in same monomer?
            neigh_ring = any(b in ring_set and
                             np.linalg.norm(coords_arr[a]-coords_arr[b]) < C_SINGLE
                             for b in atoms)
            if not neigh_ring:
                continue
            if elems[a] == "C": ald_c.append(a)
            if elems[a] == "N": am_n.append(a)
        linker_map[mid] = {"ald_c": ald_c, "amine_n": am_n}
    return linker_map

def add_imine_hydrogens(elems, coords, imine_pairs, bond_len=1.09):
    """
    For each {C,N} imine pair:  add one H to the carbon, none to nitrogen.
    Assumes NO hydrogens are present yet → no duplicate risk.
    """
    new_xyz = []
    for c_idx, n_idx in imine_pairs:
        # identify which atom is carbon
        if elems[c_idx] == "N":
            c_idx, n_idx = n_idx, c_idx      # swap
        v = coords[c_idx] - coords[n_idx]    # point away from N
        v /= np.linalg.norm(v)
        h_pos = coords[c_idx] + v * bond_len
        elems.append("H")
        coords = np.vstack([coords, h_pos])
        new_xyz.append(f"H added to imine C{c_idx}")
    return elems, coords, new_xyz

# ──────────────────────────────────────────────────────────────────────────────
def add_h_and_o(elems, coords, monomer_id, linkers, neighbor_pairs):
    """
    Modify elems, coords in-place: append new O/H atoms
    and return a list of the new atom lines for XYZ comment.
    """
    added = []
    tree = KDTree(coords)  # to query neighbours fast for valence checks

    # Build quick mapping for C=N bonds
    imine_bonds = set()
    imine_pairs = []
    for i,j in neighbor_pairs:
        dist = np.linalg.norm(coords[i]-coords[j])
        if is_imine(dist, elems[i], elems[j]):
            imine_bonds.add(frozenset((i,j)))
            imine_pairs.append((i,j))

    # ── NEW: add one H to every imine carbon ────────────────────────
    elems, coords, imine_notes = add_imine_hydrogens(
            elems, coords, imine_pairs, bond_len=1.09)
    added.extend(imine_notes)

    for mid, lk in linkers.items():
        # dangling aldehydes: aldehyde C not in an imine bond
        for c in lk["ald_c"]:
            # Is this C in any C=N?
            if any(frozenset((c,n)) in imine_bonds for n in tree.query(coords[c], k=4)[1]):
                continue  # internal
            # else: add =O and –H
            ring_nb = min((b for b in tree.query(coords[c], k=6)[1]
                           if b != c and np.linalg.norm(coords[c]-coords[b]) < C_SINGLE),
                          key=lambda x: np.linalg.norm(coords[c]-coords[x]))
            v = coords[c] - coords[ring_nb]
            v /= np.linalg.norm(v)
            # oxygen
            o_pos = coords[c] + v * ALD_C_O
            elems.append("O")
            coords = np.vstack([coords, o_pos])
            monomer_id = np.append(monomer_id, mid)
            added.append(f"O added to C{c}")
            # hydrogen (slightly off-plane)
            # pick a vector roughly perpendicular to v
            perp = np.cross(v, np.array([0.0,0.0,1.0]))
            if np.linalg.norm(perp) < 1e-3:
                perp = np.cross(v, np.array([0.0,1.0,0.0]))
            perp /= np.linalg.norm(perp)
            h_pos = coords[c] + (v * 0.1 + perp * ALD_C_H)  # ~1.09 Å net
            elems.append("H")
            coords = np.vstack([coords, h_pos])
            monomer_id = np.append(monomer_id, mid)
            added.append(f"H added to C{c}")
        # dangling amines: N not in imine
        for n in lk["amine_n"]:
            if any(frozenset((c,n)) in imine_bonds for c in tree.query(coords[n], k=4)[1]):
                continue
            ring_nb = min((b for b in tree.query(coords[n], k=6)[1]
                           if b != n and np.linalg.norm(coords[n]-coords[b]) < C_SINGLE),
                          key=lambda x: np.linalg.norm(coords[n]-coords[x]))
            v = coords[n] - coords[ring_nb]
            v /= np.linalg.norm(v)
            perp = np.cross(v, np.array([0.0,0.0,1.0]))
            if np.linalg.norm(perp) < 1e-3:
                perp = np.cross(v, np.array([0.0,1.0,0.0]))
            perp /= np.linalg.norm(perp)
            for sign in (+1, -1):
                h_pos = coords[n] + v*NH_LEN + perp*0.2*sign
                elems.append("H")
                coords = np.vstack([coords, h_pos])
                monomer_id = np.append(monomer_id, mid)
                added.append(f"H added to N{n}")
    return elems, coords, monomer_id, added

# ──────────────────────────────────────────────────────────────────────────────
def write_xyz(out_path, elems, coords, comment="patched"):
    with open(out_path, "w") as f:
        f.write(f"{len(elems)}\n{comment}\n")
        for e,(x,y,z) in zip(elems, coords):
            f.write(f"{e} {x:.6f} {y:.6f} {z:.6f}\n")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: patch_edge_sites.py cluster.xyz atom_monomer_map.csv "
              "monomer_atoms.json out_patched.xyz")
        sys.exit(1)

    xyz_in, csv_tab, json_tab, xyz_out = map(Path, sys.argv[1:5])

    elems, coords = load_xyz(xyz_in)
    atom_data, mono_atoms = load_tables(csv_tab, json_tab)
    
    # Match cluster atoms to original structure by coordinates (within tolerance)
    cluster_to_original = {}
    original_to_cluster = {}
    tolerance = 1e-5
    
    for cluster_idx, (elem, pos) in enumerate(zip(elems, coords)):
        best_match = None
        best_dist = float('inf')
        
        for orig_idx, info in atom_data.items():
            if info['element'] == elem and orig_idx not in original_to_cluster:
                dist = np.linalg.norm(pos - info['coords'])
                if dist < best_dist and dist < tolerance:
                    best_dist = dist
                    best_match = orig_idx
        
        if best_match is not None:
            cluster_to_original[cluster_idx] = best_match
            original_to_cluster[best_match] = cluster_idx
    
    # Get monomers present in this cluster
    cluster_monomers = {}
    for cluster_idx, orig_idx in cluster_to_original.items():
        for mono_id, atom_list in mono_atoms.items():
            if orig_idx in atom_list:
                if mono_id not in cluster_monomers:
                    cluster_monomers[mono_id] = []
                cluster_monomers[mono_id].append(cluster_idx)
    
    neighbor_pairs = build_neighbor_list(coords)
    linker_info = classify_linkers(cluster_monomers, elems, coords)
    
    # Create monomer_id array for cluster atoms
    mono_ids = np.full(len(elems), -1, dtype=int)
    for cluster_idx, orig_idx in cluster_to_original.items():
        mono_ids[cluster_idx] = atom_data[orig_idx]['monomer_id']

    elems, coords, mono_ids, notes = add_h_and_o(elems, coords, mono_ids,
                                                 linker_info, neighbor_pairs)

    write_xyz(xyz_out, elems, coords, comment="with dangling sites capped")
    print(f"Wrote {xyz_out} with {len(notes)} atoms added.")