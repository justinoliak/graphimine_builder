#!/usr/bin/env python3
import math
import numpy as np
import argparse
from scipy.spatial import cKDTree as KDTree

# Bond length constants
C_C_RING, C_N_IMINE, C_N_SINGLE, C_C_SINGLE, BOND_CUTOFF = 1.39, 1.28, 1.41, 1.48, 1.7

def read_xyz(path):
    with open(path) as f:
        _ = f.readline()  # Skip atom count
        _ = f.readline()  # Skip comment
        lines = f.readlines()
    
    elements = [line.split()[0] for line in lines]
    coords = np.array([[float(x) for x in line.split()[1:4]] for line in lines])
    return elements, coords

def write_xyz(path, elems, coords, comment=""):
    with open(path, 'w') as f:
        f.write(f"{len(elems)}\n{comment}\n")
        for e, (x, y, z) in zip(elems, coords):
            f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n")

def load_monomer(path):
    elems, coords = read_xyz(path)
    center = coords.mean(axis=0)
    return [(e, *(pt - center)) for e, pt in zip(elems, coords)]

def generate_hex_centers(N, a):
    centers = []
    for i in range(-N, N+1):
        for j in range(max(-N, -i-N), min(N, -i+N)+1):
            x = a * (i + 0.5 * j)
            y = a * (math.sqrt(3) / 2 * j)
            z = 0.0
            centers.append((x, y, z))
    return centers

def build_heavy(template, centers):
    elems, coords = [], []
    for cx, cy, cz in centers:
        for e, dx, dy, dz in template:
            elems.append(e)
            coords.append([cx + dx, cy + dy, cz + dz])
    return elems, np.array(coords)

def cap_edges(elems, coords):
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=BOND_CUTOFF)
    
    # Build neighbor list
    neigh = {i: set() for i in range(len(elems))}
    for i, j in pairs:
        neigh[i].add(j)
        neigh[j].add(i)
    
    # Find imine bonds (C=N with specific length)
    imine = []
    for i, j in pairs:
        if {elems[i], elems[j]} == {"C", "N"}:
            if abs(np.linalg.norm(coords[i] - coords[j]) - C_N_IMINE) < 0.07:
                imine.append((i, j))
    
    # Find terminal carbons and nitrogens
    termC = []
    termN = []
    for i in range(len(elems)):
        if elems[i] == "C" and not any(i in p for p in imine) and len(neigh[i]) == 1:
            termC.append(i)
        elif elems[i] == "N" and not any(i in p for p in imine) and len(neigh[i]) == 1:
            termN.append(i)
    
    new_e = list(elems)
    new_c = coords.tolist()
    
    # Add hydrogens to imine bonds
    for c_idx, n_idx in imine:
        if elems[c_idx] == "N":
            c_idx, n_idx = n_idx, c_idx
        
        u = (coords[n_idx] - coords[c_idx]) / np.linalg.norm(coords[n_idx] - coords[c_idx])
        p = np.array([u[1], -u[0], 0.0])
        
        new_e.append("H")
        new_c.append((coords[c_idx] + p * 1.11).tolist())
    
    # Add aldehyde groups to terminal carbons
    for i in termC:
        j = next(iter(neigh[i]))
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        p = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        
        new_e.extend(["O", "H"])
        new_c.extend([
            (coords[i] + v * 1.215).tolist(),
            (coords[i] + p * 1.11).tolist()
        ])
    
    # Add hydrogens to terminal nitrogens
    for i in termN:
        j = next(iter(neigh[i]))
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        p_norm = np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        p = np.array([v[1], -v[0], 0.0]) / p_norm if p_norm > 0 else np.array([1, 0, 0])
        
        for s in [1, -1]:
            new_e.append("H")
            new_c.append((coords[i] + p * 1.02 * s).tolist())
    
    return new_e, np.array(new_c), len(imine), len(termC)

def write_counts_table(filename, g_target, imine_count, aldehyde_count, script_type):
    import os
    
    write_header = not os.path.exists("functional_group_counts.txt")
    
    with open("functional_group_counts.txt", 'a') as f:
        if write_header:
            f.write("Filename\tScript_Type\tG_Target\tImine_Count\tAldehyde_Count\tTotal\n")
        f.write(f"{filename}\t{script_type}\t{g_target}\t{imine_count}\t{aldehyde_count}\t{imine_count + aldehyde_count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("G_target", type=int)
    parser.add_argument("--scale", type=float, default=0.8)
    args = parser.parse_args()
    
    N_hex = math.ceil(args.G_target / args.scale)
    template = load_monomer("graphimine_monomer.xyz")
    a = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
    
    # Generate centers within circular constraint
    all_centers = generate_hex_centers(N_hex, a)
    centers = [(x, y, z) for x, y, z in all_centers if x*x + y*y <= (a * args.G_target)**2]
    
    elems, coords = build_heavy(template, centers)
    elems2, coords2, imine_count, aldehyde_count = cap_edges(elems, coords)
    
    filename = f"G={args.G_target}_graphimine.xyz"
    comment = f"G_target={args.G_target} N_hex={N_hex} scale={args.scale}"
    write_xyz(filename, elems2, coords2, comment)
    
    write_counts_table(filename, args.G_target, imine_count, aldehyde_count, "circular")
    
    print(f"G_target={args.G_target}, N_hex={N_hex}, scale={args.scale}, atoms={len(elems2)}, centers={len(centers)}")
    print(f"Functional groups: {imine_count} imine, {aldehyde_count} aldehyde, {imine_count + aldehyde_count} total")