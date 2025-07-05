#!/usr/bin/env python3
import math
import numpy as np
import argparse
from scipy.spatial import cKDTree as KDTree

# Bond lengths in Angstroms
C_C_RING = 1.39
C_N_IMINE = 1.28
C_N_SINGLE = 1.41
C_C_SINGLE = 1.48
BOND_CUTOFF = 1.7

def read_xyz(path):
    """Read XYZ file and return element list and coordinates array."""
    with open(path) as f:
        _ = f.readline()  # Skip atom count
        _ = f.readline()  # Skip comment
        lines = f.readlines()
    
    elements = [line.split()[0] for line in lines]
    coords = np.array([[float(x) for x in line.split()[1:4]] for line in lines])
    return elements, coords

def write_xyz(path, elems, coords, comment=""):
    """Write XYZ file with elements and coordinates."""
    with open(path, 'w') as f:
        f.write(f"{len(elems)}\n{comment}\n")
        for e, (x, y, z) in zip(elems, coords):
            f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n")

def load_monomer(path):
    """Load monomer template and center it at origin."""
    elems, coords = read_xyz(path)
    center = coords.mean(axis=0)
    return [(e, *(pt - center)) for e, pt in zip(elems, coords)]

def generate_hex_centers(G, a):
    """Generate hexagonal lattice centers within G generations."""
    centers = []
    for i in range(-G, G + 1):
        for j in range(max(-G, -i - G), min(G, -i + G) + 1):
            x = a * (i + 0.5 * j)
            y = a * (math.sqrt(3) / 2 * j)
            centers.append((x, y, 0.0))
    return centers

def build_heavy(template, centers):
    """Build heavy atom structure by placing template at each center."""
    elems = []
    coords = []
    
    for cx, cy, cz in centers:
        for e, dx, dy, dz in template:
            elems.append(e)
            coords.append([cx + dx, cy + dy, cz + dz])
    
    return elems, np.array(coords)

def cap_edges(elems, coords):
    """Add terminal groups (imine bonds, aldehydes, NH2) to edge atoms."""
    # Build neighbor graph
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=BOND_CUTOFF)
    
    neigh = {i: set() for i in range(len(elems))}
    for i, j in pairs:
        neigh[i].add(j)
        neigh[j].add(i)
    
    # Identify imine bonds (C=N double bonds)
    imine = []
    for i, j in pairs:
        if {elems[i], elems[j]} == {"C", "N"}:
            distance = np.linalg.norm(coords[i] - coords[j])
            if abs(distance - C_N_IMINE) < 0.07:
                imine.append((i, j))
    
    # Find terminal atoms
    termC = []
    termN = []
    
    for i in range(len(elems)):
        if elems[i] == "C" and len(neigh[i]) == 1:
            if not any(i in p for p in imine):
                termC.append(i)
        elif elems[i] == "N" and len(neigh[i]) == 1:
            if not any(i in p for p in imine):
                termN.append(i)
    
    # Start with existing atoms
    new_e = list(elems)
    new_c = coords.tolist()
    
    # Add H to each imine carbon
    for c_idx, n_idx in imine:
        # Ensure c_idx is carbon
        if elems[c_idx] == "N":
            c_idx, n_idx = n_idx, c_idx
        
        # Calculate perpendicular vector
        u = (coords[n_idx] - coords[c_idx]) / np.linalg.norm(coords[n_idx] - coords[c_idx])
        p = np.array([u[1], -u[0], 0.0])
        
        # Add hydrogen
        new_e.append("H")
        new_c.append((coords[c_idx] + p * 1.11).tolist())
    
    # Add aldehyde groups to terminal carbons
    for i in termC:
        j = next(iter(neigh[i]))
        
        # Direction vector from neighbor to terminal carbon
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        
        # Perpendicular vector
        p = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        
        # Add oxygen and hydrogen
        new_e.extend(["O", "H"])
        new_c.extend([
            (coords[i] + v * 1.215).tolist(),
            (coords[i] + p * 1.11).tolist()
        ])
    
    # Add NH2 groups to terminal nitrogens
    for i in termN:
        j = next(iter(neigh[i]))
        
        # Direction vector from neighbor to terminal nitrogen
        v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        
        # Perpendicular vector
        p = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        
        # Add two hydrogens
        for s in [1, -1]:
            new_e.append("H")
            new_c.append((coords[i] + p * 1.02 * s).tolist())
    
    return new_e, np.array(new_c), len(imine), len(termC)

def write_counts_table(filename, n_target, monomer_count, imine_count, aldehyde_count, script_type):
    """Write functional group counts to TSV file."""
    import os
    
    write_header = not os.path.exists("functional_group_counts.txt")
    
    with open("functional_group_counts.txt", 'a') as f:
        if write_header:
            f.write("Filename\tScript_Type\tG_Target\tMonomer_Count\tImine_Count\tAldehyde_Count\tTotal\n")
        f.write(f"{filename}\t{script_type}\t{n_target}\t{monomer_count}\t{imine_count}\t{aldehyde_count}\t{imine_count + aldehyde_count}\n")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("G_target", type=int, help="Target number of generations")
    args = parser.parse_args()
    
    # Calculate parameters
    template = load_monomer("graphimine_monomer.xyz")
    a = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
    
    # Generate all centers (no circular filtering)
    centers = generate_hex_centers(args.G_target, a)
    
    # Count monomers (number of centers)
    monomer_count = len(centers)
    
    # Build structure
    elems, coords = build_heavy(template, centers)
    elems2, coords2, imine_count, aldehyde_count = cap_edges(elems, coords)
    
    # Write output
    filename = f"G={args.G_target}_graphimine_hexagon.xyz"
    comment = f"G_target={args.G_target} hexagon"
    write_xyz(filename, elems2, coords2, comment)
    
    # Log counts and print summary
    write_counts_table(filename, args.G_target, monomer_count, imine_count, aldehyde_count, "hexagon")
    
    print(f"G_target={args.G_target}, atoms={len(elems2)}, centers={len(centers)}")
    print(f"Functional groups: {imine_count} imine, {aldehyde_count} aldehyde, {imine_count + aldehyde_count} total")