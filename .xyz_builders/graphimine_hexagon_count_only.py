#!/usr/bin/env python3
"""
Graphimine hexagon count-only script - computes monomer, imine, and aldehyde counts without writing XYZ files
"""

import math
import argparse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict

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

def count_functional_groups(elems, coords):
    """Count imine and aldehyde groups using same logic as cap_edges."""
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=BOND_CUTOFF)
    neigh = defaultdict(set)
    for i, j in pairs:
        neigh[i].add(j)
        neigh[j].add(i)

    # Find imine C=N bonds
    imine = [(i, j) for i, j in pairs 
             if {elems[i], elems[j]} == {"C", "N"} 
             and abs(np.linalg.norm(coords[i] - coords[j]) - C_N_IMINE) < 0.07]

    # Find terminal carbons (aldehydes)
    termC = [i for i, e in enumerate(elems) 
             if e == "C" and len(neigh[i]) == 1 and not any(i in p for p in imine)]

    return len(imine), len(termC)

def write_counts_table(filename, n_target, monomer_count, imine_count, aldehyde_count, script_type):
    """Write functional group counts to TSV file."""
    import os
    
    write_header = not os.path.exists("functional_group_counts.txt")
    
    with open("functional_group_counts.txt", 'a') as f:
        if write_header:
            f.write("Filename\tScript_Type\tG_Target\tMonomer_Count\tImine_Count\tAldehyde_Count\tTotal\n")
        f.write(f"{filename}\t{script_type}\t{n_target}\t{monomer_count}\t{imine_count}\t{aldehyde_count}\t{imine_count + aldehyde_count}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Count monomers and functional groups for hexagonal graphimine")
    parser.add_argument("G_target", type=int, help="Target number of generations")
    args = parser.parse_args()
    
    # Load monomer template
    try:
        template = load_monomer("graphimine_monomer.xyz")
    except FileNotFoundError:
        print("Error: graphimine_monomer.xyz not found")
        return
    
    # Calculate parameters
    a = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
    
    # Generate all centers (no circular filtering)
    centers = generate_hex_centers(args.G_target, a)
    
    # Count monomers (number of centers)
    monomer_count = len(centers)
    
    # Build structure for counting only
    elems, coords = build_heavy(template, centers)
    
    # Count functional groups
    imine_count, aldehyde_count = count_functional_groups(elems, coords)
    
    # Generate filename and log counts
    filename = f"G={args.G_target}_graphimine_hexagon.xyz"
    write_counts_table(filename, args.G_target, monomer_count, imine_count, aldehyde_count, "hexagon")
    
    # Print results
    print(f"G_target={args.G_target}, monomers={monomer_count}, imine={imine_count}, aldehyde={aldehyde_count}, total={imine_count + aldehyde_count}")

if __name__ == "__main__":
    main()