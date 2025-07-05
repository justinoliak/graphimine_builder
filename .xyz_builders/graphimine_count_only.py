#!/usr/bin/env python3
"""
Graphimine count-only script - computes imine, aldehyde, and monomer counts without writing XYZ files
"""

import math
import argparse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict

# Constants (all in Angstrom)
C_C_RING   = 1.39  # aromatic C-C
C_N_IMINE  = 1.28  # C=N double bond
C_N_SINGLE = 1.41  # C-N single
C_C_SINGLE = 1.48  # C-C single
BOND_CUTOFF = 1.70  # generic heavy-atom cutoff for KD-tree bonding

# Benzene-centre spacing from summed bonds
LATTICE_A  = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING  # = 7.35 Angstrom

def compute_closest_G_circular(N):
    """
    For circular graphimine structures, compute the closest generation G
    that would produce approximately N monomers.
    
    For circular structures: N ≈ 3*G² + 3*G + 1
    Solving for G: G ≈ (-3 + sqrt(9 + 12*N - 12)) / 6
    """
    # Solve quadratic equation: 3*G² + 3*G + (1-N) = 0
    # Using quadratic formula: G = (-3 + sqrt(9 - 12*(1-N))) / 6
    discriminant = 9 + 12 * (N - 1)
    if discriminant < 0:
        return 0
    
    G_exact = (-3 + math.sqrt(discriminant)) / 6
    G_lower = math.floor(G_exact)
    G_upper = math.ceil(G_exact)
    
    # Calculate N for both G values
    N_lower = 3 * G_lower**2 + 3 * G_lower + 1
    N_upper = 3 * G_upper**2 + 3 * G_upper + 1
    
    # Return the G that gives N closest to target
    if abs(N - N_lower) <= abs(N - N_upper):
        return G_lower, N_lower
    else:
        return G_upper, N_upper

def generate_hex_lattice(n_max, a=LATTICE_A):
    """Generate hexagonal lattice of graphimine centers using axial coordinates."""
    sites = []
    for i in range(-n_max, n_max + 1):
        j_min = max(-n_max, -i - n_max)
        j_max = min(n_max,  -i + n_max)
        for j in range(j_min, j_max + 1):
            x = a * (i + 0.5 * j)
            y = a * (math.sqrt(3) / 2 * j)
            sites.append(np.array([x, y, 0.0]))
    return sites

def build_adjacency_list(centres, cutoff=None):
    """
    Build adjacency list for hexagonal lattice centers.
    Each center can connect to up to 6 neighbors in a hexagonal lattice.
    """
    if cutoff is None:
        cutoff = LATTICE_A * 1.1  # 10% tolerance for neighbor detection
    
    # Build KD-tree for efficient neighbor search
    kd_tree = KDTree(centres)
    
    adjacency = {}
    for i, center in enumerate(centres):
        # Find all neighbors within cutoff distance
        neighbors = kd_tree.query_ball_point(center, cutoff)
        # Remove self from neighbors
        neighbors = [n for n in neighbors if n != i]
        adjacency[i] = neighbors
    
    return adjacency

def find_origin_center(centres):
    """Find the lattice site closest to the origin (0,0,0)."""
    distances = [np.linalg.norm(center) for center in centres]
    origin_idx = np.argmin(distances)
    return origin_idx, centres[origin_idx]

def build_constrained_structure(origin_idx, adjacency, centres, target_N):
    """
    Build a structure starting from origin, with constraint that each new monomer
    must be adjacent to at least 2 existing monomers.
    """
    import random
    
    selected = {origin_idx}  # Start with origin
    
    # Step 1: Add one random neighbor of origin
    origin_neighbors = adjacency[origin_idx]
    if not origin_neighbors:
        raise RuntimeError("Origin has no neighbors!")
    
    first_neighbor = random.choice(origin_neighbors)
    selected.add(first_neighbor)
    
    # Step 2: Iteratively add monomers that connect to ≥2 existing monomers
    while len(selected) < target_N:
        # Find all candidates adjacent to current selection
        candidates = []
        for idx in selected:
            for neighbor in adjacency[idx]:
                if neighbor not in selected:
                    # Count how many existing monomers this candidate connects to
                    connections = sum(1 for n in adjacency[neighbor] if n in selected)
                    if connections >= 2:
                        candidates.append(neighbor)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        if not candidates:
            break
        
        # Randomly select one candidate
        new_monomer = random.choice(candidates)
        selected.add(new_monomer)
    
    return list(selected)

def read_xyz(path):
    """Return list(elems), ndarray(N,3) for an XYZ file."""
    with open(path) as fh:
        _ = fh.readline()            # atom count
        _ = fh.readline()            # comment
        lines = fh.readlines()
    elems  = [ln.split()[0] for ln in lines]
    coords = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines])
    return elems, coords

def load_monomer(path):
    elems, coords = read_xyz(path)
    centre = coords.mean(axis=0)
    return [(e, *(pt - centre)) for e, pt in zip(elems, coords)]

def build_structure(template, centres):
    elems, coords = [], []
    
    # Place the first monomer center at the origin
    centres_array = np.array(centres)
    first_center = centres_array[0]
    
    # Shift all centres so the first center is at origin
    centered_centres = centres_array - first_center
    
    for cx, cy, cz in centered_centres:
        for e, dx, dy, dz in template:
            elems.append(e)
            coords.append([cx + dx, cy + dy, cz + dz])
    return elems, np.array(coords)

def count_functional_groups(elems, coords):
    """Count imine and aldehyde groups using same logic as cap_edges."""
    kd = KDTree(coords)
    pairs = kd.query_pairs(r=BOND_CUTOFF)
    neighbours = defaultdict(set)
    for i, j in pairs:
        neighbours[i].add(j)
        neighbours[j].add(i)

    # Identify true imine C=N pairs (approximately 1.28 Angstrom)
    imines = [(i, j) for i, j in pairs
              if {elems[i], elems[j]} == {"C", "N"}
              and abs(np.linalg.norm(coords[i] - coords[j]) - C_N_IMINE) < 0.07]

    termC = [i for i, e in enumerate(elems)
             if e == "C" and len(neighbours[i]) == 1 and not any(i in p for p in imines)]

    return len(imines), len(termC)

def main():
    p = argparse.ArgumentParser(description="Count imine, aldehyde, and monomers for graphimine structures")
    p.add_argument("N", type=int, help="target number of monomers")
    args = p.parse_args()
    
    G, N_actual = compute_closest_G_circular(args.N)
    
    # Create hexagonal lattice 5 times larger than computed G
    lattice_size = G * 5
    centres = generate_hex_lattice(lattice_size)
    
    # Find origin center
    origin_idx, origin_coord = find_origin_center(centres)
    
    # Build adjacency list
    adjacency = build_adjacency_list(centres)
    
    # Build constrained structure
    selected_monomers = build_constrained_structure(origin_idx, adjacency, centres, args.N)
    
    # Load monomer template and build structure for counting
    try:
        template = load_monomer("graphimine_monomer.xyz")
    except FileNotFoundError:
        print("Error: graphimine_monomer.xyz not found")
        return
    
    # Get coordinates of selected centers
    selected_centres = [centres[i] for i in selected_monomers]
    
    # Build structure for counting only
    elems, coords = build_structure(template, selected_centres)
    
    # Count functional groups
    n_imine, n_ald = count_functional_groups(elems, coords)
    
    # Output tuple
    print(f"({len(selected_monomers)}, {n_imine}, {n_ald})")

if __name__ == "__main__":
    main()