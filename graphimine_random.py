#!/usr/bin/env python3
"""
Graphimine random script that computes closest G for circular structures
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

def write_xyz(filename, elems_or_coords, coords_or_comment=None, comment=""):
    """Write XYZ file with elements and coordinates."""
    with open(filename, "w") as f:
        if coords_or_comment is None:
            # Old format: write_xyz(filename, coords, comment)
            coords = elems_or_coords
            f.write(f"{len(coords)}\n")
            f.write(f"{comment}\n")
            for coord in coords:
                f.write(f"C {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}\n")
        else:
            # New format: write_xyz(filename, elems, coords, comment)
            elems = elems_or_coords
            coords = coords_or_comment
            f.write(f"{len(elems)}\n")
            f.write(f"{comment}\n")
            for e, coord in zip(elems, coords):
                f.write(f"{e} {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}\n")

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
        # In perfect hexagonal lattice, each site has exactly 6 neighbors
        # But edge sites will have fewer
        adjacency[i] = neighbors
    
    return adjacency

def find_origin_center(centres):
    """Find the lattice site closest to the origin (0,0,0)."""
    distances = [np.linalg.norm(center) for center in centres]
    origin_idx = np.argmin(distances)
    return origin_idx, centres[origin_idx]

def print_adjacency_stats(adjacency):
    """Print statistics about the adjacency list."""
    neighbor_counts = [len(neighbors) for neighbors in adjacency.values()]
    unique_counts = sorted(set(neighbor_counts))
    
    print(f"\nAdjacency statistics:")
    print(f"Total sites: {len(adjacency)}")
    for count in unique_counts:
        num_sites = neighbor_counts.count(count)
        print(f"Sites with {count} neighbors: {num_sites}")
    print(f"Average neighbors per site: {sum(neighbor_counts)/len(neighbor_counts):.2f}")

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
    print(f"Added first neighbor: {first_neighbor}")
    
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
            print(f"No valid candidates found. Stopping at {len(selected)} monomers.")
            break
        
        # Randomly select one candidate
        new_monomer = random.choice(candidates)
        selected.add(new_monomer)
        
        if len(selected) % 100 == 0:  # Progress update
            print(f"Added monomer {len(selected)}: {new_monomer}")
    
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

def cap_edges(elems, coords):
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
    termN = [i for i, e in enumerate(elems)
             if e == "N" and len(neighbours[i]) == 1 and not any(i in p for p in imines)]

    new_elems  = list(elems)
    new_coords = coords.tolist()

    # Add H to each imine carbon
    for c_idx, n_idx in imines:
        if elems[c_idx] == "N":
            c_idx, n_idx = n_idx, c_idx  # ensure c_idx is carbon
        u = (coords[n_idx] - coords[c_idx])
        u /= np.linalg.norm(u)
        perp = np.array([u[1], -u[0], 0.2])  # slight out-of-plane tilt
        new_elems.append("H")
        new_coords.append((coords[c_idx] + perp * 1.11).tolist())

    # Aldehyde on terminal carbons
    for idx in termC:
        j = next(iter(neighbours[idx]))
        v = coords[idx] - coords[j]
        v /= np.linalg.norm(v)
        perp = np.array([v[1], -v[0], 0.2])
        new_elems.extend(["O", "H"])
        new_coords.extend([
            (coords[idx] + v * 1.215).tolist(),
            (coords[idx] + perp * 1.11).tolist(),
        ])

    # NH2 on terminal nitrogens
    for idx in termN:
        j = next(iter(neighbours[idx]))
        v = coords[idx] - coords[j]
        v /= np.linalg.norm(v)
        perp = np.array([v[1], -v[0], 0.2])
        for sign in (+1, -1):
            new_elems.append("H")
            new_coords.append((coords[idx] + perp * 1.02 * sign).tolist())

    return new_elems, np.array(new_coords), len(imines), len(termC)

def log_counts(filename, n_target, imine, aldehyde, method):
    import os, csv
    header = ["Filename", "Method", "N_Target", "Imine_Count", "Aldehyde_Count", "Total"]
    needs_head = not os.path.exists("functional_group_counts.txt")
    with open("functional_group_counts.txt", "a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if needs_head:
            w.writerow(header)
        w.writerow([filename, method, n_target, imine, aldehyde, imine + aldehyde])

def write_selected_structure(filename, selected_indices, centres, comment=""):
    """Write only the selected monomer centers to XYZ file."""
    selected_coords = [centres[i] for i in selected_indices]
    write_xyz(filename, selected_coords, comment)

def main():
    p = argparse.ArgumentParser(description="Compute closest G for circular graphimine structures")
    p.add_argument("N", type=int, help="target number of monomers")
    args = p.parse_args()
    
    G, N_actual = compute_closest_G_circular(args.N)
    
    print(f"Target N: {args.N}")
    print(f"Closest G: {G}")
    print(f"Actual N for G={G}: {N_actual}")
    print(f"Difference: {abs(args.N - N_actual)}")
    
    # Create hexagonal lattice 5 times larger than computed G
    lattice_size = G * 5
    centres = generate_hex_lattice(lattice_size)
    
    print(f"\nHexagonal lattice:")
    print(f"Lattice size (n_max): {lattice_size}")
    print(f"Total lattice sites: {len(centres)}")
    print(f"Lattice spacing: {LATTICE_A:.2f} Å")
    
    # Don't write the full lattice, only the final structure
    
    # Find origin center
    origin_idx, origin_coord = find_origin_center(centres)
    print(f"\nOrigin center:")
    print(f"Index: {origin_idx}")
    print(f"Coordinates: ({origin_coord[0]:.3f}, {origin_coord[1]:.3f}, {origin_coord[2]:.3f})")
    print(f"Distance from origin: {np.linalg.norm(origin_coord):.3f} Å")
    
    # Build adjacency list
    print(f"\nBuilding adjacency list...")
    adjacency = build_adjacency_list(centres)
    print_adjacency_stats(adjacency)
    
    # Show neighbors of origin center
    origin_neighbors = adjacency[origin_idx]
    print(f"\nOrigin center (index {origin_idx}) has {len(origin_neighbors)} neighbors:")
    print(f"Neighbor indices: {origin_neighbors}")
    
    # Check actual distances to neighbors
    print(f"\nDistances from origin center to its neighbors:")
    origin_pos = centres[origin_idx]
    for neighbor_idx in origin_neighbors:
        neighbor_pos = centres[neighbor_idx]
        distance = np.linalg.norm(neighbor_pos - origin_pos)
        print(f"  Neighbor {neighbor_idx}: {distance:.3f} Å")
    
    print(f"\nAdjacency cutoff distance: {LATTICE_A * 1.1:.3f} Å")
    print(f"Expected nearest neighbor distance in hex lattice: {LATTICE_A:.3f} Å")
    
    # Build constrained structure
    print(f"\nBuilding constrained structure with N={args.N}...")
    selected_monomers = build_constrained_structure(origin_idx, adjacency, centres, args.N)
    
    print(f"Successfully built structure with {len(selected_monomers)} monomers")
    
    # Load monomer template and build full structure
    try:
        template = load_monomer("graphimine_monomer.xyz")
        print(f"Loaded monomer template with {len(template)} atoms")
    except FileNotFoundError:
        print("Warning: graphimine_monomer.xyz not found, writing centers only")
        structure_filename = f"graphimine_n{args.N}_run001.xyz"
        structure_comment = f"Constrained graphimine structure - N={len(selected_monomers)} monomers"
        write_selected_structure(structure_filename, selected_monomers, centres, structure_comment)
        print(f"Wrote centers to: {structure_filename}")
        return
    
    # Get coordinates of selected centers
    selected_centres = [centres[i] for i in selected_monomers]
    
    # Build full atomic structure
    elems, coords = build_structure(template, selected_centres)
    elems, coords, n_imine, n_ald = cap_edges(elems, coords)
    
    # Write final structure
    structure_filename = f"graphimine_n{args.N}_run001.xyz"
    structure_comment = f"Graphimine flake - {len(selected_monomers)} monomers - constrained random growth"
    write_xyz(structure_filename, elems, coords, structure_comment)
    log_counts(structure_filename, len(selected_monomers), n_imine, n_ald, "constrained_random")
    
    print(f"Wrote full structure to: {structure_filename}")
    print(f"Total atoms: {len(elems)}, Imine groups: {n_imine}, Aldehyde groups: {n_ald}")

if __name__ == "__main__":
    main()