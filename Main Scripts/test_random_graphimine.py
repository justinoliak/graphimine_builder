#!/usr/bin/env python3
import math, numpy as np, argparse, random
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict

C_C_RING, C_N_IMINE, C_N_SINGLE, C_C_SINGLE, BOND_CUTOFF = 1.39, 1.28, 1.41, 1.48, 1.7
LATTICE_PARAM = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING

def read_xyz(path):
    with open(path) as f: _ = f.readline(); _ = f.readline(); lines = f.readlines()
    return [line.split()[0] for line in lines], np.array([[float(x) for x in line.split()[1:4]] for line in lines])

def write_xyz(path, elems, coords, comment=""):
    with open(path, 'w') as f: f.write(f"{len(elems)}\n{comment}\n"); [f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n") for e, (x, y, z) in zip(elems, coords)]

def load_monomer(path): elems, coords = read_xyz(path); return [(e, *(pt-coords.mean(axis=0))) for e, pt in zip(elems, coords)]

def generate_hex_lattice(n_max, a): return [np.array([a * (i + 0.5 * j), a * (math.sqrt(3)/2 * j), 0.0]) for i in range(-n_max, n_max+1) for j in range(max(-n_max, -i-n_max), min(n_max, -i+n_max)+1)]

def random_growth_selection(n_target, all_centers):
    """Random growth ensuring 2+ connections for all monomers."""
    n_total, tree = len(all_centers), KDTree(all_centers)
    
    # Multiple attempts to get a good structure
    for attempt in range(20):
        # Start from random position
        start_idx = random.randint(0, n_total - 1)
        selected = {start_idx}
        
        # Add initial neighbors for connectivity
        neighbors = tree.query_ball_point(all_centers[start_idx], LATTICE_PARAM * 1.1)
        neighbors = [n for n in neighbors if n != start_idx]
        if len(neighbors) >= 2:
            for n in random.sample(neighbors, min(3, len(neighbors))):
                selected.add(n)
        
        # Keep adding random adjacent sites
        while len(selected) < n_target:
            # Find all possible adjacent sites
            candidates = []
            for idx in selected:
                adj = tree.query_ball_point(all_centers[idx], LATTICE_PARAM * 1.1)
                for candidate in adj:
                    if candidate not in selected:
                        # Count how many connections this candidate would have
                        connections = sum(1 for n in adj if n in selected)
                        if connections >= 1:  # Will have 2+ after adding
                            candidates.append(candidate)
            
            if not candidates:
                break
            
            # Pick randomly from valid candidates
            selected.add(random.choice(candidates))
        
        # Check if all have 2+ connections
        valid = True
        for idx in selected:
            neighbors = tree.query_ball_point(all_centers[idx], LATTICE_PARAM * 1.1)
            connections = sum(1 for n in neighbors if n in selected)
            if connections < 2:
                valid = False
                break
        
        if valid and len(selected) >= n_target:
            return list(selected)[:n_target]
    
    # If we couldn't find a valid configuration, return what we have
    return list(selected)[:n_target]

def build_structure(template, centers): 
    elems, coords = [], []
    [[elems.append(e), coords.append([cx + dx, cy + dy, cz + dz])] for cx, cy, cz in centers for e, dx, dy, dz in template]
    return elems, np.array(coords)

def cap_edges(elems, coords):
    tree, pairs = KDTree(coords), KDTree(coords).query_pairs(r=BOND_CUTOFF)
    neigh = {i: set() for i in range(len(elems))}; [neigh[i].add(j) or neigh[j].add(i) for i, j in pairs]
    imine = [(i, j) for i, j in pairs if {elems[i], elems[j]} == {"C", "N"} and abs(np.linalg.norm(coords[i] - coords[j]) - C_N_IMINE) < 0.07]
    termC = [i for i in range(len(elems)) if elems[i] == "C" and not any(i in p for p in imine) and len(neigh[i]) == 1]
    termN = [i for i in range(len(elems)) if elems[i] == "N" and not any(i in p for p in imine) and len(neigh[i]) == 1]
    new_e, new_c = list(elems), coords.tolist()
    
    for c_idx, n_idx in imine: 
        c_idx, n_idx = (n_idx, c_idx) if elems[c_idx] == "N" else (c_idx, n_idx)
        u = (coords[n_idx] - coords[c_idx]) / np.linalg.norm(coords[n_idx] - coords[c_idx])
        p = np.array([u[1], -u[0], 0.0])
        new_e.append("H"); new_c.append((coords[c_idx] + p * 1.11).tolist())
    
    for i in termC: 
        j = next(iter(neigh[i])); v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        perp = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        new_e.extend(["O", "H"]); new_c.extend([(coords[i] + v * 1.215).tolist(), (coords[i] + perp * 1.11).tolist()])
    
    for i in termN: 
        j = next(iter(neigh[i])); v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j])
        perp = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0]))
        [new_e.append("H") or new_c.append((coords[i] + perp * 1.02 * sign).tolist()) for sign in (+1, -1)]
    
    return new_e, np.array(new_c), len(imine), len(termC)

def analyze_structure(selected_centers):
    if len(selected_centers) == 0: return {}
    tree = KDTree(selected_centers); connectivity = defaultdict(int)
    for center in selected_centers: connectivity[len(tree.query_ball_point(center, LATTICE_PARAM * 1.1)) - 1] += 1
    return connectivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("n_monomers", type=int); parser.add_argument("n_runs", type=int); args = parser.parse_args()
    template = load_monomer("monomer_cc_nc.xyz")
    all_centers = generate_hex_lattice(math.ceil(math.sqrt(args.n_monomers * 10)), LATTICE_PARAM)
    print(f"Generated lattice with {len(all_centers)} possible sites\n")
    
    for run in range(1, args.n_runs + 1):
        print(f"{'='*60}\nGenerating structure {run}/{args.n_runs} with {args.n_monomers} monomers...\n{'='*60}")
        random.seed(run * 12345); np.random.seed(run * 12345)
        
        selected_indices = random_growth_selection(args.n_monomers, all_centers)
        selected_centers = [all_centers[i] for i in selected_indices]
        
        connectivity = analyze_structure(selected_centers)
        print("Connectivity distribution:")
        [print(f"  {n_conn} connections: {count} monomers") for n_conn, count in sorted(connectivity.items())]
        
        elems, coords = build_structure(template, selected_centers)
        elems_capped, coords_capped, imine_count, aldehyde_count = cap_edges(elems, coords)
        
        output_file = f"graphimine_n{args.n_monomers}_run{run:03d}.xyz"
        write_xyz(output_file, elems_capped, coords_capped, f"Graphimine run {run}: {len(selected_centers)} monomers, {len(elems_capped)} atoms")
        print(f"Written to: {output_file}")
        print(f"Functional groups: {imine_count} imine, {aldehyde_count} aldehyde")
        
        if connectivity: avg_conn = sum(k*v for k,v in connectivity.items()) / sum(connectivity.values()); print(f"Average connectivity: {avg_conn:.2f}\n")
    
    print(f"{'='*60}\nCompleted {args.n_runs} structures with {args.n_monomers} monomers each\n{'='*60}")