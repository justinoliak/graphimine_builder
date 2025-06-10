#!/usr/bin/env python3
import math, numpy as np, argparse, random
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict
C_C_RING, C_N_IMINE, C_N_SINGLE, C_C_SINGLE, BOND_CUTOFF = 1.39, 1.28, 1.41, 1.48, 1.7; LATTICE_PARAM = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
def read_xyz(path):
    with open(path) as f: _ = f.readline(); _ = f.readline(); lines = f.readlines()
    return [line.split()[0] for line in lines], np.array([[float(x) for x in line.split()[1:4]] for line in lines])
def write_xyz(path, elems, coords, comment=""):
    with open(path, 'w') as f: f.write(f"{len(elems)}\n{comment}\n"); [f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n") for e, (x, y, z) in zip(elems, coords)]
def load_monomer(path): elems, coords = read_xyz(path); return [(e, *(pt-coords.mean(axis=0))) for e, pt in zip(elems, coords)]
def generate_hex_lattice(n_max, a): return [np.array([a * (i + 0.5 * j), a * (math.sqrt(3)/2 * j), 0.0]) for i in range(-n_max, n_max+1) for j in range(max(-n_max, -i-n_max), min(n_max, -i+n_max)+1)]
def check_connectivity(selected_indices, all_centers, threshold=LATTICE_PARAM*1.1): return len(selected_indices) <= 1 if len(selected_indices) < 3 else all(len(KDTree([all_centers[i] for i in selected_indices]).query_ball_point(all_centers[i], threshold)) >= 3 for i in selected_indices)
def check_connected_graph(selected_indices, all_centers, threshold=LATTICE_PARAM*1.1):
    if len(selected_indices) <= 1: return True
    selected_centers, n, tree = [all_centers[i] for i in selected_indices], len(selected_indices), KDTree([all_centers[i] for i in selected_indices])
    adj = {i: set(j for j in tree.query_ball_point(selected_centers[i], threshold) if i != j) for i in range(n)}; visited, queue = set([0]), [0]
    while queue: current = queue.pop(0); [visited.add(neighbor) or queue.append(neighbor) for neighbor in adj[current] if neighbor not in visited]
    return len(visited) == n
def random_lattice_selection(n_target, all_centers, max_attempts=100000):
    for attempt in range(max_attempts):
        selected_indices = random.sample(range(len(all_centers)), min(n_target, len(all_centers)))
        if check_connected_graph(selected_indices, all_centers) and check_connectivity(selected_indices, all_centers): return selected_indices
    return growth_based_selection(n_target, all_centers)
def growth_based_selection(n_target, all_centers, max_attempts=50000):
    n_total, tree = len(all_centers), KDTree(all_centers)
    for start_attempt in range(10):
        start_idx, selected = random.randint(0, n_total - 1), set(); selected.add(start_idx)
        start_neighbors = [n for n in tree.query_ball_point(all_centers[start_idx], LATTICE_PARAM * 1.1) if n != start_idx]
        if len(start_neighbors) >= 2: [selected.add(n) for n in random.sample(start_neighbors, min(3, len(start_neighbors)))]
        attempts = 0
        while len(selected) < n_target and attempts < max_attempts:
            attempts += 1; boundary = set(); [boundary.update(n for n in tree.query_ball_point(all_centers[idx], LATTICE_PARAM * 1.1) if n not in selected and n < n_total) for idx in selected]
            if not boundary: break
            candidates, added = list(boundary), False; random.shuffle(candidates)
            for candidate in candidates[:10]:
                if sum(1 for n in tree.query_ball_point(all_centers[candidate], LATTICE_PARAM * 1.1) if n in selected | {candidate}) >= 2: selected.add(candidate); added = True; break
            if not added and candidates: selected.add(random.choice(candidates))
        if len(selected) >= n_target: return list(selected)[:n_target]
    return list(selected)
def metropolis_selection(n_target, all_centers, temperature=1.0, n_steps=10000):
    n_total, selected = len(all_centers), growth_based_selection(n_target, all_centers)
    def energy(selection): return float('inf') if not check_connected_graph(selection, all_centers) else sum(100 if (n_conn := len(KDTree([all_centers[i] for i in selection]).query_ball_point(all_centers[i], LATTICE_PARAM * 1.1)) - 1) < 2 else 1 if n_conn == 2 else 0 for i in selection)
    current_energy = energy(selected)
    for step in range(n_steps):
        if random.random() < 0.5 and len(selected) > 3:
            old_idx, available = random.choice(selected), [i for i in range(n_total) if i not in selected]
            if available:
                new_idx, new_selected, new_energy = random.choice(available), [i for i in selected if i != old_idx] + [new_idx], energy([i for i in selected if i != old_idx] + [new_idx])
                if new_energy < current_energy or random.random() < math.exp(-(new_energy - current_energy) / temperature): selected, current_energy = new_selected, new_energy
    return selected
def build_structure(template, centers): elems, coords = [], []; [[elems.append(e), coords.append([cx + dx, cy + dy, cz + dz])] for cx, cy, cz in centers for e, dx, dy, dz in template]; return elems, np.array(coords)
def cap_edges(elems, coords):
    tree, pairs = KDTree(coords), KDTree(coords).query_pairs(r=BOND_CUTOFF); neigh = {i: set() for i in range(len(elems))}; [neigh[i].add(j) or neigh[j].add(i) for i, j in pairs]
    imine = [(i, j) for i, j in pairs if {elems[i], elems[j]} == {"C", "N"} and abs(np.linalg.norm(coords[i] - coords[j]) - C_N_IMINE) < 0.07]
    termC, termN = [i for i in range(len(elems)) if elems[i] == "C" and not any(i in p for p in imine) and len(neigh[i]) == 1], [i for i in range(len(elems)) if elems[i] == "N" and not any(i in p for p in imine) and len(neigh[i]) == 1]
    new_e, new_c = list(elems), coords.tolist()
    for c_idx, n_idx in imine: c_idx, n_idx = (n_idx, c_idx) if elems[c_idx] == "N" else (c_idx, n_idx); u = (coords[n_idx] - coords[c_idx]) / np.linalg.norm(coords[n_idx] - coords[c_idx]); p = np.array([u[1], -u[0], 0.0]); new_e.append("H"); new_c.append((coords[c_idx] + p * 1.11).tolist())
    for i in termC: j = next(iter(neigh[i])); v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j]); perp = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0])); new_e.extend(["O", "H"]); new_c.extend([(coords[i] + v * 1.215).tolist(), (coords[i] + perp * 1.11).tolist()])
    for i in termN: j = next(iter(neigh[i])); v = (coords[i] - coords[j]) / np.linalg.norm(coords[i] - coords[j]); perp = np.array([v[1], -v[0], 0.0]) / np.linalg.norm(np.array([v[1], -v[0], 0.0])); [new_e.append("H") or new_c.append((coords[i] + perp * 1.02 * sign).tolist()) for sign in (+1, -1)]
    return new_e, np.array(new_c), len(imine), len(termC)
def write_counts_table(filename, n_target, imine_count, aldehyde_count, script_type): import os; write_header = not os.path.exists("functional_group_counts.txt"); f = open("functional_group_counts.txt", 'a'); f.write("Filename\tScript_Type\tN_Target\tImine_Count\tAldehyde_Count\tTotal\n" if write_header else ""); f.write(f"{filename}\t{script_type}\t{n_target}\t{imine_count}\t{aldehyde_count}\t{imine_count + aldehyde_count}\n"); f.close()
def analyze_structure(selected_centers):
    if len(selected_centers) == 0: return {}
    tree, connectivity = KDTree(selected_centers), defaultdict(int)
    for center in selected_centers: connectivity[len(tree.query_ball_point(center, LATTICE_PARAM * 1.1)) - 1] += 1
    return connectivity
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("n_monomers", type=int); parser.add_argument("n_runs", type=int); parser.add_argument("--method", choices=['random', 'growth', 'metropolis'], default='growth'); args = parser.parse_args()
    template, all_centers = load_monomer("monomer_cc_nc.xyz"), generate_hex_lattice(math.ceil(math.sqrt(args.n_monomers * 10)), LATTICE_PARAM); print(f"Generated lattice with {len(all_centers)} possible sites\n")
    for run in range(1, args.n_runs + 1):
        print(f"{'='*60}\nGenerating structure {run}/{args.n_runs} with {args.n_monomers} monomers...\n{'='*60}")
        selected_indices = random_lattice_selection(args.n_monomers, all_centers) if args.method == 'random' else growth_based_selection(args.n_monomers, all_centers) if args.method == 'growth' else metropolis_selection(args.n_monomers, all_centers)
        selected_centers = [all_centers[i] for i in selected_indices]; print(f"Selected {len(selected_centers)} sites from lattice\n")
        connectivity = analyze_structure(selected_centers)
        if connectivity: print("Connectivity distribution:"); [print(f"  {n_conn} connections: {count} monomers") for n_conn, count in sorted(connectivity.items())]
        elems, coords = build_structure(template, selected_centers); elems_capped, coords_capped, imine_count, aldehyde_count = cap_edges(elems, coords)
        output_file = f"graphimine_n{args.n_monomers}_run{run:03d}.xyz"; write_xyz(output_file, elems_capped, coords_capped, f"Graphimine run {run}: {len(selected_centers)} monomers, {len(elems_capped)} atoms")
        write_counts_table(output_file, args.n_monomers, imine_count, aldehyde_count, "random"); print(f"Written to: {output_file}"); print(f"Functional groups: {imine_count} imine, {aldehyde_count} aldehyde, {imine_count + aldehyde_count} total")
        if connectivity: avg_conn = sum(k*v for k,v in connectivity.items()) / sum(connectivity.values()); print(f"Average connectivity: {avg_conn:.2f}")
        poorly_connected = sum(count for n_conn, count in connectivity.items() if n_conn < 2)
        if poorly_connected > 0: print(f"WARNING: {poorly_connected} monomers with <2 connections!")
        print()
    print(f"{'='*60}\nCompleted {args.n_runs} structures with {args.n_monomers} monomers each\n{'='*60}")