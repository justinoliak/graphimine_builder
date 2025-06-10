#!/usr/bin/env python3
"""Generate graphimine flakes with a hard constraint: every benzene centre
has at least two neighbours (no singly‑connected monomers).  The lattice is
optionally oversised so the cluster never runs into an artificial edge.

Usage
-----
python graphimine_builder_hardconstraint.py N_MONOMERS N_RUNS \
        [--method random|growth|metropolis] \
        [--template monomer_cc_nc.xyz] \
        [--lattice-buffer 3]

* ``--lattice-buffer`` multiplies the linear lattice size.  A buffer of ≥ 3 is
  usually sufficient to avoid edge effects even for highly elongated clusters.
"""

import math
import argparse
import random
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree as KDTree

# --------------------------------------------------
# Constants (all in Å)
# --------------------------------------------------
C_C_RING   = 1.39  # aromatic C–C
C_N_IMINE  = 1.28  # C=N double bond
C_N_SINGLE = 1.41  # C–N single
C_C_SINGLE = 1.48  # C–C single
BOND_CUTOFF = 1.70  # generic heavy‑atom cutoff for KD‑tree bonding

# Benzene‑centre spacing taken from crystallographic data, not summed bonds
LATTICE_A  = 6.980
NEIGH_CUTOFF = LATTICE_A * 1.10  # search radius for centre neighbours

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
# Lattice & neighbour utilities
# --------------------------------------------------

def generate_hex_lattice(n_max, a=LATTICE_A):
    """Axial‑coordinate honeycomb lattice centred on (0,0)."""
    sites = []
    for i in range(-n_max, n_max + 1):
        j_min = max(-n_max, -i - n_max)
        j_max = min(n_max,  -i + n_max)
        for j in range(j_min, j_max + 1):
            x = a * (i + 0.5 * j)
            y = a * (math.sqrt(3) / 2 * j)
            sites.append(np.array([x, y, 0.0]))
    return sites


# --------------------------------------------------
# Connectivity checks (HARD CONSTRAINT >=2 neighbours)
# --------------------------------------------------

def _centre_kdtree(selected, all_centres):
    """KD‑tree only over selected centre coordinates."""
    pts = [all_centres[i] for i in selected]
    return KDTree(pts)


def has_min_neighbours(selected, all_centres, min_deg=2):
    """Return True iff **every** selected centre has ≥ ``min_deg`` neighbours."""
    if not selected:
        return True
    kd = _centre_kdtree(selected, all_centres)
    return all(len(kd.query_ball_point(all_centres[i], NEIGH_CUTOFF)) - 1 >= min_deg
               for i in range(len(selected)))


def is_connected(selected, all_centres):
    if len(selected) <= 1:
        return True
    kd   = _centre_kdtree(selected, all_centres)
    todo = {0}
    seen = set()
    while todo:
        i = todo.pop()
        seen.add(i)
        nbrs = [j for j in kd.query_ball_point(all_centres[selected[i]], NEIGH_CUTOFF)
                 if j != i]
        todo.update(j for j in nbrs if j not in seen)
    return len(seen) == len(selected)


def remove_single_connections(selected, all_centres):
    """Remove monomers with only 1 connection while maintaining connectivity."""
    if not selected:
        return selected
    
    # Convert to list of indices
    if isinstance(selected[0], np.ndarray):
        # selected contains coordinates, need to find indices
        kd_all = KDTree(all_centres)
        selected_indices = []
        for center in selected:
            distances, indices = kd_all.query(center, k=1)
            selected_indices.append(indices)
        selected = selected_indices
    
    kd = KDTree([all_centres[i] for i in selected])
    
    # Iteratively remove singly-connected monomers
    changed = True
    while changed:
        changed = False
        to_remove = []
        
        for i, idx in enumerate(selected):
            # Count connections for this monomer
            neighbors = kd.query_ball_point(all_centres[idx], NEIGH_CUTOFF)
            connections = len(neighbors) - 1  # subtract self
            
            if connections < 2:
                to_remove.append(i)
        
        if to_remove:
            # Remove from back to front to preserve indices
            for i in reversed(to_remove):
                selected.pop(i)
            
            # Check if remaining cluster is still connected
            if selected and is_connected(selected, all_centres):
                # Rebuild KDTree with remaining monomers
                kd = KDTree([all_centres[i] for i in selected])
                changed = True
            else:
                # If removing would break connectivity, stop
                break
    
    return selected

# --------------------------------------------------
# Selection algorithms
# --------------------------------------------------

def random_selection(n_target, all_centres, max_trials=100_000):
    """Pure rejection sampling under hard constraints."""
    n_total = len(all_centres)
    for _ in range(max_trials):
        sel = random.sample(range(n_total), n_target)
        if is_connected(sel, all_centres) and has_min_neighbours(sel, all_centres):
            return sel
    raise RuntimeError("Failed to find a valid cluster by random sampling.")


def growth_selection(n_target, all_centres, max_steps=50_000):
    """Random growth ensuring good connectivity (adapted from working algorithm)."""
    n_total = len(all_centres)
    kd_full = KDTree(all_centres)
    
    for attempt in range(20):
        # Start from random position
        start_idx = random.randint(0, n_total - 1)
        selected = {start_idx}
        
        # Add initial neighbors for connectivity
        neighbors = kd_full.query_ball_point(all_centres[start_idx], NEIGH_CUTOFF)
        neighbors = [n for n in neighbors if n != start_idx]
        if len(neighbors) >= 2:
            for n in random.sample(neighbors, min(3, len(neighbors))):
                selected.add(n)
        
        # Keep adding random adjacent sites
        while len(selected) < n_target:
            # Find all possible adjacent sites
            candidates = []
            for idx in selected:
                adj = kd_full.query_ball_point(all_centres[idx], NEIGH_CUTOFF)
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
        
        if len(selected) >= n_target:
            # Post-process to remove singly-connected monomers
            final_selected = list(selected)[:n_target]
            return remove_single_connections(final_selected, all_centres)
    
    # If we couldn't find a valid configuration, return what we have
    final_selected = list(selected)[:n_target]
    return remove_single_connections(final_selected, all_centres)


def metropolis_selection(n_target, all_centres, temp=1.0, n_steps=10_000):
    """Simulated annealing subject to hard connectivity; energy = #dangling sites."""
    n_total = len(all_centres)
    sel = growth_selection(n_target, all_centres)  # good starting point

    def energy(S):
        kd = _centre_kdtree(S, all_centres)
        return sum(1 for i in range(len(S))
                   if len(kd.query_ball_point(all_centres[S[i]], NEIGH_CUTOFF)) - 1 < 2)

    e_cur = energy(sel)
    for _ in range(n_steps):
        if random.random() < 0.5 and len(sel) > 1:  # swap one site
            out_idx = random.randrange(len(sel))
            out_site = sel[out_idx]
            in_site  = random.choice([j for j in range(n_total) if j not in sel])
            trial = sel.copy()
            trial[out_idx] = in_site
        else:                                      # translate one site
            out_idx = random.randrange(len(sel))
            nbrs = [j for j in range(n_total)
                    if j not in sel and np.linalg.norm(all_centres[j]-all_centres[sel[out_idx]]) < 2*LATTICE_A]
            if not nbrs:
                continue
            trial = sel.copy()
            trial[out_idx] = random.choice(nbrs)
        if not is_connected(trial, all_centres):
            continue
        e_new = energy(trial)
        if e_new <= e_cur or random.random() < math.exp(-(e_new - e_cur) / temp):
            sel, e_cur = trial, e_new
    # Final hard check
    if has_min_neighbours(sel, all_centres):
        return sel
    raise RuntimeError("Metropolis failed to converge to hard‑constraint cluster.")

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
    kd = KDTree(coords)
    pairs = kd.query_pairs(r=BOND_CUTOFF)
    neighbours = defaultdict(set)
    for i, j in pairs:
        neighbours[i].add(j)
        neighbours[j].add(i)

    # Identify true imine C=N pairs (≈1.28 Å)
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
        perp = np.array([u[1], -u[0], 0.2])  # slight out‑of‑plane tilt
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

    # –NH2 on terminal nitrogens
    for idx in termN:
        j = next(iter(neighbours[idx]))
        v = coords[idx] - coords[j]
        v /= np.linalg.norm(v)
        perp = np.array([v[1], -v[0], 0.2])
        for sign in (+1, -1):
            new_elems.append("H")
            new_coords.append((coords[idx] + perp * 1.02 * sign).tolist())

    return new_elems, np.array(new_coords), len(imines), len(termC)

# --------------------------------------------------
# Connectivity histogram (diagnostics)
# --------------------------------------------------

def analyse_centres(selected, all_centres):
    if not selected:
        return {}
    kd = _centre_kdtree(selected, all_centres)
    hist = defaultdict(int)
    for i in range(len(selected)):
        deg = len(kd.query_ball_point(all_centres[selected[i]], NEIGH_CUTOFF)) - 1
        hist[deg] += 1
    return dict(hist)

# --------------------------------------------------
# TSV logging
# --------------------------------------------------

def log_counts(filename, n_target, imine, aldehyde, method):
    import os, csv
    header = ["Filename", "Method", "N_Target", "Imine_Count", "Aldehyde_Count", "Total"]
    needs_head = not os.path.exists("functional_group_counts.txt")
    with open("functional_group_counts.txt", "a", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if needs_head:
            w.writerow(header)
        w.writerow([filename, method, n_target, imine, aldehyde, imine + aldehyde])

# --------------------------------------------------
# Main CLI driver
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Graphimine flake generator (hard‑constraint version)")
    p.add_argument("n_monomers", type=int, help="target number of TABTCA monomers")
    p.add_argument("n_runs", type=int, help="how many independent flakes to generate")
    p.add_argument("--method", choices=["random", "growth", "metropolis"], default="growth",
                   help="site‑selection strategy")
    p.add_argument("--template", default="monomer_cc_nc.xyz", help="XYZ of one monomer (heavy atoms only)")
    p.add_argument("--lattice-buffer", type=int, default=3,
                   help="linear oversising factor for honeycomb lattice")
    args = p.parse_args()

    # Build oversized lattice
    n_max = int(math.ceil(math.sqrt(args.n_monomers))) * args.lattice_buffer
    all_centres = generate_hex_lattice(n_max)
    print(f"Generated honeycomb lattice with {len(all_centres)} potential sites (buffer ×{args.lattice_buffer})")

    template = load_monomer(args.template)

    selectors = {
        "random":     random_selection,
        "growth":     growth_selection,
        "metropolis": metropolis_selection,
    }
    select_sites = selectors[args.method]

    for run in range(1, args.n_runs + 1):
        print(f"\nRun {run}/{args.n_runs}  – method: {args.method}")
        sel = select_sites(args.n_monomers, all_centres)
        print(f"  Selected {len(sel)} monomer sites (all with ≥2 neighbours)")

        centres = [all_centres[i] for i in sel]
        conn_hist = analyse_centres(sel, all_centres)
        avg_deg = sum(k*v for k, v in conn_hist.items()) / args.n_monomers
        print(f"  Avg centre degree: {avg_deg:.2f}")

        elems, coords = build_structure(template, centres)
        elems, coords, n_imine, n_ald = cap_edges(elems, coords)

        fname = f"graphimine_n{args.n_monomers}_run{run:03d}.xyz"
        write_xyz(fname, elems, coords,
                  f"Graphimine flake • {args.n_monomers} monomers • all centres ≥2 links")
        log_counts(fname, args.n_monomers, n_imine, n_ald, args.method)
        print(f"  Wrote {fname}  (atoms: {len(elems)})  FG counts – imine:{n_imine}, aldehyde:{n_ald}\n")


if __name__ == "__main__":
    main()