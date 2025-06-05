#!/usr/bin/env python3
"""
Classify every TABTCA-based monomer in a 100 k-atom graphimine sheet
and mint a unique monomer ID for each.

Outputs:
  • atom_monomer_map.csv
  • monomer_atoms.json
"""

import json
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree as KDTree
import networkx as nx

# ------------------------------------------------------------
# 1. Parameters – tweak if your structure is distorted
# ------------------------------------------------------------
XYZ_PATH = Path("graphimine_100k.xyz")
C_C_RING    = 1.39          # Å   ideal aromatic C–C
C_C_TOL     = 0.02          # ±   tolerance for ring bonds (tightened)
C_N_IMINE   = 1.30          # Å   C=N      (double)
C_N_TOL     = 0.02          # tightened from 0.05
C_SINGLE_MAX = 1.60         # Å   generic single-bond cutoff

# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------
def load_xyz(path: Path):
    """Return elements list and N×3 coordinate array (Å)."""
    with path.open() as f:
        n_atoms = int(next(f).strip())
        _ = next(f)  # skip comment
        elems, coords = [], []
        for line in f:
            symbol, x, y, z = line.split()
            elems.append(symbol)
            coords.append([float(x), float(y), float(z)])
    return elems, np.asarray(coords, dtype=np.float64)

def build_neighbor_graph(coords, elems):
    """Return NetworkX graph of heavy-atom neighbours within cutoffs."""
    heavy_idx = [i for i, e in enumerate(elems) if e != "H"]
    heavy_coords = coords[heavy_idx]
    kdt = KDTree(heavy_coords)
    pairs = kdt.query_pairs(r=C_SINGLE_MAX)  # absolute max heavy-atom bond
    G = nx.Graph()
    for i in heavy_idx:
        G.add_node(i)
    for i_local, j_local in pairs:
        i_global = heavy_idx[i_local]
        j_global = heavy_idx[j_local]
        dist = np.linalg.norm(coords[i_global] - coords[j_global])
        G.add_edge(i_global, j_global, dist=dist)
    return G

def find_benzene_rings(G, elems):
    """Return list of 6-membered cycles that look like benzene rings."""
    rings = []
    # candidate edges ~1.39 Å between carbon atoms
    carbon_edges = [
        (u, v) for u, v, d in G.edges(data="dist")
        if elems[u] == elems[v] == "C" and abs(d - C_C_RING) <= C_C_TOL
    ]
    # Build C-C subgraph and search for 6-cycles
    CC = nx.Graph()
    CC.add_edges_from(carbon_edges)
    # Use cycle_basis then filter length-6 cycles
    for cycle in nx.cycle_basis(CC):
        if len(cycle) == 6:
            # Verify all internal edges are aromatic‐length
            ok = True
            for i in range(6):
                u, v = cycle[i], cycle[(i + 1) % 6]
                if not CC.has_edge(u, v):
                    ok = False
                    break
            if ok:
                rings.append(tuple(sorted(cycle)))
    # Deduplicate by sorted tuple
    rings = list(dict.fromkeys(rings))
    return rings

def ring_substituents(ring, G, elems, coords):
    """Return exactly 6 substituents (C or N) directly bonded to ring carbons."""
    linkers = set()
    for c in ring:
        for neigh in G.neighbors(c):
            if neigh in ring:
                continue
            dist = G.edges[c, neigh]["dist"]
            if dist <= C_SINGLE_MAX and elems[neigh] in {"C", "N"}:
                linkers.add(neigh)
                # Only add the direct substituent, not its partners
    # Expect exactly 6 linkers (one per ring carbon).
    return linkers

# ------------------------------------------------------------
# 3. Main routine
# ------------------------------------------------------------
def build_monomers():
    elems, coords = load_xyz(XYZ_PATH)
    print(f"Loaded {len(elems):,} atoms from {XYZ_PATH}")
    G = build_neighbor_graph(coords, elems)
    rings = find_benzene_rings(G, elems)
    print(f"Found {len(rings):,} benzene cores")

    monomer_atoms = []
    atom_mono_id  = np.full(len(elems), fill_value=-1, dtype=np.int32)

    valid_monomers = 0
    for m_id, ring in enumerate(rings):
        ring_set = set(ring)
        linkers  = ring_substituents(ring, G, elems, coords)
        
        # Validate: should have exactly 6 ring carbons + 6 linkers = 12 atoms
        if len(linkers) != 6:
            print(f"Warning: Ring {m_id} has {len(linkers)} linkers instead of 6, skipping")
            continue
            
        monomer = ring_set | linkers
        if len(monomer) != 12:
            print(f"Warning: Monomer {m_id} has {len(monomer)} atoms instead of 12, skipping")
            continue
            
        monomer_atoms.append(sorted(monomer))
        valid_monomers += 1
        for a in monomer:
            if atom_mono_id[a] != -1:
                # Overlap should not happen; warn if it does.
                print(f"Warning: atom {a} already classified")
            atom_mono_id[a] = len(monomer_atoms) - 1  # Use valid monomer count

    print(f"Valid monomers: {valid_monomers} out of {len(rings)} rings")
    
    # Report unclassified heavy atoms (should be none)
    unclassified = np.where((atom_mono_id == -1) & (np.array(elems) != "H"))[0]
    if len(unclassified):
        print(f"⚠️  {len(unclassified)} heavy atoms not mapped to any monomer")

    # ------------- write outputs -----------------
    write_csv(atom_mono_id, elems, coords)
    write_json(monomer_atoms)
    print("Done. Outputs: atom_monomer_map.csv, monomer_atoms.json")

def write_csv(atom_mono_id, elems, coords):
    with open("atom_monomer_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["atom_index", "element", "x", "y", "z", "monomer_id"])
        for idx, (e, xyz, mid) in enumerate(zip(elems, coords, atom_mono_id)):
            x, y, z = xyz
            w.writerow([idx, e, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", mid])

def write_json(monomer_atoms):
    data = {str(mid): atoms for mid, atoms in enumerate(monomer_atoms)}
    with open("monomer_atoms.json", "w") as fp:
        json.dump(data, fp, indent=2)

# ------------------------------------------------------------
if __name__ == "__main__":
    build_monomers()