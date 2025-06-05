#!/usr/bin/env python3
"""
Enumerate every non-isomorphic N-monomer cluster in a graphimine sheet.

Prerequisites (from classify_monomers.py):
  • graphimine_100k.xyz          – original coordinates
  • atom_monomer_map.csv         – atom → monomer_id table
  • monomer_atoms.json           – monomer_id → [atom_indices]

Outputs:
  • clusters.h5                  – HDF5 with /N=k/cluster_m datasets
  • (optional) *.xyz files       – if --xyz-folder is supplied
"""

from pathlib import Path
import argparse, json, csv, math, itertools, sys, gzip

import numpy as np
from scipy.spatial import cKDTree as KDTree
import networkx as nx
import h5py

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, unit=None):
        return iterable

# ---------- bond-geometry constants (same as before) ----------
C_N_IMINE   = 1.30     # Å
C_N_TOL     = 0.05
COVALENT_MAX = 1.7     # Å - max distance for any covalent bond
# --------------------------------------------------------------

def load_atom_table(csv_path):
    """Return lists: elements, coords[N,3], monomer_ids[N]."""
    elems, coords, mids = [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            elems.append(row["element"])
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
            mids.append(int(row["monomer_id"]))
    return (elems, np.asarray(coords, float), np.asarray(mids, int))

def load_monomer_atoms(json_path):
    with open(json_path) as fp:
        dat = json.load(fp)
    # keys → int, values → list[int]
    return {int(k): v for k, v in dat.items()}

# ---------- 1.  build monomer graph ------------------------------------------
def build_monomer_graph(elems, coords, mono_ids):
    heavy_idx = np.where(np.array(elems) != "H")[0]
    heavy_coords = coords[heavy_idx]
    kdt = KDTree(heavy_coords)
    # Look for any covalent bonds between monomers
    pairs = kdt.query_pairs(r=COVALENT_MAX)
    G = nx.Graph()
    
    # Only add monomers that are not -1 (unassigned)
    valid_mono_ids = np.unique(mono_ids[heavy_idx])
    valid_mono_ids = valid_mono_ids[valid_mono_ids >= 0]
    for mid in valid_mono_ids:
        G.add_node(mid)

    edge_count = 0
    inter_monomer_bonds = 0
    bond_distances = []
    
    for i_local, j_local in pairs:
        ia = heavy_idx[i_local]
        ja = heavy_idx[j_local]
        d = np.linalg.norm(coords[ia] - coords[ja])
        
        m1, m2 = mono_ids[ia], mono_ids[ja]
        if m1 != m2 and m1 >= 0 and m2 >= 0:
            inter_monomer_bonds += 1
            bond_distances.append(d)
            # Add edge for any reasonable covalent bond
            if d <= COVALENT_MAX:
                G.add_edge(m1, m2)
                edge_count += 1
    
    print(f"Found {inter_monomer_bonds} inter-monomer bonds")
    if bond_distances:
        print(f"Bond distance range: {min(bond_distances):.3f} - {max(bond_distances):.3f} Å")
    print(f"Added {edge_count} edges to graph")
    return G

# ---------- 2.  central monomer ----------------------------------------------
def find_central_monomer(monomer_atoms, coords):
    centroids = {mid: coords[ids].mean(axis=0) for mid, ids in monomer_atoms.items()}
    central_id = min(centroids, key=lambda k: np.linalg.norm(centroids[k]))
    return central_id

# ---------- 3.  DFS enumeration with isomorphism guard ------------------------
def unique_clusters_of_size(G, centre, N):
    """Yield sets of monomer IDs (size N) that are non-isomorphic."""
    # pre-compute canonical graphs to weed duplicates
    uniques = []
    stack = [(centre, {centre})]

    while stack:
        current, visited = stack.pop()
        if len(visited) == N:
            # build subgraph
            subG = G.subgraph(visited).copy()
            if not any(nx.is_isomorphic(subG, g) for g in uniques):
                uniques.append(subG)
                yield visited
            continue
        for neigh in sorted(G.neighbors(current)):
            if neigh in visited or len(visited) >= N:
                continue
            stack.append((neigh, visited | {neigh}))
    # store as attribute so we can reuse in bigger N (optional)
    unique_clusters_of_size._cache = getattr(unique_clusters_of_size, "_cache", {}) 
    unique_clusters_of_size._cache[N] = uniques

# ---------- 4.  write helpers -------------------------------------------------
def cluster_xyz_text(atom_indices, elements, coords, cluster_id):
    out = [str(len(atom_indices)), f"cluster {cluster_id}"]
    for idx in atom_indices:
        e = elements[idx]
        x, y, z = coords[idx]
        out.append(f"{e} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(out)

def h5_write_cluster(h5grp, cluster_idx, elements, coords):
    subgrp = h5grp.create_group(f"cluster_{cluster_idx}")
    subgrp.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
    # Store elements as fixed-length string without compression for simplicity
    subgrp.create_dataset("elements", data=np.array(elements, dtype='S2'))

# ---------- 5.  main orchestrator --------------------------------------------
def main(xyz_path, csv_path, json_path, N_values, out_h5, xyz_folder=None):
    print("loading tables…")
    elems, coords, mono_ids = load_atom_table(csv_path)
    monomer_atoms = load_monomer_atoms(json_path)

    print("building monomer graph…")
    G = build_monomer_graph(elems, coords, mono_ids)
    centre = find_central_monomer(monomer_atoms, coords)
    print(f"central monomer  = {centre}")
    print(f"graph            = {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    h5 = h5py.File(out_h5, "w")

    for N in N_values:
        print(f"\n=== Enumerating N = {N} ===")
        grp = h5.create_group(f"N={N}")
        for c_idx, cluster in enumerate(tqdm(unique_clusters_of_size(G, centre, N),
                                            unit="cluster")):
            # aggregate atoms
            atom_ids = sorted({a for mid in cluster for a in monomer_atoms[mid]})
            el_list = np.array([elems[i].encode() for i in atom_ids])
            coord_arr = coords[atom_ids]
            h5_write_cluster(grp, c_idx, el_list, coord_arr)
            if xyz_folder:
                xyz_text = cluster_xyz_text(atom_ids, elems, coords, c_idx)
                (xyz_folder / f"N{N}_cluster{c_idx}.xyz").write_text(xyz_text)

    h5.close()
    print(f"\nAll done ➜ {out_h5}")

# ---------- 6.  CLI -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract unique N-monomer clusters")
    p.add_argument("--xyz", default="graphimine_100k.xyz", help="full XYZ (not parsed, only for provenance)")
    p.add_argument("--csv", default="atom_monomer_map.csv", help="atom↔monomer table")
    p.add_argument("--json", default="monomer_atoms.json", help="monomer↔atoms table")
    p.add_argument("--N", required=True,
                   help="comma-sep list or range (e.g. 3,4,5 or 2-6)")
    p.add_argument("--h5", default="clusters.h5", help="output HDF5 file")
    p.add_argument("--xyz-folder", help="also dump XYZ files into this folder")
    args = p.parse_args()

    # parse N list/range
    if "-" in args.N:
        a, b = map(int, args.N.split("-"))
        Nvals = list(range(a, b + 1))
    else:
        Nvals = [int(n) for n in args.N.split(",")]

    xyz_path  = Path(args.xyz)
    csv_path  = Path(args.csv)
    json_path = Path(args.json)
    h5_path   = Path(args.h5)
    xyz_dir   = Path(args.xyz_folder) if args.xyz_folder else None
    if xyz_dir:
        xyz_dir.mkdir(parents=True, exist_ok=True)

    main(xyz_path, csv_path, json_path, Nvals, h5_path, xyz_dir)