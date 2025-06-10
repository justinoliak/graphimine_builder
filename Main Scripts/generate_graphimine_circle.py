#!/usr/bin/env python3
import math, numpy as np, argparse
from scipy.spatial import cKDTree as KDTree
C_C_RING, C_N_IMINE, C_N_SINGLE, C_C_SINGLE, BOND_CUTOFF = 1.39, 1.28, 1.41, 1.48, 1.7
def read_xyz(path):
    with open(path) as f: _ = f.readline(); _ = f.readline(); lines = f.readlines()
    return [line.split()[0] for line in lines], np.array([[float(x) for x in line.split()[1:4]] for line in lines])
def write_xyz(path, elems, coords, comment=""):
    with open(path,'w') as f: f.write(f"{len(elems)}\n{comment}\n"); [f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n") for e, (x,y,z) in zip(elems, coords)]
def load_monomer(path): elems, coords = read_xyz(path); return [(e, *(pt-coords.mean(axis=0))) for e,pt in zip(elems, coords)]
def generate_hex_centers(N, a): return [(a*(i+0.5*j), a*(math.sqrt(3)/2*j), 0.0) for i in range(-N, N+1) for j in range(max(-N, -i-N), min(N, -i+N)+1)]
def build_heavy(template, centers): elems, coords = [], []; [[elems.append(e), coords.append([cx+dx, cy+dy, cz+dz])] for cx,cy,cz in centers for e, dx, dy, dz in template]; return elems, np.array(coords)
def cap_edges(elems, coords):
    tree, pairs = KDTree(coords), KDTree(coords).query_pairs(r=BOND_CUTOFF); neigh = {i:set() for i in range(len(elems))}; [neigh[i].add(j) or neigh[j].add(i) for i,j in pairs]
    imine = [(i,j) for i,j in pairs if {elems[i],elems[j]}=={"C","N"} and abs(np.linalg.norm(coords[i]-coords[j])-C_N_IMINE)<0.07]
    termC, termN = [i for i in range(len(elems)) if elems[i]=="C" and not any(i in p for p in imine) and len(neigh[i])==1], [i for i in range(len(elems)) if elems[i]=="N" and not any(i in p for p in imine) and len(neigh[i])==1]
    new_e, new_c = list(elems), coords.tolist()
    for c_idx,n_idx in imine: c_idx,n_idx = (n_idx,c_idx) if elems[c_idx]=="N" else (c_idx,n_idx); u = (coords[n_idx]-coords[c_idx])/np.linalg.norm(coords[n_idx]-coords[c_idx]); p = np.array([u[1], -u[0], 0.0]); new_e.append("H"); new_c.append((coords[c_idx]+p*1.11).tolist())
    for i in termC: j = next(iter(neigh[i])); v = (coords[i]-coords[j])/np.linalg.norm(coords[i]-coords[j]); p = np.array([v[1], -v[0], 0.0])/np.linalg.norm(np.array([v[1], -v[0], 0.0])); new_e.extend(["O","H"]); new_c.extend([(coords[i]+v*1.215).tolist(), (coords[i]+p*1.11).tolist()])
    for i in termN: j = next(iter(neigh[i])); v = (coords[i]-coords[j])/np.linalg.norm(coords[i]-coords[j]); p = np.array([v[1], -v[0], 0.0])/np.linalg.norm(np.array([v[1], -v[0], 0.0])); [new_e.append("H") or new_c.append((coords[i]+p*1.02*s).tolist()) for s in [1,-1]]
    return new_e, np.array(new_c), len(imine), len(termC)
def write_counts_table(filename, n_target, imine_count, aldehyde_count, script_type): import os; write_header = not os.path.exists("functional_group_counts.txt"); f = open("functional_group_counts.txt", 'a'); f.write("Filename\tScript_Type\tN_Target\tImine_Count\tAldehyde_Count\tTotal\n" if write_header else ""); f.write(f"{filename}\t{script_type}\t{n_target}\t{imine_count}\t{aldehyde_count}\t{imine_count + aldehyde_count}\n"); f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("N_target", type=int); parser.add_argument("--scale", type=float, default=0.8); args = parser.parse_args()
    N_hex, template, a = math.ceil(args.N_target / args.scale), load_monomer("monomer_cc_nc.xyz"), C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
    centers = [(x,y,z) for x,y,z in generate_hex_centers(N_hex, a) if x*x+y*y <= (a*args.N_target)**2]; elems, coords = build_heavy(template, centers); elems2, coords2, imine_count, aldehyde_count = cap_edges(elems, coords)
    filename = f"N={args.N_target}_graphimine.xyz"; write_xyz(filename, elems2, coords2, f"N_target={args.N_target} N_hex={N_hex} scale={args.scale}")
    write_counts_table(filename, args.N_target, imine_count, aldehyde_count, "circular"); print(f"N_target={args.N_target}, N_hex={N_hex}, scale={args.scale}, atoms={len(elems2)}, centers={len(centers)}"); print(f"Functional groups: {imine_count} imine, {aldehyde_count} aldehyde, {imine_count + aldehyde_count} total")