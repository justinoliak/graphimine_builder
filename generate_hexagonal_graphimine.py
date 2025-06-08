#!/usr/bin/env python3
import math, sys, numpy as np
from scipy.spatial import cKDTree as KDTree

# Bond parameters
C_C_RING, C_N_IMINE, C_N_SINGLE, C_C_SINGLE, BOND_CUTOFF = 1.39, 1.28, 1.41, 1.48, 1.7

def read_xyz(path):
    with open(path) as f: _ = f.readline(); _ = f.readline(); lines = f.readlines()
    return [line.split()[0] for line in lines], np.array([[float(x) for x in line.split()[1:4]] for line in lines])

def write_xyz(path, elems, coords, comment=""):
    with open(path,'w') as f:
        f.write(f"{len(elems)}\n{comment}\n")
        for e, (x,y,z) in zip(elems, coords): f.write(f"{e} {x:10.6f} {y:10.6f} {z:10.6f}\n")

def load_monomer(path):
    elems, coords = read_xyz(path); center = coords.mean(axis=0)
    return [(e, *(pt-center)) for e,pt in zip(elems, coords)]

def generate_hex_centers(N, a):
    ctrs = []
    for i in range(-N, N+1):
        for j in range(max(-N, -i-N), min(N, -i+N)+1):
            x, y = a*(i+0.5*j), a*(math.sqrt(3)/2*j)
            ctrs.append((x,y,0.0))
    return ctrs

def build_heavy(template, centers):
    elems, coords = [], []
    for cx,cy,cz in centers:
        for e, dx, dy, dz in template: elems.append(e); coords.append([cx+dx, cy+dy, cz+dz])
    return elems, np.array(coords)

def cap_edges(elems, coords):
    tree = KDTree(coords); pairs = tree.query_pairs(r=BOND_CUTOFF)
    neigh = {i:set() for i in range(len(elems))}
    for i,j in pairs: neigh[i].add(j); neigh[j].add(i)
    imine = [(i,j) for i,j in pairs if {elems[i],elems[j]}=={"C","N"} and abs(np.linalg.norm(coords[i]-coords[j])-C_N_IMINE)<0.07]
    termC = [i for i in range(len(elems)) if elems[i]=="C" and not any(i in p for p in imine) and len(neigh[i])==1]
    termN = [i for i in range(len(elems)) if elems[i]=="N" and not any(i in p for p in imine) and len(neigh[i])==1]
    new_e, new_c = list(elems), coords.tolist()
    
    for c_idx,n_idx in imine:
        if elems[c_idx]=="N": c_idx,n_idx = n_idx,c_idx
        u = (coords[n_idx]-coords[c_idx]); u = u/np.linalg.norm(u)
        p = np.array([u[1], -u[0], 0.0])
        d = (-0.5*u + math.sqrt(3)/2*p); d = d/np.linalg.norm(d)
        new_e.append("H"); new_c.append((coords[c_idx]+d*1.08).tolist())
    
    for i in termC:
        j = next(iter(neigh[i])); v = (coords[i]-coords[j]); v = v/np.linalg.norm(v)
        new_e.append("O"); new_c.append((coords[i]+v*1.215).tolist())
        perp = np.array([v[1], -v[0], 0.0]); perp = perp/np.linalg.norm(perp)
        new_e.append("H"); new_c.append((coords[i]+perp*1.11).tolist())
    
    for i in termN:
        j = next(iter(neigh[i])); v = (coords[i]-coords[j]); v = v/np.linalg.norm(v)
        perp = np.array([v[1], -v[0], 0.0]); perp = perp/np.linalg.norm(perp)
        for sign in (+1,-1): new_e.append("H"); new_c.append((coords[i]+perp*1.02*sign).tolist())
    
    return new_e, np.array(new_c)

if len(sys.argv) != 2: print("Usage: python3 generate_hexagonal_graphimine.py N"); sys.exit(1)
N = int(sys.argv[1])
template = load_monomer("monomer_cc_nc.xyz")
a = C_C_RING + C_C_SINGLE + C_N_IMINE + C_N_SINGLE + C_C_RING
centers = [(x,y,z) for x,y,z in generate_hex_centers(N, a) if x*x+y*y <= (a*N*0.9)**2]
elems, coords = build_heavy(template, centers)
elems2, coords2 = cap_edges(elems, coords)
write_xyz(f"N={N}_graphimine.xyz", elems2, coords2, f"patched N={N} circle 90deg caps")
print(f"Output: N={N}_graphimine.xyz, total atoms: {len(elems2)}, centers: {len(centers)}")