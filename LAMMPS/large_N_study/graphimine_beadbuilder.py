#!/usr/bin/env python3
"""
Graphimine / Graphamide bead–spring sheet builder  —  **v1.4**
================================================================
▪ **Copies flag**   `--copies N` replicates the reference flake N times.
▪ **Auto‑scale copies**   If `--copies` omitted you may give `--target_beads`.
  The script then sets copies ≈ target_beads / flake_size × (0.40/ϕ) so
  bead count stays ~constant and high‑ϕ packing is easier.
▪ **High-density packing**   Up to 100,000 placement attempts per flake with
  multiple restarts (max 10). Maintains exact target density without box inflation.
  Can achieve higher ϕ values through persistence.
▪ **Molecule IDs** retained; cubic box by default; `--enforce2d` collapses z.
"""
import math, random, sys, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree as KDTree

SIGMA = 1.0                                  # bead diameter (reduced)
HEX_CNT = lambda g: 3 * g * (g + 1) + 1      # beads in perfect hexagon with g shells
g_from_N = lambda n: int(round((math.sqrt(12 * n - 3) - 3) / 6))

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def generate_hex_flake(N: int, b: float) -> np.ndarray:
    g = g_from_N(N)
    N_exact = HEX_CNT(g)
    if N_exact != N:
        print(f"[info] N {N} adjusted → perfect hexagon {N_exact}", file=sys.stderr)
        N = N_exact
    a = b
    coords = []
    for n in range(-g, g + 1):
        for m in range(-g, g + 1):
            if max(abs(n), abs(m), abs(-n - m)) <= g:
                x = a * (m + 0.5 * n)
                y = a * (math.sqrt(3) / 2 * n)
                coords.append([x, y, 0.0])
    return np.asarray(coords)


def bond_pairs(centres: np.ndarray, cut: float) -> List[Tuple[int, int]]:
    kd = KDTree(centres)
    return [tuple(sorted(p)) for p in kd.query_pairs(cut)]

# -----------------------------------------------------------------------------
# Packing routine  (high-density with multiple restarts)
# -----------------------------------------------------------------------------

def pack_flakes(flakes: List[np.ndarray], phi: float, b_len: float, max_restarts=10):
    bead_vol = math.pi * SIGMA ** 3 / 6
    total_beads = sum(len(f) for f in flakes)
    L = (total_beads * bead_vol / phi) ** (1 / 3)

    for restart in range(max_restarts):
        if restart > 0:
            print(f"[info] restart attempt {restart + 1}/{max_restarts}")
            
        kd = KDTree(np.empty((0, 3)))
        centres, bonds, flake_sizes = [], [], []
        offset = 0
        success = True

        for flake_idx, flake in enumerate(flakes):
            placed = False
            for attempt in range(100000):  # Increased from 20,000 to 100,000
                if attempt % 20000 == 0 and attempt > 0:
                    print(f"[info] flake {flake_idx + 1}/{len(flakes)}: {attempt} placement attempts...")
                    
                dx, dy, dz = (random.uniform(-L / 2, L / 2) for _ in range(3))
                theta = random.uniform(0, 2 * math.pi)
                R = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                              [math.sin(theta),  math.cos(theta), 0.0],
                              [0.0, 0.0, 1.0]])
                trial = flake.dot(R.T) + np.array([dx, dy, dz])
                if len(kd.data) == 0 or kd.query(trial, k=1)[0].min() >= 1.05 * b_len:
                    kd = KDTree(np.vstack([kd.data, trial]))
                    bonds.extend([(i + offset, j + offset) for i, j in bond_pairs(trial, 1.05 * b_len)])
                    centres.extend(trial.tolist())
                    flake_sizes.append(len(trial))
                    offset += len(trial)
                    placed = True
                    break
            if not placed:
                print(f"[info] failed to place flake {flake_idx + 1} after 100,000 attempts")
                success = False
                break

        if success:
            actual_phi = total_beads * bead_vol / (L ** 3)
            print(f"[info] packing successful: φ_actual = {actual_phi:.3f}")
            return np.asarray(centres), bonds, L, flake_sizes

    sys.exit(f"[error] packing failed after {max_restarts} restarts at φ={phi:.3f}; density may be too high")

# -----------------------------------------------------------------------------
# LAMMPS writer
# -----------------------------------------------------------------------------

def write_lammps(path: Path, centres: np.ndarray, bonds: List[Tuple[int, int]], L: float, enforce2d: bool, flake_sizes: List[int]):
    zlo, zhi = (0.0, 0.0) if enforce2d else (-L / 2, L / 2)
    with path.open("w") as f:
        f.write(f"Graphimine – {len(centres)} atoms, {len(bonds)} bonds\n\n")
        f.write(f"{len(centres)} atoms\n{len(bonds)} bonds\n\n")
        f.write("1 atom types\n1 bond types\n\n")
        f.write(f"{-L / 2:.6f} {L / 2:.6f} xlo xhi\n")
        f.write(f"{-L / 2:.6f} {L / 2:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 1.0\n\n")            # one bead type, mass = 1
        f.write("Atoms  # id mol type x y z\n\n")
        id_, start, mol = 0, 0, 0
        for sz in flake_sizes:
            mol += 1
            for (x, y, z) in centres[start:start + sz]:
                id_ += 1
                f.write(f"{id_} {mol} 1 {x:.6f} {y:.6f} {z:.6f}\n")
            start += sz
        f.write("\nBonds\n\n")
        for b_id, (i, j) in enumerate(bonds, 1):
            f.write(f"{b_id} 1 {i + 1} {j + 1}\n")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_cli():
    ap = argparse.ArgumentParser(description="Generate bead–spring graphimine sheets (v1.4)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--N", type=int, help="monodisperse flake size (beads)")
    sel.add_argument("--flory", type=float, metavar="p", help="Flory P(N)")

    ap.add_argument("--phi", type=float, required=True, help="target volume fraction 0<ϕ<1")
    ap.add_argument("--b", type=float, default=1.0, help="bond length / σ (default 1.0)")
    ap.add_argument("--copies", type=int, help="explicit flake copies (overrides --target_beads)")
    ap.add_argument("--target_beads", type=int, default=700, help="auto‑scale copies so beads≈target*(0.40/ϕ)")
    ap.add_argument("--enforce2d", action="store_true", help="write zlo=zhi=0 (strict monolayer)")
    ap.add_argument("--output", type=str, default="graphimine.data")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    P = parse_cli()
    random.seed(P.seed)
    if not (0.0 < P.phi < 1.0):
        sys.exit("[error] --phi must be between 0 and 1")

    # ---------------- flake list --------------------------------------------
    flakes: List[np.ndarray] = []
    if P.N is not None:
        flake = generate_hex_flake(P.N, P.b)
        if P.copies:
            copies = P.copies
        else:
            copies = max(1, int(round(P.target_beads / len(flake) * 0.40 / P.phi)))
        flakes.extend([flake] * copies)
    else:
        p = P.flory
        if not (0.0 < p < 1.0):
            sys.exit("[error] Flory p must be 0 < p < 1")
        bead_vol = math.pi * SIGMA ** 3 / 6
        target_vol = P.target_beads * bead_vol * 0.40 / P.phi
        current_vol = 0.0
        while current_vol < target_vol:
            N_geom = np.random.geometric(1 - p)
            flake = generate_hex_flake(N_geom, P.b)
            flakes.append(flake)
            current_vol += len(flake) * bead_vol

    # ---------------- pack + write -----------------------------------------
    centres, bonds, L, flake_sizes = pack_flakes(flakes, P.phi, P.b)
    write_lammps(Path(P.output), centres, bonds, L, P.enforce2d, flake_sizes)

    phi_actual = (math.pi * SIGMA ** 3 / 6 * len(centres)) / L ** 3
    status = "OK" if abs(phi_actual - P.phi) / P.phi < 0.02 else "WARN"
    print(f"[done] Wrote {P.output} — {len(centres)} beads, φ_actual = {phi_actual:.3f} ({status})")